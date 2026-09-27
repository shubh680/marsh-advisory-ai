"""
Policy PDF ingestion pipeline.
Extracts text from PDF preserving page numbers, chunks text,
generates embeddings, and indexes chunks into Qdrant vector database.
"""
import uuid
import re
import logging
from typing import List, Dict, Any
from pathlib import Path
from pypdf import PdfReader
import pdfplumber
from qdrant_client.http import models as qmodels
from django.conf import settings

from services.qdrant_service import get_qdrant_client
from services.embedding_service import embed_batch, get_embedding_dimension

logger = logging.getLogger(__name__)


def extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF file preserving 1-indexed page numbers.
    Returns a list of dicts: [{'page_number': 1, 'text': '...'}]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    reader = PdfReader(str(path))
    pages_data = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        page_text = page.extract_text() or ''
        # Clean extra whitespace
        cleaned_text = re.sub(r'[ \t]+', ' ', page_text).strip()
        if cleaned_text:
            pages_data.append({
                'page_number': page_num,
                'text': cleaned_text
            })

    return pages_data


def chunk_text_by_paragraphs(
    pages_data: List[Dict[str, Any]],
    document_id: str,
    document_name: str,
    target_chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Splits page text into coherent chunks while preserving page numbers and metadata.
    """
    chunks = []

    for page in pages_data:
        page_number = page['page_number']
        raw_text = page['text']

        # Split page into sections or paragraphs
        paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [raw_text]

        current_chunk = ""
        current_section = "General"

        for para in paragraphs:
            # Check for simple section headers (e.g. "Section 1", "Coverage:", all caps, etc.)
            first_line = para.split('\n')[0].strip()
            if len(first_line) < 60 and (first_line.isupper() or any(k in first_line.lower() for k in ['section', 'benefit', 'coverage', 'exclusion', 'clause', 'schedule', 'table'])):
                current_section = first_line

            if len(current_chunk) + len(para) <= target_chunk_size:
                current_chunk += ("\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append({
                        'document_id': str(document_id),
                        'document_name': document_name,
                        'page_number': page_number,
                        'chunk_text': current_chunk.strip(),
                        'section': current_section
                    })
                    # Keep a tail overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                    current_chunk = (overlap_text + "\n" + para).strip()
                else:
                    # Single very long paragraph
                    for i in range(0, len(para), target_chunk_size - overlap):
                        part = para[i:i + target_chunk_size].strip()
                        if part:
                            chunks.append({
                                'document_id': str(document_id),
                                'document_name': document_name,
                                'page_number': page_number,
                                'chunk_text': part,
                                'section': current_section
                            })
                    current_chunk = ""

        if current_chunk:
            chunks.append({
                'document_id': str(document_id),
                'document_name': document_name,
                'page_number': page_number,
                'chunk_text': current_chunk.strip(),
                'section': current_section
            })

    return chunks


def ensure_qdrant_collection(client, collection_name: str, vector_dim: int):
    """
    Ensures that the target Qdrant collection exists with proper vector configurations,
    and ensures that a payload index exists on document_id for efficient filtering.
    """
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        logger.info(f"Creating Qdrant collection '{collection_name}' with dim {vector_dim}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_dim,
                distance=qmodels.Distance.COSINE
            )
        )

    # Ensure payload index on document_id for metadata filtering
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD
        )
    except Exception as e:
        logger.debug(f"Payload index on document_id: {e}")


def extract_tables_from_pdf(file_path: str) -> Dict[int, List[str]]:
    """
    Uses pdfplumber to detect tables and convert them to context-aware markdown.
    Returns {page_number: [markdown_table_1, ...]} preserving headers with every chunk.
    """
    tables_by_page: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                tables = page.extract_tables()
                if not tables:
                    continue
                page_tables = []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    md = _table_to_markdown(table)
                    if md:
                        page_tables.append(md)
                if page_tables:
                    tables_by_page[page_num] = page_tables
    except Exception as e:
        logger.warning(f"pdfplumber table extraction failed for {file_path}: {e}")
    return tables_by_page


def _table_to_markdown(table: list) -> str:
    """Converts a pdfplumber table to markdown preserving headers, rows, and exact values."""
    if not table or not table[0]:
        return ""
    cleaned = []
    for row in table:
        cleaned_row = [(cell or "").strip().replace("\n", " ") for cell in row]
        if any(cleaned_row):
            cleaned.append(cleaned_row)
    if len(cleaned) < 2:
        return ""
    headers = cleaned[0]
    num_cols = len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * num_cols) + " |",
    ]
    for row in cleaned[1:]:
        padded = (row + [""] * num_cols)[:num_cols]
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


def chunk_tables_with_context(
    tables_by_page: Dict[int, List[str]],
    document_id: str,
    document_name: str,
    target_chunk_size: int = 800,
) -> List[Dict[str, Any]]:
    """
    Creates chunks from extracted tables.  Large tables are split into
    groups of rows but the header row always travels with each group.
    """
    chunks: List[Dict[str, Any]] = []
    for page_num, tables in tables_by_page.items():
        for table_md in tables:
            lines = table_md.split("\n")
            if len(lines) < 3:
                continue
            header_line = lines[0]
            separator = lines[1]
            data_rows = lines[2:]

            if len(table_md) <= target_chunk_size:
                chunks.append({
                    'document_id': str(document_id),
                    'document_name': document_name,
                    'page_number': page_num,
                    'chunk_text': table_md,
                    'section': 'Benefit Table',
                })
            else:
                preamble_len = len(header_line) + len(separator) + 2
                group: List[str] = []
                group_size = preamble_len
                for row in data_rows:
                    if group_size + len(row) + 1 > target_chunk_size and group:
                        chunks.append({
                            'document_id': str(document_id),
                            'document_name': document_name,
                            'page_number': page_num,
                            'chunk_text': "\n".join([header_line, separator] + group),
                            'section': 'Benefit Table',
                        })
                        group = []
                        group_size = preamble_len
                    group.append(row)
                    group_size += len(row) + 1
                if group:
                    chunks.append({
                        'document_id': str(document_id),
                        'document_name': document_name,
                        'page_number': page_num,
                        'chunk_text': "\n".join([header_line, separator] + group),
                        'section': 'Benefit Table',
                    })
    return chunks


def enrich_chunks_with_context(
    chunks: List[Dict[str, Any]],
    document_name: str,
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    LLM contextual enrichment (Pillar 1).
    Prepends a 2-sentence standardised underwriting context header.
    Stores result as ``enriched_text``; original ``chunk_text`` is untouched.
    """
    from services.llm_client import generate

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        numbered = []
        for j, chunk in enumerate(batch):
            snippet = chunk['chunk_text'][:400]
            page = chunk.get('page_number', 1)
            numbered.append(f"[{j + 1}] Page {page}:\n{snippet}")

        prompt = (
            f"Document: {document_name}\n\n"
            + "\n\n".join(numbered)
            + "\n\nFor each numbered excerpt above, write ONE line:\n"
            "[N] <2-sentence context in standard commercial insurance / "
            "underwriting terminology describing what medical benefits, "
            "financial limits, or coverage conditions are mentioned>\n\n"
            "Use terms like: inpatient hospitalization, room rent, ICU, "
            "pre/post hospitalization expenses, day-care procedures, "
            "ambulance coverage, cumulative bonus, restoration benefit, "
            "waiting period, sub-limits, co-payment, sum insured, "
            "cashless network, critical illness.\n"
            "Do NOT invent benefits not in the text."
        )

        try:
            response = generate(
                prompt=prompt,
                system_prompt=(
                    "You are an insurance document normalizer. "
                    "Output ONLY numbered context lines, nothing else."
                ),
                temperature=0.1,
                max_tokens=1000,
            )
            context_map: Dict[int, str] = {}
            for line in response.strip().split("\n"):
                m = re.match(r'\[(\d+)]\s*(.*)', line.strip())
                if m:
                    context_map[int(m.group(1)) - 1] = m.group(2).strip()

            for j, chunk in enumerate(batch):
                ctx = context_map.get(j, "")
                if ctx:
                    chunk['enriched_text'] = (
                        f"[{document_name} | Page {chunk.get('page_number', 1)}] "
                        f"{ctx}\n{chunk['chunk_text']}"
                    )
                else:
                    chunk['enriched_text'] = chunk['chunk_text']
        except Exception as e:
            logger.warning(f"Chunk enrichment failed for batch {i}: {e}")
            for chunk in batch:
                chunk['enriched_text'] = chunk['chunk_text']

    return chunks


def ingest_policy_document(policy_doc) -> Dict[str, Any]:
    """
    Full pipeline to ingest a PolicyDocument model instance into Qdrant.
    1. Extract text with page numbers
    2. Chunk text
    3. Generate embeddings
    4. Index points into Qdrant
    5. Update document status
    """
    from pitch.models import PolicyDocument

    policy_doc.processing_status = PolicyDocument.STATUS_PROCESSING
    policy_doc.save(update_fields=['processing_status'])

    try:
        file_path = policy_doc.file.path
        pages_data = extract_pdf_pages(file_path)
        if not pages_data:
            raise ValueError("No extractable text found in uploaded PDF.")

        # Pillar 2a: paragraph-based text chunks
        text_chunks = chunk_text_by_paragraphs(
            pages_data=pages_data,
            document_id=policy_doc.document_id,
            document_name=policy_doc.document_name,
        )

        # Pillar 2b: table-aware chunks (headers preserved)
        tables_by_page = extract_tables_from_pdf(file_path)
        table_chunks = chunk_tables_with_context(
            tables_by_page=tables_by_page,
            document_id=policy_doc.document_id,
            document_name=policy_doc.document_name,
        )

        chunks = text_chunks + table_chunks
        if not chunks:
            raise ValueError("Document produced 0 text chunks.")

        # Pillar 1: contextual enrichment (LLM context headers)
        chunks = enrich_chunks_with_context(chunks, policy_doc.document_name)

        # Embed using enriched_text for better retrieval
        embed_texts = [c.get('enriched_text', c['chunk_text']) for c in chunks]
        embeddings = embed_batch(embed_texts)
        dim = len(embeddings[0])

        # Qdrant client & collection
        client = get_qdrant_client()
        collection_name = getattr(settings, 'QDRANT_COLLECTION_NAME', 'policy_documents')
        ensure_qdrant_collection(client, collection_name, vector_dim=dim)

        # Build Qdrant points (store both original + enriched text)
        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = uuid.uuid4().hex
            payload = {
                'document_id': str(chunk['document_id']),
                'document_name': chunk['document_name'],
                'page_number': chunk['page_number'],
                'chunk_text': chunk['chunk_text'],
                'enriched_text': chunk.get('enriched_text', chunk['chunk_text']),
                'section': chunk['section'],
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert into Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        # Invalidate BM25 cache so next search rebuilds with new data
        from services.policy_retrieval import invalidate_bm25_cache
        invalidate_bm25_cache()

        # Update database record
        policy_doc.page_count = len(pages_data)
        policy_doc.chunk_count = len(chunks)
        policy_doc.processing_status = PolicyDocument.STATUS_INDEXED
        policy_doc.error_message = None
        policy_doc.save(update_fields=['page_count', 'chunk_count', 'processing_status', 'error_message'])

        return {
            'success': True,
            'document_id': policy_doc.document_id,
            'pages': len(pages_data),
            'chunks': len(chunks)
        }

    except Exception as e:
        logger.exception(f"Failed to ingest policy document {policy_doc.document_id}")
        policy_doc.processing_status = PolicyDocument.STATUS_FAILED
        policy_doc.error_message = str(e)
        policy_doc.save(update_fields=['processing_status', 'error_message'])
        return {
            'success': False,
            'document_id': policy_doc.document_id,
            'error': str(e)
        }
