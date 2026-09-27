"""
Policy retrieval service with hybrid search (Dense + BM25 → RRF → Reranking).
Grounds pitches in verified document evidence with exact page numbers.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from django.conf import settings
from qdrant_client.http import models as qmodels

from services.qdrant_service import get_qdrant_client
from services.embedding_service import embed_text

logger = logging.getLogger(__name__)

# ── BM25 index cache ────────────────────────────────────────────────
_BM25_INDEX = None
_BM25_CORPUS_PAYLOADS: List[Dict[str, Any]] = []

# ── Cross-encoder reranker singleton ────────────────────────────────
_RERANKER_INSTANCE = None


def invalidate_bm25_cache():
    """Call after ingesting new documents to rebuild BM25 index on next search."""
    global _BM25_INDEX, _BM25_CORPUS_PAYLOADS
    _BM25_INDEX = None
    _BM25_CORPUS_PAYLOADS = []


# ====================================================================
# INTERNAL HELPERS
# ====================================================================

def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer for BM25."""
    return re.findall(r'\b\w+\b', text.lower())


def _build_bm25_index(collection_name: str):
    """Scroll all Qdrant points and build an in-memory BM25 index."""
    global _BM25_INDEX, _BM25_CORPUS_PAYLOADS
    from rank_bm25 import BM25Okapi

    client = get_qdrant_client()
    all_payloads: List[Dict[str, Any]] = []
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result
        for pt in points:
            if pt.payload:
                all_payloads.append(pt.payload)
        if next_offset is None:
            break
        offset = next_offset

    if not all_payloads:
        _BM25_INDEX = None
        _BM25_CORPUS_PAYLOADS = []
        return

    corpus_tokens = []
    valid_payloads = []
    for payload in all_payloads:
        text = payload.get('enriched_text', '') or payload.get('chunk_text', '')
        if text.strip():
            corpus_tokens.append(_tokenize(text))
            valid_payloads.append(payload)

    if corpus_tokens:
        _BM25_INDEX = BM25Okapi(corpus_tokens)
        _BM25_CORPUS_PAYLOADS = valid_payloads
    else:
        _BM25_INDEX = None
        _BM25_CORPUS_PAYLOADS = []

    logger.info(f"BM25 index built with {len(valid_payloads)} chunks")


def _bm25_search(
    query: str,
    collection_name: str,
    document_ids: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """BM25 keyword search over the cached corpus."""
    global _BM25_INDEX, _BM25_CORPUS_PAYLOADS

    if _BM25_INDEX is None:
        _build_bm25_index(collection_name)
    if _BM25_INDEX is None or not _BM25_CORPUS_PAYLOADS:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = _BM25_INDEX.get_scores(tokens)

    scored: List[Dict[str, Any]] = []
    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        payload = _BM25_CORPUS_PAYLOADS[idx]
        if document_ids and payload.get('document_id', '') not in document_ids:
            continue
        scored.append({
            'document_id': payload.get('document_id', ''),
            'document_name': payload.get('document_name', ''),
            'page_number': payload.get('page_number', 1),
            'text': payload.get('chunk_text', ''),
            'section': payload.get('section', 'General'),
            'score': float(score),
        })

    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:limit]


def _dense_search(
    query: str,
    collection_name: str,
    document_ids: Optional[List[str]] = None,
    limit: int = 10,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Dense vector search via Qdrant."""
    client = get_qdrant_client()
    query_vector = embed_text(query.strip())

    query_filter = None
    if document_ids:
        cleaned = [str(d).strip() for d in document_ids if str(d).strip()]
        if len(cleaned) == 1:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=cleaned[0]),
                )]
            )
        elif len(cleaned) > 1:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchAny(any=cleaned),
                )]
            )

    try:
        if hasattr(client, 'query_points'):
            response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )
            hits = response.points
        else:
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )
    except Exception as e:
        logger.exception(f"Dense search error: {e}")
        return []

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append({
            'document_id': payload.get('document_id', ''),
            'document_name': payload.get('document_name', ''),
            'page_number': payload.get('page_number', 1),
            'text': payload.get('chunk_text', ''),
            'section': payload.get('section', 'General'),
            'score': round(float(hit.score), 4),
        })
    return results


def _rrf_merge(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion.
    RRF_score(d) = Σ 1/(k + rank_i) across each ranked list.
    """
    merged: Dict[tuple, Dict[str, Any]] = {}

    for rank, item in enumerate(dense_results):
        sig = (item.get('document_name', ''), item.get('page_number', 1),
               item.get('text', '')[:80])
        if sig not in merged:
            merged[sig] = {**item, '_rrf': 0.0}
        merged[sig]['_rrf'] += 1.0 / (k + rank + 1)

    for rank, item in enumerate(bm25_results):
        sig = (item.get('document_name', ''), item.get('page_number', 1),
               item.get('text', '')[:80])
        if sig not in merged:
            merged[sig] = {**item, '_rrf': 0.0}
        merged[sig]['_rrf'] += 1.0 / (k + rank + 1)

    results = sorted(merged.values(), key=lambda x: x['_rrf'], reverse=True)
    for r in results:
        r['score'] = round(r.pop('_rrf'), 6)
    return results


def _get_reranker():
    """Singleton cross-encoder reranker (downloads model on first call)."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        from fastembed import TextCrossEncoder
        _RERANKER_INSTANCE = TextCrossEncoder(
            model_name="Xenova/ms-marco-MiniLM-L-6-v2"
        )
    return _RERANKER_INSTANCE


def _rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Cross-encoder reranking via fastembed.
    Falls back to RRF ordering if the reranker is unavailable.
    """
    if not candidates:
        return []

    try:
        reranker = _get_reranker()
        documents = [c.get('text', '') for c in candidates]
        rerank_results = list(reranker.rerank(query, documents))

        scored: List[Dict[str, Any]] = []
        for result in rerank_results:
            idx = getattr(result, 'index', None)
            score = getattr(result, 'score', None)
            if idx is None or score is None:
                continue
            if 0 <= idx < len(candidates):
                entry = dict(candidates[idx])
                entry['score'] = round(float(score), 4)
                scored.append(entry)

        if scored:
            scored.sort(key=lambda x: x['score'], reverse=True)
            return scored[:limit]
    except Exception as e:
        logger.debug(f"Cross-encoder reranking unavailable, using RRF scores: {e}")

    return candidates[:limit]


def _apply_diversity(
    results: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    """
    Soft document diversity: caps any single document at ~60 % of slots.
    Promotes underrepresented docs without forcing equal quotas.
    """
    if len(results) <= limit:
        return results

    max_per_doc = max(2, int(limit * 0.6))
    doc_counts: Dict[str, int] = {}
    selected: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []

    for r in results:
        doc = r.get('document_name', '')
        count = doc_counts.get(doc, 0)
        if count < max_per_doc:
            selected.append(r)
            doc_counts[doc] = count + 1
        else:
            deferred.append(r)
        if len(selected) >= limit:
            break

    # Fill remaining from deferred (no artificial exclusion)
    while len(selected) < limit and deferred:
        selected.append(deferred.pop(0))

    return selected[:limit]


# ====================================================================
# PUBLIC API  (unchanged signature)
# ====================================================================

def search_policy(
    query: str,
    document_ids: Optional[List[str]] = None,
    limit: int = 5,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid search: Dense + BM25 → RRF → Cross-encoder Rerank → Diversity.

    Returns list of dicts:
        [{'document_id', 'document_name', 'page_number', 'text',
          'section', 'score'}]
    """
    if not query or not query.strip():
        return []

    collection_name = getattr(
        settings, 'QDRANT_COLLECTION_NAME', 'policy_documents'
    )
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        logger.warning(f"Qdrant collection '{collection_name}' does not exist.")
        return []

    # Wider candidate pool for fusion (3× requested limit, min 15)
    pool = max(limit * 3, 15)

    # 1. Dense vector search
    dense = _dense_search(
        query, collection_name, document_ids, pool, score_threshold
    )
    # 2. BM25 keyword search
    bm25 = _bm25_search(query, collection_name, document_ids, pool)

    # 3. Reciprocal Rank Fusion
    fused = _rrf_merge(dense, bm25)

    # 4. Cross-encoder reranking
    reranked = _rerank(query, fused[:pool], limit=pool)

    # 5. Soft document diversity
    final = _apply_diversity(reranked, limit)

    return final
