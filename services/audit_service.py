"""
Independent 4-Tier Policy Claim Audit Service.
Verifies every factual policy claim made in the pitch against actual PDF text.
Taxonomy:
- SUPPORTED: Fully substantiated by verbatim PDF text.
- PARTIALLY_SUPPORTED: Benefit exists but pitch omitted sub-limits, deductibles, or conditions.
- UNSUPPORTED: Claim has no factual basis or is completely absent from the policy.
- CONTRADICTED: Policy explicitly contradicts or excludes what the pitch claimed.
"""
import logging
import re
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from django.conf import settings
from fpdf import FPDF

from pitch.models import PolicyDocument
from services.policy_ingestion import extract_pdf_pages
from services.llm_client import generate_structured
from services.policy_retrieval import search_policy

logger = logging.getLogger(__name__)

# Cache for extracted PDF pages to avoid re-parsing on every claim
_PDF_PAGE_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def get_pdf_page_text(policy_name: str, page_number: int) -> str:
    """
    Retrieves the actual verbatim text of a specific page from the uploaded PDF document.
    """
    clean_name = policy_name.strip().lower()
    
    # Check cache first
    for cached_key, pages in _PDF_PAGE_CACHE.items():
        if clean_name in cached_key.lower():
            for p in pages:
                if p.get("page_number") == page_number:
                    return p.get("text", "")

    # Look up in PolicyDocument model
    doc = PolicyDocument.objects.filter(document_name__icontains=policy_name.strip()).first()
    if not doc:
        # Fallback partial matching
        base_name = policy_name.split(".")[0].strip()
        doc = PolicyDocument.objects.filter(document_name__icontains=base_name).first()

    if doc and doc.file:
        file_path = doc.file.path
        if Path(file_path).exists():
            pages = extract_pdf_pages(file_path)
            _PDF_PAGE_CACHE[doc.document_name] = pages
            for p in pages:
                if p.get("page_number") == page_number:
                    return p.get("text", "")
            # If exact page out of bounds, return text of closest page
            if pages:
                idx = min(max(0, page_number - 1), len(pages) - 1)
                return pages[idx].get("text", "")

    # Direct filesystem fallback in media/policies/ and media/scratch/
    for folder in ["policies", "scratch"]:
        search_dir = Path(settings.MEDIA_ROOT) / folder
        if search_dir.exists():
            for f in search_dir.glob("*.pdf"):
                f_norm = f.name.lower().replace(" ", "").replace("_", "")
                p_norm = policy_name.lower().replace(" ", "").replace("_", "")
                p_base = policy_name.split(".")[0].lower().replace(" ", "").replace("_", "")
                if p_norm in f_norm or p_base in f_norm or f_norm in p_norm:
                    pages = extract_pdf_pages(str(f))
                    _PDF_PAGE_CACHE[policy_name] = pages
                    for p in pages:
                        if p.get("page_number") == page_number:
                            return p.get("text", "")
                    if pages:
                        idx = min(max(0, page_number - 1), len(pages) - 1)
                        return pages[idx].get("text", "")


    # Fallback to Qdrant chunks if PDF file cannot be read directly
    try:
        hits = search_policy(query=policy_name, limit=5)
        matched_chunks = [h.get("text", "") for h in hits if h.get("page_number") == page_number]
        if matched_chunks:
            return "\n\n".join(matched_chunks)
    except Exception as e:
        logger.warning(f"Failed fallback chunk retrieval for {policy_name}: {e}")

    return ""


def resolve_document_id(policy_name: str) -> Optional[str]:
    """Resolves policy document name to indexed document_id."""
    clean_name = policy_name.strip().lower()
    doc = PolicyDocument.objects.filter(document_name__icontains=clean_name).first()
    if not doc:
        base_name = policy_name.split(".")[0].strip()
        doc = PolicyDocument.objects.filter(document_name__icontains=base_name).first()
    return doc.document_id if doc else None


def retrieve_dual_angle_evidence(
    claim_text: str,
    cited_policy: str,
    cited_page: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Dual-Angle evidence retrieval against Qdrant Cloud:
    1. Affirmative Query: coverage terms, benefit scope, and eligibility.
    2. Adversarial Query: sub-limits, caps, co-payments, and exclusions.
    
    Returns merged, deduplicated evidence chunks with source page numbers.
    """
    doc_id = resolve_document_id(cited_policy)
    doc_ids = [doc_id] if doc_id else None

    # Angle 1: Affirmative / Benefit grant
    q1 = f"coverage terms benefit scope eligibility for {claim_text}"
    # Angle 2: Adversarial / Sub-limits & Exclusions
    q2 = f"sub-limits caps co-payment exclusions waiting periods for {claim_text}"

    chunks: List[Dict[str, Any]] = []
    seen = set()

    for q in [q1, q2]:
        try:
            hits = search_policy(query=q, document_ids=doc_ids, limit=3)
            for h in hits:
                # Deduplicate by (page_number, text prefix)
                sig = (h.get("page_number"), h.get("text", "")[:80])
                if sig not in seen and h.get("text", "").strip():
                    seen.add(sig)
                    chunks.append(h)
        except Exception as e:
            logger.warning(f"Dual-angle retrieval failed for query '{q}': {e}")

    # Fallback to direct physical page text if Qdrant returns no chunks
    if not chunks:
        raw_text = get_pdf_page_text(cited_policy, cited_page or 1)
        if raw_text:
            chunks.append({
                "document_name": cited_policy,
                "page_number": cited_page or 1,
                "text": raw_text,
                "section": "Cited PDF Page",
                "score": 1.0
            })

    return chunks


def _normalize_for_matching(text: str) -> str:
    """Normalizes whitespace and alphanumeric tokens for reliable substring verification."""
    import re
    return re.sub(r'[^a-z0-9]', ' ', text.lower()).strip()


def audit_single_claim(
    claim_text: str,
    cited_policy: str,
    cited_page: int,
    slide_number: int = 1
) -> Dict[str, Any]:
    """
    Audits an insurance assertion using Natural Language Inference (NLI)
    over dual-angle retrieved policy chunks, backed by programmatic verbatim quote verification.
    """
    chunks = retrieve_dual_angle_evidence(claim_text, cited_policy, cited_page)

    if not chunks:
        return {
            "claim_text": claim_text,
            "slide_number": slide_number,
            "cited_policy": cited_policy,
            "cited_page": cited_page,
            "verdict": "UNSUPPORTED",
            "confidence": 0.95,
            "evidence_quote": "No corresponding text found on the cited policy document.",
            "explanation": f"The cited document '{cited_policy}' could not be located in indexed records."
        }

    # Format premise context with exact page headers
    evidence_blocks = []
    for idx, c in enumerate(chunks, 1):
        p_num = c.get("page_number", cited_page)
        sec = c.get("section", "Policy Terms")
        txt = c.get("text", "").strip()
        evidence_blocks.append(f"[Evidence Clause {idx} | Page {p_num} | Section: {sec}]\n{txt}")

    premise_context = "\n\n".join(evidence_blocks)

    system_prompt = (
        "You are an expert commercial insurance auditor and underwriting claims adjuster. "
        "Your mandate is to audit assertions made in corporate pitch decks against the actual governing text of the insurance policy.\n\n"
        "STRICT 4-TIER NATURAL LANGUAGE INFERENCE TAXONOMY (Choose exactly ONE):\n"
        "1. SUPPORTED: Direct logical entailment. The policy explicitly grants coverage for the claimed procedure/benefit, "
        "AND the pitch accurately reflects the financial scope without omitting mandatory sub-limits, caps, or co-pays.\n"
        "2. PARTIALLY_SUPPORTED: Conditional entailment. The medical benefit exists in the policy, BUT the pitch omitted "
        "or obscured an explicit sub-limit (e.g., capped at ₹10,000, 10% of Sum Insured, or per-day limit), a co-payment, or a waiting period.\n"
        "3. CONTRADICTED: Direct contradiction. The policy explicitly excludes, denies, or imposes a strict limit contrary to what the pitch asserts "
        "(e.g., pitch claims 'no capping' or '100% covered without limits' when the policy imposes a cap, or claims an excluded condition is covered).\n"
        "4. UNSUPPORTED: Lack of factual entailment. The provided policy text does not grant, define, or substantiate this specific benefit "
        "(e.g., claiming coverage for an unmentioned procedure or a fabricated wellness allowance).\n\n"
        "CRITICAL REQUIREMENT: For SUPPORTED, PARTIALLY_SUPPORTED, or CONTRADICTED, you MUST provide an exact, verbatim quotation "
        "from the provided policy text in 'evidence_quote'. Do NOT paraphrase."
    )

    prompt = f"""CLAIM TO AUDIT:
"{claim_text}"

CITED POLICY: {cited_policy} (Originally cited Page {cited_page})

GOVERNING POLICY TEXT (PREMISE):
\"\"\"
{premise_context[:4500]}
\"\"\"

Audit this claim strictly using Natural Language Inference and output valid JSON:
{{
    "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED",
    "confidence": <float between 0.0 and 1.0>,
    "evidence_quote": "<Exact verbatim sentence from the policy text proving your verdict. If UNSUPPORTED, write 'Not mentioned in policy text'>",
    "verified_page_number": <Integer page number where the evidence quote was found, or null>,
    "explanation": "<1-2 concise sentences explaining why the claim is supported, qualified with sub-limits, unsupported, or contradicted>"
}}
"""

    try:
        res = generate_structured(prompt=prompt, system_prompt=system_prompt)
        verdict = res.get("verdict", "UNSUPPORTED").upper()
        if verdict not in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]:
            verdict = "UNSUPPORTED"

        evidence_quote = str(res.get("evidence_quote", "")).strip()
        explanation = str(res.get("explanation", ""))
        verified_page = res.get("verified_page_number")

        # Programmatic Verbatim Substring Verification Guardrail
        norm_premise = _normalize_for_matching(premise_context)
        norm_quote = _normalize_for_matching(evidence_quote)

        quote_verified = False
        if norm_quote and norm_quote != "not mentioned in policy text":
            if norm_quote in norm_premise:
                quote_verified = True
            else:
                # Check if first 6 words match (handles minor punctuation/whitespace splits)
                words = norm_quote.split()
                if len(words) >= 5 and " ".join(words[:5]) in norm_premise:
                    quote_verified = True

        # Demote if positive or contradictory verdict has no verbatim proof in text
        if verdict in ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED"]:
            if not quote_verified:
                verdict = "UNSUPPORTED"
                explanation = "Audit demoted to UNSUPPORTED: Cited evidence quote could not be verified verbatim in policy text."
            else:
                # Find the true chunk that contains this quote to auto-correct the page number
                for c in chunks:
                    if norm_quote in _normalize_for_matching(c.get("text", "")):
                        verified_page = c.get("page_number")
                        break
                    words = norm_quote.split()
                    if len(words) >= 5 and " ".join(words[:5]) in _normalize_for_matching(c.get("text", "")):
                        verified_page = c.get("page_number")
                        break

        # Fallback to originally cited page if verified page is unassigned
        final_page = int(verified_page) if verified_page and str(verified_page).isdigit() else cited_page

        return {
            "claim_text": claim_text,
            "slide_number": slide_number,
            "cited_policy": cited_policy,
            "cited_page": final_page,
            "verdict": verdict,
            "confidence": float(res.get("confidence", 0.9)),
            "evidence_quote": evidence_quote if quote_verified else "No verified quotation available.",
            "explanation": explanation
        }

    except Exception as e:
        logger.exception(f"Error auditing claim '{claim_text}' via LLM: {e}")
        raise


def audit_pitch_claims(pitch_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Audits every policy claim in the pitch slides against actual PDF text.
    
    Returns structured audit report:
    - total_claims: int
    - supported_count: int
    - partially_supported_count: int
    - unsupported_count: int
    - contradicted_count: int
    - grounding_score: float (0 - 100)
    - audit_status: "PASSED" | "REVIEW_REQUIRED"
    - audited_claims: List[Dict[str, Any]]
    """
    slides = pitch_data.get("slides", [])
    audited_claims = []

    for slide in slides:
        slide_num = slide.get("slide_number", 1)
        claims = slide.get("policy_claims", [])
        for cl in claims:
            claim_text = cl.get("claim_text", "").strip()
            cited_policy = cl.get("cited_policy", "Policy.pdf").strip()
            cited_page = int(cl.get("cited_page", 1))

            if claim_text:
                audit_result = audit_single_claim(
                    claim_text=claim_text,
                    cited_policy=cited_policy,
                    cited_page=cited_page,
                    slide_number=slide_num
                )
                if audit_result.get("cited_page"):
                    cl["cited_page"] = audit_result["cited_page"]
                audited_claims.append(audit_result)

    total = len(audited_claims)
    supported = sum(1 for c in audited_claims if c["verdict"] == "SUPPORTED")
    partially = sum(1 for c in audited_claims if c["verdict"] == "PARTIALLY_SUPPORTED")
    unsupported = sum(1 for c in audited_claims if c["verdict"] == "UNSUPPORTED")
    contradicted = sum(1 for c in audited_claims if c["verdict"] == "CONTRADICTED")

    if total > 0:
        # Supported = 100%, Partially = 50%, Unsupported/Contradicted = 0%
        grounding_score = round(((supported * 1.0 + partially * 0.5) / total) * 100, 1)
        status = "PASSED" if (unsupported == 0 and contradicted == 0) else "REVIEW_REQUIRED"
    else:
        grounding_score = 0.0
        status = "NO_CLAIMS"

    report_result = {
        "company_name": pitch_data.get("company_name", ""),
        "total_claims": total,
        "supported_count": supported,
        "partially_supported_count": partially,
        "unsupported_count": unsupported,
        "contradicted_count": contradicted,
        "grounding_score": grounding_score,
        "audit_status": status,
        "audited_claims": audited_claims
    }

    try:
        report_path, report_url = compile_audit_report_pdf(report_result)
        report_result["report_path"] = str(report_path)
        report_result["report_url"] = report_url
    except Exception as e:
        logger.warning(f"Could not compile audit report PDF: {e}")
        report_result["report_path"] = None
        report_result["report_url"] = None

    return report_result


def clean_pdf_text(text: str) -> str:
    """Sanitizes text for standard latin-1 encoding in FPDF."""
    if not text:
        return ""
    replacements = {
        '₹': 'INR ',
        '’': "'",
        '‘': "'",
        '“': '"',
        '”': '"',
        '–': '-',
        '—': '-',
        '•': '*',
        '…': '...',
        '\u2013': '-',
        '\u2014': '-',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')


class CorporateAuditPDF(FPDF):
    """Branded corporate PDF document with running header and footer."""
    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 8, f'Marsh Commercial Advisory  |  Policy Audit & Traceability Dossier  |  Confidential  |  Page {self.page_no()}', align='C')


def clean_pdf_quote(text: str) -> str:
    """Sanitizes policy evidence quote and removes raw markdown table pipes and formatting."""
    if not text:
        return ""
    # Remove markdown table dividers like |---|---|
    text = re.sub(r'\|[ -:]*\|', ' ', text)
    # Replace individual pipe characters with comma or spaces
    text = text.replace('|', ' ')
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return clean_pdf_text(text)


def compile_audit_report_pdf(audit_report: Dict[str, Any]) -> tuple[Path, str]:
    """
    Compiles structured claim audit data into an executive, branded audit report PDF.
    Meets Marsh Case Study Objective 2.1, 2.2, and Deliverable 3.
    """
    company_name = audit_report.get("company_name", "Corporate Client")
    company_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name.lower())
    presentations_dir = Path(settings.MEDIA_ROOT) / "presentations"
    presentations_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"{company_slug}_audit_report.pdf"
    out_path = presentations_dir / pdf_filename

    pdf = CorporateAuditPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    # Brand Header Bar (Deep Navy with Sky Accent)
    pdf.set_fill_color(10, 37, 64)
    pdf.rect(0, 0, 210, 26, 'F')
    pdf.set_xy(10, 5)
    pdf.set_font('Helvetica', 'B', 15)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 8, 'MARSH COMMERCIAL ADVISORY', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(147, 197, 253)
    pdf.cell(190, 5, 'Independent 4-Tier Policy Claim Audit & Traceability Report', new_x='LMARGIN', new_y='NEXT')

    # Executive Metadata Box (30mm height, cleanly structured rows with zero collision)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 29, 190, 30, 'FD')

    # Row 1: Target Client (left) & Timestamped Audit Date (right)
    pdf.set_xy(14, 32)
    pdf.set_font('Helvetica', 'B', 10.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(100, 6, clean_pdf_text(f'Target Client: {company_name}'))
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(100, 116, 139)
    date_str = datetime.datetime.now().strftime('%d %B %Y, %H:%M UTC')
    pdf.cell(82, 6, f'Audit Date: {date_str}', align='R')

    # Row 2: Status & Grounding Score
    pdf.set_xy(14, 40)
    status = audit_report.get('audit_status', 'REVIEW_REQUIRED')
    score = audit_report.get('grounding_score', 0.0)
    r, g, b = (16, 185, 129) if status == 'PASSED' else (217, 119, 6)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(r, g, b)
    pdf.cell(182, 6, f'Overall Compliance Status: {status}  |  Grounding Score: {score}%')

    # Row 3: Verification Methodology
    pdf.set_xy(14, 47)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(182, 5, 'Audit Verification Framework: Natural Language Inference (NLI) + Dual-Stage Verbatim OCR Concordance')

    # KPI Summary Table
    pdf.set_y(64)
    pdf.set_font('Helvetica', 'B', 11.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(190, 7, '1. Executive Audit Summary & Metrics', new_x='LMARGIN', new_y='NEXT')

    headers = ['Total Claims', 'Supported', 'Partially Supported', 'Contradicted', 'Unsupported', 'Grounding Score']
    values = [
        str(audit_report.get('total_claims', 0)),
        str(audit_report.get('supported_count', 0)),
        str(audit_report.get('partially_supported_count', 0)),
        str(audit_report.get('contradicted_count', 0)),
        str(audit_report.get('unsupported_count', 0)),
        f'{score}%'
    ]
    col_w = [30, 28, 38, 30, 32, 32]
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(51, 65, 85)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, h, border=1, align='C', fill=True)
    pdf.set_xy(10, pdf.get_y() + 7)

    pdf.set_font('Helvetica', 'B', 10.5)
    pdf.set_text_color(15, 23, 42)
    for w, v in zip(col_w, values):
        pdf.cell(w, 8, v, border=1, align='C')
    pdf.set_xy(10, pdf.get_y() + 12)

    # Claim Details Header
    pdf.set_font('Helvetica', 'B', 11.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(190, 7, '2. Claim-by-Claim Verification & Evidence Matrix', new_x='LMARGIN', new_y='NEXT')

    claims = audit_report.get('audited_claims', [])
    for idx, c in enumerate(claims, 1):
        verdict = c.get('verdict', 'UNSUPPORTED').upper()
        claim_text = clean_pdf_text(c.get('claim_text', ''))
        policy = clean_pdf_text(c.get('cited_policy', 'Policy'))
        page = c.get('cited_page', 1)
        slide_num = c.get('slide_number', 1)
        quote = clean_pdf_quote(c.get('evidence_quote', ''))
        expl = clean_pdf_text(c.get('explanation', ''))

        # Set color palette based on verdict
        if verdict == 'SUPPORTED':
            fill_r, fill_g, fill_b = (236, 253, 245) # Emerald tint
            head_r, head_g, head_b = (6, 95, 70)    # Deep emerald text
        elif verdict == 'PARTIALLY_SUPPORTED':
            fill_r, fill_g, fill_b = (254, 243, 199) # Amber tint
            head_r, head_g, head_b = (146, 64, 14)  # Deep amber text
        elif verdict == 'CONTRADICTED':
            fill_r, fill_g, fill_b = (254, 226, 226) # Red tint
            head_r, head_g, head_b = (153, 27, 27)   # Deep red text
        else: # UNSUPPORTED
            fill_r, fill_g, fill_b = (254, 242, 242) # Crimson tint
            head_r, head_g, head_b = (153, 27, 27)

        pdf.set_draw_color(203, 213, 225)
        pdf.set_x(10)
        pdf.set_font('Helvetica', 'B', 9.5)
        pdf.set_fill_color(fill_r, fill_g, fill_b)
        pdf.set_text_color(head_r, head_g, head_b)
        pdf.cell(190, 7.5, f' Claim #{idx} [Slide {slide_num}] - Verdict: {verdict}', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')

        # Claim statement
        pdf.set_x(10)
        pdf.set_font('Helvetica', 'BI', 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.multi_cell(190, 5.5, f'  \"{claim_text}\"', border='LR')

        # Cited policy source
        pdf.set_x(10)
        pdf.set_font('Helvetica', 'B', 8.5)
        pdf.set_text_color(29, 78, 216) # Brand blue
        pdf.multi_cell(190, 5, f'  [Policy Reference: {policy} | Page {page}]', border='LR')

        # Verbatim evidence quote
        if quote and quote.lower() != 'no verified quotation available.':
            pdf.set_x(10)
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(15, 23, 42)
            pdf.multi_cell(190, 5.5, f'  Verbatim Policy Excerpt: \"{quote}\"', border='LR')

        # Auditor analysis
        pdf.set_x(10)
        if expl:
            pdf.set_font('Helvetica', '', 8.5)
            pdf.set_text_color(71, 85, 105)
            pdf.multi_cell(190, 5, f'  Auditor Analysis: {expl}', border='LRB')
        else:
            pdf.cell(190, 1, '', border='B', new_x='LMARGIN', new_y='NEXT')

        pdf.set_y(pdf.get_y() + 4)

    # Advisor Sign-off Box
    pdf.set_y(pdf.get_y() + 4)
    pdf.set_x(10)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_font('Helvetica', 'B', 10.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(190, 7.5, '3. Advisor Governance & Compliance Sign-Off', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
    pdf.set_x(10)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(190, 5.5, '[  ] APPROVED: All policy assertions fully verified against governing documents.\n[  ] CONDITIONAL APPROVAL: Omitted sub-limits or caps verbally clarified to client.\n[  ] REJECTED: Unsubstantiated claims detected. Deck requires revision prior to presentation.\n\nAdvisor Name: _______________________      Signature: _______________________      Date: ___________', border=1)

    pdf.output(str(out_path))
    return out_path, f"{settings.MEDIA_URL}presentations/{pdf_filename}"


def auditPitchContent(
    pitch_slides: List[Dict[str, Any]],
    policy_docs: Optional[List[str]] = None,
    company_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Marsh Case Study Objective 2.2:
    Implements auditPitchContent(pitch_slides, policy_docs) returning a structured audit report.
    """
    return audit_pitch_claims({"slides": pitch_slides, "company_name": company_name or ""})


def filter_pitch_claims_by_audit(pitch_data: Dict[str, Any], audit_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes presentation slides to ensure ONLY verified claims
    (SUPPORTED or PARTIALLY_SUPPORTED) are included in the PowerPoint (.pptx).
    Preserves all claims in pitch_data so the frontend audit scorecard and slide
    drawers can display all verified and unverified claims with their findings.
    Re-compiles the PowerPoint presentation deck with verified claims only.
    """
    import copy
    valid_verdicts = {"SUPPORTED", "PARTIALLY_SUPPORTED"}

    # Map (slide_number, normalized_claim_text) -> audit item
    audit_map = {}
    for ac in audit_report.get("audited_claims", []):
        text_norm = re.sub(r'[^a-zA-Z0-9]', '', ac.get("claim_text", "").lower())
        slide_num = ac.get("slide_number")
        audit_map[(slide_num, text_norm)] = ac
        audit_map[text_norm] = ac

    # Annotate ALL claims in pitch_data with audit findings (retains full history for UI)
    for slide in pitch_data.get("slides", []):
        slide_num = slide.get("slide_number")
        for cl in slide.get("policy_claims", []):
            c_text_norm = re.sub(r'[^a-zA-Z0-9]', '', cl.get("claim_text", "").lower())
            ac = audit_map.get((slide_num, c_text_norm)) or audit_map.get(c_text_norm)
            if ac:
                if ac.get("cited_page"):
                    cl["cited_page"] = ac["cited_page"]
                cl["audit_verdict"] = ac.get("verdict")
                cl["audit_explanation"] = ac.get("explanation")
                cl["audit_quote"] = ac.get("evidence_quote")

    # Build a sanitized copy strictly for compiling the PowerPoint (.pptx) deck
    pptx_pitch_data = copy.deepcopy(pitch_data)
    for slide in pptx_pitch_data.get("slides", []):
        slide_num = slide.get("slide_number")
        verified_claims = []
        for cl in slide.get("policy_claims", []):
            c_text_norm = re.sub(r'[^a-zA-Z0-9]', '', cl.get("claim_text", "").lower())
            ac = audit_map.get((slide_num, c_text_norm)) or audit_map.get(c_text_norm)
            if ac and ac.get("verdict") in valid_verdicts:
                verified_claims.append(cl)
        slide["policy_claims"] = verified_claims

    # Re-compile PowerPoint (.pptx) with ONLY verified claims
    from services.pitch_generator import compile_powerpoint
    pptx_path, pptx_url = compile_powerpoint(pptx_pitch_data)
    pitch_data["pptx_path"] = str(pptx_path)
    pitch_data["pptx_url"] = pptx_url
    return pitch_data


