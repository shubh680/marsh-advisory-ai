"""
Inngest Workflow Orchestrator for Marsh Commercial Insurance Advisory Pipeline.
Single Orchestrated Inngest Function: marsh-advisory-pitch-pipeline
Coordinates the 4 pipeline steps within a single Inngest execution:
1. research-company-profile
2. formulate-workforce-health
3. generate-marketing-pitch
4. audit-pitch-claims
"""
import json
import logging
import datetime
import time
import uuid
from typing import Dict, Any, List, Optional
import inngest
import inngest.django

logger = logging.getLogger("marsh.pipeline")

# Initialize Inngest client (is_production=False for local dev server auto-discovery)
inngest_client = inngest.Inngest(
    app_id="marsh-commercial-advisor",
    is_production=False
)

# In-memory storage for completed pipeline runs by run_id
PIPELINE_RUNS: Dict[str, Any] = {}


def log_event(logs_list: List[Dict[str, str]], stage: str, message: str, level: str = "info") -> None:
    """Helper to log to Python logger and print to server console without creating extra Inngest calls."""
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "time": now_str,
        "stage": stage,
        "message": message,
        "level": level
    }
    logs_list.append(log_entry)
    
    # Direct server console print for terminal observability
    print(f"[{now_str}] [{stage}] {message}", flush=True)

    if level == "error":
        logger.error(f"[{stage}] {message}")
    elif level == "warning":
        logger.warning(f"[{stage}] {message}")
    else:
        logger.info(f"[{stage}] {message}")


# =========================================================================
# SINGLE UNIFIED INNGEST PIPELINE FUNCTION
# =========================================================================

@inngest_client.create_function(
    fn_id="marsh-advisory-pitch-pipeline",
    name="Marsh Commercial Advisory & 4-Tier Audit Pipeline",
    trigger=inngest.TriggerEvent(event="marsh/pitch.generate")
)
def orchestrate_pitch_pipeline(ctx: inngest.ContextSync, step: Optional[Any] = None) -> Dict[str, Any]:
    """
    Unified Inngest Multi-Step Pipeline.
    One single Inngest call executes all 4 sequential pipeline steps in order:
    1. research-company-profile
    2. formulate-workforce-health
    3. generate-marketing-pitch
    4. audit-pitch-claims
    """
    step = step or getattr(ctx, "step", None)
    event = getattr(ctx, "event", None)
    data = (getattr(event, "data", None) if event else None) or {}
    company_name = data.get("company_name", "Corporate Client")
    selected_docs = data.get("selected_docs") or []
    cached_profile = data.get("company_profile")
    run_id = data.get("run_id")

    print(f"[INNGEST_PIPELINE] Running Step 1/4: research-company-profile for '{company_name}'", flush=True)

    # Step 1: Research Company Profile
    def _run_step_1():
        if cached_profile and isinstance(cached_profile, dict):
            cached_company = (cached_profile.get("company_name") or "").strip().lower()
            target_company = company_name.strip().lower()
            if cached_company and (cached_company in target_company or target_company in cached_company):
                return cached_profile
        from services.company_research import generateCompanyProfile
        return generateCompanyProfile(company_name)

    company_profile = step.run("research-company-profile", _run_step_1) if step else _run_step_1()

    print(f"[INNGEST_PIPELINE] Running Step 2/4: formulate-workforce-health for '{company_name}'", flush=True)

    # Step 2: Formulate Workforce Health Exposures
    def _run_step_2():
        from services.workforce_health import formulate_workforce_health_exposures
        return formulate_workforce_health_exposures(company_profile=company_profile)

    workforce_health = step.run("formulate-workforce-health", _run_step_2) if step else _run_step_2()

    print(f"[INNGEST_PIPELINE] Running Step 3/4: generate-marketing-pitch from Qdrant Cloud", flush=True)

    # Step 3: Retrieve Qdrant Policy Evidence & Generate PPTX Pitch
    def _run_step_3():
        from services.pitch_generator import generateMarketingPitch
        return generateMarketingPitch(
            company_profile=company_profile,
            workforce_health=workforce_health,
            document_ids=selected_docs
        )

    pitch_data = step.run("generate-marketing-pitch", _run_step_3) if step else _run_step_3()

    print(f"[INNGEST_PIPELINE] Running Step 4/4: audit-pitch-claims against verbatim PDF text", flush=True)

    # Step 4: Independent 4-Tier Claim Audit
    def _run_step_4():
        from services.audit_service import auditPitchContent
        return auditPitchContent(
            pitch_slides=pitch_data.get("slides", []),
            policy_docs=selected_docs,
            company_name=company_name
        )

    audit_report = step.run("audit-pitch-claims", _run_step_4) if step else _run_step_4()

    # Sanitize PPTX: Keep ONLY verified claims (SUPPORTED or PARTIALLY_SUPPORTED)
    from services.audit_service import filter_pitch_claims_by_audit
    pitch_data = filter_pitch_claims_by_audit(pitch_data, audit_report)

    result = {
        "success": True,
        "company_name": company_name,
        "company_profile": company_profile,
        "workforce_health": workforce_health,
        "pitch": pitch_data,
        "audit": audit_report
    }

    if run_id:
        PIPELINE_RUNS[run_id] = result
        print(f"[INNGEST_PIPELINE] Run '{run_id[:8]}' completed all 4 steps successfully!", flush=True)

    return result


def execute_direct_pipeline(
    company_name: str,
    selected_docs: Optional[List[str]] = None,
    cached_profile: Optional[Dict[str, Any]] = None,
    execution_logs: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Helper to run the 4 steps directly if Inngest is not responding."""
    if execution_logs is None:
        execution_logs = []

    # 1. Company Profile
    company_profile = None
    if cached_profile and isinstance(cached_profile, dict):
        cached_company = (cached_profile.get("company_name") or "").strip().lower()
        target_company = company_name.strip().lower()
        if cached_company and (cached_company in target_company or target_company in cached_company):
            company_profile = cached_profile

    if not company_profile or not isinstance(company_profile, dict):
        log_event(execution_logs, "RESEARCH_AGENT", f"Querying external corporate sources for '{company_name}'...")
        try:
            from services.company_research import generateCompanyProfile
            company_profile = generateCompanyProfile(company_name=company_name)
            sector = company_profile.get("industry_sector") or company_profile.get("industry", "Commercial")
            scale = company_profile.get("company_size", "Enterprise")
            log_event(execution_logs, "RESEARCH_AGENT", f"Synthesized Profile: Sector = '{sector}', Scale = '{scale}'", "success")
        except Exception as e:
            log_event(execution_logs, "RESEARCH_AGENT", f"Research failed: {e}", "error")
            raise

    # 2. Workforce Health
    log_event(execution_logs, "WORKFORCE_HEALTH", "Translating operations into employee healthcare exposures...")
    from services.workforce_health import formulate_workforce_health_exposures
    try:
        workforce_health = formulate_workforce_health_exposures(company_profile=company_profile)
        exposures = workforce_health.get("priority_healthcare_exposures", [])
        log_event(execution_logs, "WORKFORCE_HEALTH", f"Formulated {len(exposures)} medical exposure priorities", "success")
    except Exception as e:
        log_event(execution_logs, "WORKFORCE_HEALTH", f"Workforce health formulation failed: {e}", "error")
        raise

    # 3. Qdrant RAG & Pitch Synthesis
    log_event(execution_logs, "QDRANT_RAG", "Querying Qdrant Cloud collection 'policy_documents' with 384d BAAI dense vectors...")
    from services.pitch_generator import generateMarketingPitch
    try:
        pitch_data = generateMarketingPitch(
            company_profile=company_profile,
            workforce_health=workforce_health,
            document_ids=selected_docs
        )
        slides = pitch_data.get("slides", [])
        evidence_chunks = pitch_data.get("candidate_evidence_chunks", [])
        pptx_filename = pitch_data.get("pptx_url", "").split("/")[-1]
        log_event(execution_logs, "QDRANT_RAG", f"Retrieved {len(evidence_chunks)} candidate clauses from Qdrant Cloud", "success")
        log_event(execution_logs, "PITCH_SYNTHESIS", f"LLM generated {len(slides)} slides matching Marsh Commercial Advisory standard", "success")
        log_event(execution_logs, "PPTX_RENDERER", f"Compiled native 16:9 presentation deck: '{pptx_filename}'", "success")
    except Exception as e:
        log_event(execution_logs, "PITCH_SYNTHESIS", f"Pitch generation failed: {e}", "error")
        raise

    # 4. Independent 4-Tier Audit
    total_claims = sum(len(s.get("policy_claims", [])) for s in pitch_data.get("slides", []))
    log_event(execution_logs, "AUDIT_ENGINE", f"Auditing {total_claims} claims across 4 tiers...")
    from services.audit_service import auditPitchContent, filter_pitch_claims_by_audit
    try:
        audit_report = auditPitchContent(
            pitch_slides=pitch_data.get("slides", []),
            policy_docs=selected_docs,
            company_name=company_name
        )
        score = audit_report.get("grounding_score", 0.0)
        status = audit_report.get("audit_status", "REVIEW_REQUIRED")
        log_event(execution_logs, "AUDIT_ENGINE", f"Audit Completed: Grounding Score = {score}% | Status = {status}", "success")

        # Sanitize PPTX: Keep ONLY verified claims (SUPPORTED or PARTIALLY_SUPPORTED)
        pitch_data = filter_pitch_claims_by_audit(pitch_data, audit_report)
        log_event(execution_logs, "PPTX_RENDERER", "Sanitized PowerPoint deck: excluded unsupported claims, verified only supported terms.", "success")
    except Exception as e:
        log_event(execution_logs, "AUDIT_ENGINE", f"Audit claim interpretation failed: {e}", "error")
        raise

    return {
        "success": True,
        "company_name": company_name,
        "company_profile": company_profile,
        "workforce_health": workforce_health,
        "pitch": pitch_data,
        "audit": audit_report,
        "execution_logs": execution_logs
    }


def run_orchestrated_pipeline(
    company_name: str,
    selected_docs: Optional[List[str]] = None,
    cached_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes the 4-phase advisory pipeline.
    Sends ONE single Inngest call ('marsh/pitch.generate') which executes all 4 steps
    sequentially within Inngest, with fallback to direct execution if Inngest is offline.
    """
    run_id = str(uuid.uuid4())
    execution_logs: List[Dict[str, str]] = []
    log_event(execution_logs, "INITIALIZATION", f"Starting commercial advisory pipeline for '{company_name}' (Run ID: {run_id[:8]})")

    # Dispatch ONE Inngest call
    inngest_dispatched = False
    try:
        inngest_client.send_sync(inngest.Event(
            name="marsh/pitch.generate",
            data={
                "run_id": run_id,
                "company_name": company_name,
                "selected_docs": selected_docs,
                "company_profile": cached_profile
            }
        ))
        log_event(execution_logs, "INNGEST", f"Dispatched 1 Inngest pipeline call (marsh/pitch.generate). Executing 4 steps...", "success")
        inngest_dispatched = True
    except Exception as e:
        logger.warning(f"Inngest dispatch unavailable: {e}")

    # If Inngest was dispatched, wait for Inngest to complete the 4 steps
    if inngest_dispatched:
        start_time = time.time()
        while time.time() - start_time < 90:
            if run_id in PIPELINE_RUNS:
                res = PIPELINE_RUNS.pop(run_id)
                res["execution_logs"] = execution_logs
                log_event(execution_logs, "INNGEST", "Inngest completed all 4 steps in single workflow run.", "success")
                return res
            # Check if Inngest server did not pick up run within 6 seconds
            if time.time() - start_time > 6 and not any(run_id in k for k in PIPELINE_RUNS):
                logger.info("Inngest server did not pick up run in 6s, falling back to direct execution...")
                break
            time.sleep(0.4)

    # Fallback to direct execution
    return execute_direct_pipeline(company_name, selected_docs, cached_profile, execution_logs)


# Native Django View for /api/inngest serving the single multi-step pipeline
inngest_django_pattern = inngest.django.serve(
    inngest_client,
    [orchestrate_pitch_pipeline]
)
