"""
Workforce Healthcare & Medical Exposure Formulator.
Translates generalized company research (industry, workforce, operations, footprint)
strictly into corporate employee healthcare, medical risk, and hospitalization coverage needs.
Explicitly avoids non-medical insurance concepts (e.g. machinery breakdown, cargo, property).
"""
import logging
from typing import Dict, Any, List
from services.llm_client import generate_structured

logger = logging.getLogger(__name__)


def formulate_workforce_health_exposures(company_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes a company's corporate profile to determine its employee healthcare
    priorities and generate targeted medical policy search queries for Qdrant retrieval.
    
    Args:
        company_profile: Output from research_company() containing industry,
                         workforce scale, activities, footprint, and risks.
                         
    Returns:
        Structured dictionary containing:
        - company_name: str
        - workforce_profile_summary: str
        - priority_healthcare_exposures: List[Dict[str, str]]
        - medical_search_queries: List[str]
    """
    company_name = company_profile.get("company_name", "Target Company")
    industry = company_profile.get("industry_sector") or company_profile.get("industry") or "Corporate Commercial"
    size = company_profile.get("company_size", "Enterprise scale")
    activities = company_profile.get("business_activities", [])
    footprint = company_profile.get("operational_footprint", [])
    risks = company_profile.get("key_risks_and_exposures") or company_profile.get("key_risks", [])

    system_prompt = (
        "You are an expert corporate health insurance underwriter and employee benefits advisor at Marsh.\n"
        "Your task is to analyze a company's operational profile and identify its critical "
        "EMPLOYEE HEALTHCARE, MEDICAL PROTECTION, AND HOSPITALIZATION exposures.\n\n"
        "CRITICAL DOMAIN RULES:\n"
        "1. STRICT MEDICAL & HEALTH FOCUS: You must ONLY identify health, medical, wellness, and hospitalization needs.\n"
        "2. FORBIDDEN DOMAINS: NEVER generate property, casualty, or general commercial queries "
        "(e.g., NO machinery breakdown, NO cargo/transit, NO cybersecurity, NO physical property damage, NO general liability).\n"
        "   Even if the company is an industrial manufacturer, map physical plant hazards to 'workplace trauma care, emergency hospitalization, ICU coverage, and air ambulance for remote sites'.\n"
        "3. GROUNDED IN UPLOADED MEDICAL POLICIES: The available insurance documents are comprehensive health/medical policies covering:\n"
        "   - Inpatient Hospitalization & ICU Room Rent\n"
        "   - Emergency Road and Air Ambulance\n"
        "   - Day Care Procedures & Modern Robotic Treatments\n"
        "   - Pre & Post Hospitalization Medical Expenses\n"
        "   - Cumulative Bonus & Unlimited Restoration\n"
        "   - Pre-existing Diseases, Waiting Periods & Health Checkups\n"
        "   - AYUSH / Alternative Hospitalization Treatments\n\n"
        "4. Output MUST strictly be valid JSON."
    )

    prompt = f"""Target Company: {company_name}
Industry Sector: {industry}
Workforce Size / Scale: {size}
Business Activities: {activities}
Operational Footprint: {footprint}
Operational Risks / Workplace Environment: {risks}

Synthesize the employee healthcare exposure profile and generate 3 to 5 highly specific medical policy search queries.
Output JSON conforming strictly to this format:
{{
    "company_name": "{company_name}",
    "workforce_profile_summary": "<Concise summary of workforce demographics, operational work environments (plants, offices, branches, field travel), and primary health risks>",
    "priority_healthcare_exposures": [
        {{
            "exposure_name": "<e.g. Industrial Plant Trauma & Critical Care>",
            "workforce_impact": "<Why this matters for their specific workforce>",
            "policy_coverage_focus": "<e.g. ICU charges, emergency air ambulance, accidental trauma care>"
        }}
    ],
    "medical_search_queries": [
        "<query 1: e.g. inpatient hospitalization ICU room rent limits>",
        "<query 2: e.g. emergency air ambulance road ambulance reimbursement>",
        "<query 3: e.g. day care treatments modern robotic surgery coverage>",
        "<query 4: e.g. pre and post hospitalization expenses medical checkups>"
    ]
}}
"""

    try:
        result = generate_structured(prompt=prompt, system_prompt=system_prompt)
        
        # Validate that queries exist and are strings
        queries = result.get("medical_search_queries", [])
        if not queries or not isinstance(queries, list):
            queries = _fallback_medical_queries(industry)
            result["medical_search_queries"] = queries

        # Sanitize queries to ensure strictly medical terms
        cleaned_queries = []
        forbidden_terms = ["machinery", "breakdown", "cargo", "transit", "cyber", "property damage", "directors and officers"]
        for q in queries:
            q_str = str(q).lower()
            if not any(term in q_str for term in forbidden_terms):
                cleaned_queries.append(str(q))
            else:
                logger.warning(f"Filtered out non-medical query generated by LLM: {q}")

        if not cleaned_queries:
            cleaned_queries = _fallback_medical_queries(industry)

        result["medical_search_queries"] = cleaned_queries[:5]
        result["company_name"] = company_name
        return result

    except Exception as e:
        logger.exception(f"Failed to formulate workforce health exposures for {company_name}: {e}")
        raise


def _fallback_medical_queries(industry: str) -> List[str]:
    """Provides deterministic medical policy queries tailored to industry profile."""
    ind_lower = str(industry).lower()
    if any(k in ind_lower for k in ["manufactur", "automotive", "factory", "industrial", "mining", "construction"]):
        return [
            "emergency air ambulance and road ambulance coverage",
            "inpatient hospitalization ICU room rent limits",
            "accidental trauma care and day care procedures",
            "pre and post hospitalization medical expenses"
        ]
    elif any(k in ind_lower for k in ["bank", "fintech", "financial", "consulting", "software", "tech", "it"]):
        return [
            "inpatient hospitalization room rent eligibility",
            "pre and post hospitalization medical expenses and health checkup",
            "modern robotic treatments and advanced surgeries",
            "pre-existing disease waiting period and cumulative bonus"
        ]
    else:
        return [
            "inpatient hospitalization expenses and ICU room charges",
            "emergency ambulance coverage road and air ambulance",
            "day care treatments and modern medical procedures",
            "pre and post hospitalization medical expenses"
        ]
