"""
Company Research Agent using GPT-OSS-120B and external research tools.
Autonomously searches the web and opens URLs to build a grounded corporate profile
with business activities, operational footprint, and operational risk exposures.
Explicitly flags assumptions without hallucinations when information is missing.
"""
import json
import logging
import urllib.parse
from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup

from services.llm_client import call_with_tools, generate_structured

logger = logging.getLogger(__name__)


# ==================================================
# RESEARCH TOOLS
# ==================================================

def search_web(query: str, max_results: int = 4) -> str:
    """
    Searches the web for company information, industry background, and risk exposures.
    Combines search engine and encyclopedia queries for high factual reliability.
    """
    results = []

    # 1. Wikipedia Search API
    try:
        wiki_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
        )
        resp = requests.get(wiki_url, headers={'User-Agent': 'MarshPitchAdvisor/1.0'}, timeout=6)
        if resp.status_code == 200:
            items = resp.json().get('query', {}).get('search', [])
            for item in items[:max_results]:
                title = item.get('title', '')
                snippet = BeautifulSoup(item.get('snippet', ''), 'html.parser').get_text()
                url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
    except Exception as e:
        logger.warning(f"Wikipedia search failed for '{query}': {e}")

    # 2. DuckDuckGo Instant Answer / HTML Search fallback
    try:
        ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
        resp = requests.get(ddg_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get('AbstractText', '')
            if abstract:
                src_url = data.get('AbstractURL', '')
                results.append(f"Title: {data.get('Heading', 'Summary')}\nURL: {src_url}\nSnippet: {abstract}")
    except Exception as e:
        logger.debug(f"DuckDuckGo search fallback skipped for '{query}': {e}")

    if not results:
        return f"No direct search results found for query: '{query}'."

    return "\n\n---\n\n".join(results)


def open_url(url: str, max_chars: int = 3500) -> str:
    """
    Fetches a web page and extracts readable text paragraphs.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return f"Failed to retrieve URL {url}: HTTP status {resp.status_code}"

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove scripts, styles, and navigation
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        # Extract title and body text
        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
        paragraphs = [p.get_text().strip() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']) if len(p.get_text().strip()) > 30]
        text_content = "\n\n".join(paragraphs)

        truncated = text_content[:max_chars]
        return f"Page Title: {title}\nURL: {url}\n\nContent:\n{truncated}"

    except Exception as e:
        return f"Error opening URL {url}: {str(e)}"


# Tool definitions for LLM function calling
RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the web for factual information regarding a company, its industry, workforce size, operational activities, physical footprint, and risk exposures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query (e.g. 'Tata Motors manufacturing plants footprint and employee count')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Fetches and reads the textual content of a specific web URL discovered during research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The exact web URL to open and read"
                    },
                    "loc": {
                        "type": "integer",
                        "description": "Optional location pointer"
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Optional cursor position"
                    }
                }
            }
        }
    }

]


# ==================================================
# RESEARCH AGENT LOOP
# ==================================================

def research_company(company_name: str, max_iterations: int = 4) -> Dict[str, Any]:
    """
    Executes an autonomous research workflow for a given company name.
    The agent uses external research tools (search_web, open_url) to find factual information.
    
    Fields researched:
    - Industry sector (e.g. automotive manufacturing, fintech, pharmaceuticals)
    - Company size (employees, revenue/scale, number of sites/locations)
    - Business activities (what the company actually makes/sells/does)
    - Operational footprint (countries, factories, offices, warehouses, distribution network)
    - Key risks & exposures (operational hazards, workforce health, supply chain, business interruption)
    - Evidence/sources (URL + source title)
    - Unavailable facts (explicitly labeled rather than guessing)
    """
    if not company_name or not company_name.strip():
        return {
            "company_name": "",
            "error": "Company name is required",
            "assumption": True,
            "reason": "Empty company name provided"
        }

    company_name = company_name.strip()
    sources_collected: List[Dict[str, str]] = []
    seen_urls = set()
    research_log: List[str] = []

    def record_source(url_str: str, title_str: str = ""):
        if url_str and url_str not in seen_urls:
            seen_urls.add(url_str)
            sources_collected.append({
                "title": title_str or url_str.split("/")[-1].replace("_", " "),
                "url": url_str
            })

    # 1. Multi-facet automated exploratory gathering for factual baseline
    queries = [
        f"{company_name} industry sector business activities products services",
        f"{company_name} employees workforce scale revenue sites locations",
        f"{company_name} operational footprint factories offices warehouses countries distribution",
        f"{company_name} operational risks manufacturing workplace safety hazards supply chain"
    ]

    for q in queries:
        summary = search_web(q, max_results=3)
        research_log.append(f"Web Search ({q}):\n{summary}")
        current_title = ""
        for line in summary.splitlines():
            if line.startswith("Title: "):
                current_title = line.replace("Title: ", "").strip()
            elif line.startswith("URL: "):
                url_found = line.replace("URL: ", "").strip()
                record_source(url_found, current_title)

                # If exact company page found on Wikipedia, read it
                sanitized_name = company_name.replace(" ", "_").lower()
                if "wikipedia.org/wiki/" in url_found.lower() and sanitized_name in url_found.lower() and len(seen_urls) <= 6:
                    page_content = open_url(url_found, max_chars=3500)
                    research_log.append(f"Primary Document ({url_found}):\n{page_content}")

    system_prompt = (
        "You are an expert commercial insurance underwriting and corporate risk research agent for Marsh advisors.\n"
        "Your task is to gather verified external facts on the target company across these critical underwriting dimensions:\n\n"
        "1. Industry sector: Specific sector (e.g. automotive manufacturing, fintech, pharmaceuticals).\n"
        "2. Company size: Employees/workforce count, revenue/scale if available, and number of locations/sites.\n"
        "3. Business activities: What the company actually makes, sells, or does in daily commercial operations.\n"
        "4. Operational footprint: Geographic reach across countries, factories, corporate offices, warehouses, and distribution networks.\n"
        "5. Key risks & exposures: Concrete risks and operational liabilities that logically arise from the company's actual operations (e.g., plant machinery hazards, employee health/hospitalization, supply chain disruption, product defect liability, business interruption).\n"
        "6. Evidence/sources: URL and source title for every verified fact.\n"
        "7. Unavailable facts: If reliable data for any field is not found in external research, EXPLICITLY state that it is unavailable or format it as an assumption object:\n"
        "   {\"assumption\": true, \"reason\": \"Reliable data on ... was unavailable from public sources.\"}\n"
        "   NEVER guess, invent, or hallucinate facts.\n\n"
        "RESEARCH PROTOCOL:\n"
        "- Use the tools ('search_web' and 'open_url') to look up missing details.\n"
        "- Do NOT invent facts. Rely strictly on external findings.\n"
    )

    initial_findings = "\n\n".join(research_log)
    user_prompt = (
        f"Target Company: '{company_name}'.\n"
        f"Initial search results:\n{initial_findings}\n\n"
        "Use your research tools to verify industry sector, company size, business activities, operational footprint, and key risks if needed."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 2. Agent tool execution loop
    for iteration in range(max_iterations):
        try:
            message = call_with_tools(messages=messages, tools=RESEARCH_TOOLS)
        except Exception as e:
            logger.warning(f"Tool call request ended or failed: {e}")
            break

        tool_calls = getattr(message, 'tool_calls', None)
        if not tool_calls:
            # Model finished research
            if message.content:
                research_log.append(f"Agent Analysis:\n{message.content}")
            break

        messages.append(message)

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            logger.info(f"Executing tool '{fn_name}' with args: {fn_args}")

            if fn_name == "search_web":
                query_str = fn_args.get("query", company_name)
                tool_output = search_web(query=query_str)
                current_title = ""
                for line in tool_output.splitlines():
                    if line.startswith("Title: "):
                        current_title = line.replace("Title: ", "").strip()
                    elif line.startswith("URL: "):
                        record_source(line.replace("URL: ", "").strip(), current_title)
            elif fn_name == "open_url":
                url = fn_args.get("url", "")
                if url:
                    record_source(url)
                    tool_output = open_url(url=url)
                else:
                    tool_output = "No URL provided."
            else:
                tool_output = f"Unknown tool: {fn_name}"

            research_log.append(f"Tool: {fn_name}({fn_args})\nOutput:\n{tool_output}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output[:3000]
            })

    # 3. Final Synthesis Step into strict JSON schema matching user requirements
    compiled_research = "\n\n---\n\n".join(research_log)
    synthesis_prompt = f"""Target Company: {company_name}

Here is the verified external research data collected:
{compiled_research}

Extract and synthesize a structured company profile JSON conforming strictly to this format:
{{
    "company_name": "{company_name}",
    "industry_sector": "<specific industry sector e.g. automotive manufacturing, or {{\"assumption\": true, \"reason\": \"...\"}}>",
    "company_size": "<employees, revenue/scale, number of locations/sites, or {{\"assumption\": true, \"reason\": \"...\"}}>",
    "business_activities": ["what the company actually makes/sells/does", ...],
    "operational_footprint": ["countries, factories, offices, warehouses, distribution network details", ...],
    "key_risks_and_exposures": ["risks that logically arise from actual operations (e.g. factory machinery injury, supply chain, business interruption)", ...],
    "sources": [
        {{"title": "<source title>", "url": "<source url>"}}
    ]
}}

CRITICAL RULES:
1. Every factual detail must come from the research findings.
2. If reliable information for any attribute is NOT found, DO NOT fabricate it; format that field as:
   {{"assumption": true, "reason": "Reliable data was unavailable from public sources."}}
3. For operational footprint, list physical locations, manufacturing plants, warehouses, or global presence found.
4. Do NOT invent insurance policy benefits (policy knowledge belongs strictly to Phase 2/4).
"""

    try:
        final_profile = generate_structured(
            prompt=synthesis_prompt,
            system_prompt="You are a strict corporate risk analyst. Output only valid JSON strictly matching the company profile schema."
        )

        if not final_profile.get("company_name"):
            final_profile["company_name"] = company_name

        # Ensure backward compatibility aliases
        if "industry_sector" in final_profile and "industry" not in final_profile:
            final_profile["industry"] = final_profile["industry_sector"]
        elif "industry" in final_profile and "industry_sector" not in final_profile:
            final_profile["industry_sector"] = final_profile["industry"]

        if "key_risks_and_exposures" in final_profile and "key_risks" not in final_profile:
            final_profile["key_risks"] = final_profile["key_risks_and_exposures"]
        elif "key_risks" in final_profile and "key_risks_and_exposures" not in final_profile:
            final_profile["key_risks_and_exposures"] = final_profile["key_risks"]

        # Ensure sources are formatted as list of dicts with title and url
        raw_sources = final_profile.get("sources", [])
        formatted_sources = []
        seen_res_urls = set()

        for s in raw_sources:
            if isinstance(s, dict) and s.get("url"):
                formatted_sources.append(s)
                seen_res_urls.add(s["url"])
            elif isinstance(s, str) and s.startswith("http"):
                formatted_sources.append({"title": s.split("/")[-1].replace("_", " "), "url": s})
                seen_res_urls.add(s)

        # Merge with collected sources
        for s in sources_collected:
            if s["url"] not in seen_res_urls:
                formatted_sources.append(s)
                seen_res_urls.add(s["url"])

        final_profile["sources"] = formatted_sources
        return final_profile

    except Exception as e:
        logger.exception(f"Failed to synthesize structured profile for {company_name}: {e}")
        raise


def generateCompanyProfile(company_name: str) -> Dict[str, Any]:
    """
    Marsh Case Study Objective 1.2:
    Gathers company info (industry, size, key risks); falls back to clearly-labelled assumptions if data is unavailable.
    """
    return research_company(company_name)
