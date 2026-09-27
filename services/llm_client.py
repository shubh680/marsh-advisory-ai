"""
LLM Client Abstraction Layer.
Decouples LLM provider (Groq, OpenAI, GCP managed endpoint) from the application.
Configured via Django settings and environment variables.
"""
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from django.conf import settings
from openai import OpenAI

logger = logging.getLogger(__name__)

_LLM_CLIENT_INSTANCE: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """
    Returns configured OpenAI-compatible client singleton.
    """
    global _LLM_CLIENT_INSTANCE
    if _LLM_CLIENT_INSTANCE is None:
        base_url = getattr(settings, 'LLM_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai/')
        api_key = getattr(settings, 'LLM_API_KEY', '')
        if not api_key:
            raise ValueError("LLM_API_KEY is not configured in settings or .env file.")
        _LLM_CLIENT_INSTANCE = OpenAI(base_url=base_url, api_key=api_key)
    return _LLM_CLIENT_INSTANCE


def get_model_name() -> str:
    """
    Returns the configured LLM model name.
    """
    return getattr(settings, 'LLM_MODEL', 'gemini-2.5-flash')


def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> str:
    """
    Basic text generation using the configured LLM.
    """
    client = get_openai_client()
    model = get_model_name()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        message = response.choices[0].message
        return (message.content or "").strip()
    except Exception as e:
        logger.exception(f"LLM generation failed with model {model}: {e}")
        raise


def generate_structured(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048
) -> Dict[str, Any]:
    """
    Generates and parses structured JSON output from LLM.
    """
    client = get_openai_client()
    model = get_model_name()

    enforced_system = (
        (system_prompt or "") + 
        "\nIMPORTANT: Your output MUST be a single valid JSON object. "
        "Do not include any surrounding markdown text, explanations, or commentary outside the JSON."
    ).strip()

    messages = [
        {"role": "system", "content": enforced_system},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        content = (response.choices[0].message.content or "").strip()
        
        # Parse JSON directly or extract first json block
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError(f"Could not parse valid JSON from response: {content[:200]}...")

    except Exception as e:
        logger.exception(f"LLM structured generation failed: {e}")
        raise


def call_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tool_choice: str = "auto",
    temperature: float = 0.2,
    max_tokens: int = 2048
):
    """
    Executes a chat completion call with tool definitions enabled.
    Returns the raw ChatCompletionMessage object with tool_calls and content.
    """
    client = get_openai_client()
    model = get_model_name()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message
    except Exception as e:
        logger.warning(f"LLM tool call failed: {e}")
        raise

