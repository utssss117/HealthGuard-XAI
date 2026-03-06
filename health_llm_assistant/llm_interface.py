"""
llm_interface.py
─────────────────
Groq API gateway for the HealthGuard-XAI Health Guidance Assistant.

Uses the Groq Python SDK with model: llama-3.1-8b-instant.
Reads GROQ_API_KEY from the environment.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads GROQ_API_KEY from .env automatically
except ImportError:
    pass  # dotenv not installed; fall back to environment variable

from groq import Groq

_DEFAULT_MODEL = "llama-3.1-8b-instant"
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_MAX_TOKENS = 500


def _get_client() -> Groq:
    """Instantiate the Groq client, raising a clear error if the key is missing."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Export it before running the assistant:\n"
            "  export GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=api_key)


def call_llm(
    system_prompt: str,
    messages: List[Dict[str, str]],
    model: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """
    Call the Groq LLM with a system prompt and conversation history.

    Parameters
    ----------
    system_prompt : str
        Fully assembled system context injected as the system role.
    messages      : list of {role, content} dicts representing the conversation.
    model         : Groq model identifier (default: llama-3.1-8b-instant).
    temperature   : Sampling temperature (enforced max 0.3 for safety).
    max_tokens    : Maximum number of tokens in the response.

    Returns
    -------
    str: The LLM-generated assistant response.
    """
    client = _get_client()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=min(temperature, 0.3),
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"[LLM Error] Groq API call failed: {exc}. Please try again later."
