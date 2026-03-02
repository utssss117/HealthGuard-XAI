"""
llm_interface.py
────────────────
Modular LLM gateway for the Phase 4 Conversational Health Assistant.

Architecture:
    - Defines a base LLMBackend protocol.
    - Implements two backends:
        • OpenAIBackend   : OpenAI ChatCompletion API (gpt-4o-mini)
        • LocalStubBackend: Deterministic rule-based response generator for
                            testing and development (no API key needed).
    - auto_select_backend() picks the best available backend at runtime.
    - call_llm() is the single public entry point used by the conversation manager.

All calls enforce: temperature ≤ 0.3, bounded max_tokens, and structured error handling.
"""

from __future__ import annotations

import os
import json
import textwrap
from typing import Dict, Any, List, Optional, Protocol


# ──────────────────────────────────────────────────────────────────────────────
# Backend Protocol (interface contract)
# ──────────────────────────────────────────────────────────────────────────────

class LLMBackend(Protocol):
    """Minimal interface any LLM backend must satisfy."""
    def complete(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str: ...


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI Backend
# ──────────────────────────────────────────────────────────────────────────────

class OpenAIBackend:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model

    def complete(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        try:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            response = self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=min(temperature, 0.3),   # Enforce ceiling
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM Error] OpenAI call failed: {e}. Please try again."


# ──────────────────────────────────────────────────────────────────────────────
# Local Stub Backend (no API key, deterministic, for testing)
# ──────────────────────────────────────────────────────────────────────────────

_STUB_RESPONSES = {
    "risk":           "Based on your health profile, the model has identified an elevated risk pattern. "
                      "Key factors include elevated glucose, BMI, and blood pressure readings. "
                      "Lifestyle modifications such as dietary changes and regular physical activity "
                      "are the most evidence-supported approaches to reduce these risks.",
    "recommendation": "Your top recommendation focuses on reducing dietary glycemic load and increasing "
                      "physical activity to 150 minutes per week. These changes have demonstrated "
                      "significant risk reduction in peer-reviewed studies.",
    "explain":        "This AI system uses a machine learning model trained on population health data "
                      "to estimate statistical risk. It does not diagnose conditions, but highlights "
                      "biomarkers that statistically correlate with increased disease probability.",
    "improve":        "Improvements in BMI, fasting glucose, and blood pressure are the highest-impact "
                      "changes for your current risk profile. Even a 5–10% reduction in body weight "
                      "can meaningfully lower cardiometabolic risk.",
    "default":        "I can help you understand your health risk profile, explain what specific biomarkers "
                      "mean, and discuss lifestyle strategies to reduce your risk. What would you like "
                      "to explore? Please remember to consult your healthcare provider for clinical guidance.",
}

def _classify_intent(user_message: str) -> str:
    """Naive keyword-based intent classifier for the stub backend."""
    msg = user_message.lower()
    if any(k in msg for k in ["risk", "probability", "chance", "likely", "score"]):
        return "risk"
    if any(k in msg for k in ["recommend", "advice", "suggest", "should i", "what can"]):
        return "recommendation"
    if any(k in msg for k in ["how does", "explain", "what is", "mean", "understand"]):
        return "explain"
    if any(k in msg for k in ["improve", "reduce", "lower", "decrease", "better", "change"]):
        return "improve"
    return "default"


class LocalStubBackend:
    """
    Deterministic rule-based backend for offline testing.
    Provides realistic-looking responses without any API calls.
    """
    def complete(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        intent = _classify_intent(last_user_msg)
        response = _STUB_RESPONSES.get(intent, _STUB_RESPONSES["default"])

        # Append standard disclaimer on educational responses
        response += (
            "\n\n*This information is educational only and does not constitute medical advice. "
            "Please consult your healthcare provider for clinical guidance.*"
        )
        return response


# ──────────────────────────────────────────────────────────────────────────────
# Backend Factory
# ──────────────────────────────────────────────────────────────────────────────

def auto_select_backend(api_key: Optional[str] = None) -> LLMBackend:
    """
    Select the best available LLM backend.
    Order: explicit api_key → OPENAI_API_KEY env var → LocalStubBackend
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        try:
            import openai
            return OpenAIBackend(api_key=key)
        except ImportError:
            pass
    return LocalStubBackend()


# ──────────────────────────────────────────────────────────────────────────────
# Public Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    messages: List[Dict[str, str]],
    backend: Optional[LLMBackend] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Call the LLM with the given system prompt and conversation history.

    Parameters
    ----------
    system_prompt : Fully assembled system context
    messages      : List of {role, content} dicts (history)
    backend       : LLMBackend instance; auto-selected if None
    temperature   : Sampling temperature (capped at 0.3)
    max_tokens    : Maximum response tokens

    Returns
    -------
    str: Generated assistant response
    """
    if backend is None:
        backend = auto_select_backend()

    return backend.complete(
        system_prompt=system_prompt,
        messages=messages,
        temperature=min(temperature, 0.3),
        max_tokens=max_tokens,
    )
