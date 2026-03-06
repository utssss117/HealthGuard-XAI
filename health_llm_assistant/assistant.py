"""
assistant.py
─────────────
Orchestrator for the HealthGuard-XAI Health Guidance LLM Assistant.

Ties together:
  - safety_filter   : Pre-screens every input for safety violations
  - prompt_builder  : Constructs contextual system + user prompts
  - llm_interface   : Sends the request to the Groq API

Returns the standard output format:
{
    "assistant_response": str,
    "safety_flag": bool,
    "escalation_required": bool
}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from health_llm_assistant.safety_filter import check_safety
from health_llm_assistant.prompt_builder import (
    build_system_prompt,
    build_user_message,
    build_conversation_history,
)
from health_llm_assistant.llm_interface import call_llm


def ask(
    user_input: str,
    patient_data: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Process a user query in the context of a patient's health data.

    Parameters
    ----------
    user_input           : Raw text message from the user.
    patient_data         : Structured patient risk data (standard Phase 4 schema).
    conversation_history : Prior conversation turns for multi-turn support.

    Returns
    -------
    dict: {assistant_response, safety_flag, escalation_required}
    """
    # Step 1: Safety screening
    safety_result = check_safety(user_input)

    if not safety_result.is_safe:
        return {
            "assistant_response": safety_result.redirection_message,
            "safety_flag": safety_result.safety_flag,
            "escalation_required": safety_result.escalation_required,
        }

    # Step 2: Build prompts
    system_prompt = build_system_prompt(patient_data)
    history = build_conversation_history(conversation_history or [])
    history.append(build_user_message(user_input))

    # Step 3: LLM call
    response_text = call_llm(
        system_prompt=system_prompt,
        messages=history,
    )

    return {
        "assistant_response": response_text,
        "safety_flag": False,
        "escalation_required": False,
    }
