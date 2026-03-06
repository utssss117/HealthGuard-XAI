"""
routers/chat.py
────────────────
POST /chat — Health guidance LLM assistant endpoint.
Routes user queries through the Phase 4 safety filter + Groq LLM.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import ChatRequest, ChatResponse
from health_llm_assistant.assistant import ask

router = APIRouter(prefix="/chat", tags=["LLM Health Assistant"])


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    """
    Send a health-related question to the AI assistant.

    - Safety guardrails intercept emergency symptoms, medication, and diagnosis requests.
    - Safe queries are forwarded to Groq (llama-3.1-8b-instant).
    - Requires GROQ_API_KEY set in environment or .env file.
    """
    result = ask(
        user_input=body.message,
        patient_data=body.patient_data,
        conversation_history=body.history or [],
    )

    return ChatResponse(
        assistant_response=result["assistant_response"],
        safety_flag=result["safety_flag"],
        escalation_required=result["escalation_required"],
    )
