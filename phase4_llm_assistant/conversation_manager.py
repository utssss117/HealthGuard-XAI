"""
conversation_manager.py
────────────────────────
Stateful dialogue manager for the Phase 4 Conversational Health Assistant.

Responsibilities:
    - Maintains per-session conversation history
    - Injects risk context into system prompt on first turn
    - Routes each user message through the safety guardrail layer
    - Delegates safe messages to the LLM via llm_interface.call_llm()
    - Logs every turn (user + assistant + safety metadata) to a JSON log file
    - Supports dynamic explanation mode switching per turn

Usage:
    session = ConversationSession(context, explanation_mode="simple")
    result  = session.chat("What does my glucose level mean?")
    # result = {"response": "...", "safety_flag": false, "escalation_required": false}
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from phase4_llm_assistant.safety_guardrails import check_safety, build_safety_response
from phase4_llm_assistant.prompt_templates import (
    build_context_block,
    build_system_prompt,
    EXPLANATION_MODES,
    DEFAULT_EXPLANATION_MODE,
)
from phase4_llm_assistant.llm_interface import call_llm, auto_select_backend, LLMBackend

# ──────────────────────────────────────────────────────────────────────────────
# Log Directory
# ──────────────────────────────────────────────────────────────────────────────

_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Conversation Session
# ──────────────────────────────────────────────────────────────────────────────

class ConversationSession:
    """
    Manages one patient conversation session end-to-end.
    """

    def __init__(
        self,
        patient_context: Dict[str, Any],
        explanation_mode: str = DEFAULT_EXPLANATION_MODE,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
        max_history_turns: int = 10,
    ):
        self.session_id        = session_id or str(uuid.uuid4())[:8]
        self.explanation_mode  = explanation_mode
        self.max_history_turns = max_history_turns
        self._is_first_turn    = True
        self._history: List[Dict[str, str]] = []    # [{role, content}, ...]
        self._log: List[Dict[str, Any]] = []

        # Build LLM backend
        self._backend: LLMBackend = auto_select_backend(api_key)

        # Extract and store context fields
        self._predicted_risks       = patient_context.get("predicted_risks", {})
        self._risk_level            = patient_context.get("risk_level", "unknown")
        self._top_positive          = patient_context.get("top_positive_risk_factors", [])
        self._protective            = patient_context.get("protective_factors", [])
        self._recommendations       = patient_context.get("prioritized_recommendations", [])
        self._patient_profile       = patient_context.get("patient_profile", {})

        # Build the context block once (injected into every system prompt)
        self._context_block = build_context_block(
            predicted_risks=self._predicted_risks,
            risk_level=self._risk_level,
            top_positive_risk_factors=self._top_positive,
            protective_factors=self._protective,
            prioritized_recommendations=self._recommendations,
            patient_profile=self._patient_profile,
        )

        # Initialise session log file
        self._log_path = os.path.join(_LOG_DIR, f"session_{self.session_id}.json")

    # ── Public Interface ─────────────────────────────────────────────────────

    def chat(
        self,
        user_input: str,
        explanation_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process one user turn and return the structured response.

        Parameters
        ----------
        user_input        : Raw user message
        explanation_mode  : Override mode for this turn (optional)

        Returns
        -------
        {response, safety_flag, escalation_required}
        """
        mode = explanation_mode or self.explanation_mode

        # 1. Safety check first
        safety = check_safety(user_input)
        if not safety.is_safe:
            result = build_safety_response(safety)
            self._log_turn(user_input, result["response"], safety_metadata=result)
            return result

        # 2. Build system prompt for this turn
        system_prompt = build_system_prompt(
            context_block=self._context_block,
            explanation_mode=mode,
            is_first_turn=self._is_first_turn,
        )

        # 3. Add user message to history
        self._history.append({"role": "user", "content": user_input})

        # 4. Call LLM
        assistant_response = call_llm(
            system_prompt=system_prompt,
            messages=self._history,
            backend=self._backend,
            temperature=0.2,
            max_tokens=512,
        )

        # 5. Store assistant response in history
        self._history.append({"role": "assistant", "content": assistant_response})

        # 6. Trim history to max_history_turns (keep pairs)
        if len(self._history) > self.max_history_turns * 2:
            self._history = self._history[-(self.max_history_turns * 2):]

        self._is_first_turn = False

        result = {
            "response": assistant_response,
            "safety_flag": False,
            "escalation_required": False,
        }
        self._log_turn(user_input, assistant_response)
        return result

    def set_explanation_mode(self, mode: str) -> None:
        """Switch the default explanation mode for the session."""
        if mode not in EXPLANATION_MODES:
            raise ValueError(f"Unknown explanation mode '{mode}'. Options: {list(EXPLANATION_MODES)}")
        self.explanation_mode = mode

    def get_history(self) -> List[Dict[str, str]]:
        """Return the full conversation history."""
        return list(self._history)

    def reset(self) -> None:
        """Clear conversation history (keep context and session ID)."""
        self._history.clear()
        self._is_first_turn = True

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_turn(
        self,
        user_input: str,
        response: str,
        safety_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "turn": len(self._log) + 1,
            "explanation_mode": self.explanation_mode,
            "user_input": user_input,
            "assistant_response": response,
            "safety_triggered": safety_metadata is not None,
            "safety_metadata": safety_metadata or {},
        }
        self._log.append(entry)
        # Overwrite log file with full session log
        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump(self._log, f, indent=2, ensure_ascii=False)

    def get_log(self) -> List[Dict[str, Any]]:
        """Return the full conversation log."""
        return list(self._log)
