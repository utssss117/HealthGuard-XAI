"""
safety_guardrails.py
─────────────────────
Multi-layer safety guardrail engine for the HealthGuard-XAI conversational assistant.

Implements three enforcement layers:
    Layer 1 — Emergency Detection  : detects life-threatening symptom mentions
    Layer 2 — Medication Boundary  : detects dosage/prescription requests
    Layer 3 — Diagnostic Boundary  : detects attempts to solicit diagnoses

Each layer returns a SafetyResult with structured redirection messaging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Keyword Lexicons
# ──────────────────────────────────────────────────────────────────────────────

# Layer 1: Emergency / life-threatening symptoms
_EMERGENCY_PATTERNS: List[str] = [
    r"\bchest\s*pain\b", r"\bchest\s*tightness\b",
    r"\bstroke\b", r"\bheart\s*attack\b", r"\bmyocardial\b",
    r"\bsevere\s*bleeding\b", r"\bheavy\s*bleeding\b",
    r"\bshortness\s*of\s*breath\b", r"\bcan'?t\s*breathe\b",
    r"\bdifficulty\s*breathing\b",
    r"\bunconscious\b", r"\bblacked?\s*out\b", r"\bseizure\b",
    r"\bsuicid(e|al)\b", r"\bself\s*harm\b",
    r"\bparalysis\b", r"\barm\s*numb\b", r"\bface\s*drooping\b",
    r"\bslurred\s*speech\b",
    r"\bsevere\s*(head|abdominal|stomach)\s*pain\b",
    r"\boverdos(e|ing)\b", r"\bpoison(ing|ed)?\b",
    r"\bcall\s*(911|999|112|ambulance|emergency)\b",
]

# Layer 2: Medication / dosage requests
_MEDICATION_PATTERNS: List[str] = [
    r"\bhow\s*much\b.{0,30}\b(mg|milligram|dose|pill|tablet|capsule)\b",
    r"\b(dose|dosage|dosing)\s*of\b",
    r"\bshould\s*i\s*take\b",
    r"\bprescri(be|ption|bed)\b",
    r"\b(metformin|insulin|aspirin|atorvastatin|lisinopril|amlodipine|warfarin|ozempic|wegovy|jardiance)\b",
    r"\b(antibiotic|statin|beta.?blocker|ace.?inhibitor|diuretic)\b",
    r"\bstop\s*(taking|my)\b.{0,20}\b(medic|pill|tablet|drug)\b",
    r"\b(medication|medicine|drug)\s*(question|advice|recommendation)\b",
]

# Layer 3: Diagnostic requests
_DIAGNOSTIC_PATTERNS: List[str] = [
    r"\bdo\s*i\s*have\b",
    r"\bam\s*i\s*(diabetic|hypertensive|sick|dying|at\s*risk)\b",
    r"\bdiagnos(e|is|ed)\s*(me|my)\b",
    r"\bwhat\s*(disease|condition|illness)\s*(do\s*i|have\s*i)\b",
    r"\bis\s*this\s*(cancer|diabetes|heart\s*disease|serious|fatal)\b",
    r"\bconfirm\b.{0,20}\b(diagnosis|disease|condition)\b",
]

# ──────────────────────────────────────────────────────────────────────────────
# Safe Redirection Responses
# ──────────────────────────────────────────────────────────────────────────────

_EMERGENCY_RESPONSE = (
    "🚨 **Immediate Action Required**: Your message mentions symptoms that may indicate "
    "a medical emergency. Please call your local emergency services (911 / 999 / 112) "
    "or go to the nearest emergency room immediately. Do not rely on this assistant "
    "in an emergency situation. Your safety is the highest priority."
)

_MEDICATION_RESPONSE = (
    "I'm not able to provide medication dosage, prescription, or advice on starting, "
    "stopping, or adjusting any medication. This falls outside the safe boundaries of "
    "this educational assistant. Please consult your physician, pharmacist, or a "
    "licensed healthcare provider for all medication-related questions."
)

_DIAGNOSTIC_RESPONSE = (
    "I'm not able to provide a diagnosis or confirm a medical condition. This assistant "
    "uses predictive models to estimate statistical risk based on biomarker patterns — "
    "this is not equivalent to a clinical diagnosis. For an accurate diagnosis, please "
    "consult a qualified healthcare professional who can evaluate your full medical history."
)


# ──────────────────────────────────────────────────────────────────────────────
# Data Structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SafetyResult:
    is_safe: bool                          # True if input passed all checks
    safety_flag: bool                      # True if any guardrail triggered
    escalation_required: bool              # True for emergency-tier triggers
    triggered_layer: Optional[str]         # "emergency" | "medication" | "diagnostic" | None
    redirection_message: Optional[str]     # Safe response to return to user
    matched_patterns: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Core Detection Functions
# ──────────────────────────────────────────────────────────────────────────────

def _scan(text: str, patterns: List[str]) -> List[str]:
    """Return all patterns that match in text (case-insensitive)."""
    matches = []
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            matches.append(pat)
    return matches


def check_safety(user_input: str) -> SafetyResult:
    """
    Run all three safety layers on the user input.
    Layers are checked in priority order: Emergency → Medication → Diagnostic.

    Returns a SafetyResult with a structured redirection if any layer triggers.
    """
    text = user_input.strip()

    # Layer 1: Emergency
    matches = _scan(text, _EMERGENCY_PATTERNS)
    if matches:
        return SafetyResult(
            is_safe=False,
            safety_flag=True,
            escalation_required=True,
            triggered_layer="emergency",
            redirection_message=_EMERGENCY_RESPONSE,
            matched_patterns=matches,
        )

    # Layer 2: Medication
    matches = _scan(text, _MEDICATION_PATTERNS)
    if matches:
        return SafetyResult(
            is_safe=False,
            safety_flag=True,
            escalation_required=False,
            triggered_layer="medication",
            redirection_message=_MEDICATION_RESPONSE,
            matched_patterns=matches,
        )

    # Layer 3: Diagnostic
    matches = _scan(text, _DIAGNOSTIC_PATTERNS)
    if matches:
        return SafetyResult(
            is_safe=False,
            safety_flag=True,
            escalation_required=False,
            triggered_layer="diagnostic",
            redirection_message=_DIAGNOSTIC_RESPONSE,
            matched_patterns=matches,
        )

    # All clear
    return SafetyResult(
        is_safe=True,
        safety_flag=False,
        escalation_required=False,
        triggered_layer=None,
        redirection_message=None,
    )


def build_safety_response(safety_result: SafetyResult) -> dict:
    """
    Format a SafetyResult into the standard Phase 4 output schema.
    """
    return {
        "response": safety_result.redirection_message,
        "safety_flag": safety_result.safety_flag,
        "escalation_required": safety_result.escalation_required,
        "triggered_layer": safety_result.triggered_layer,
    }
