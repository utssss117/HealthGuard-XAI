"""
safety_filter.py
─────────────────
Multi-layer safety guardrail engine for the HealthGuard-XAI LLM assistant.

Layers:
    1. Emergency Detection  — life-threatening symptom mentions
    2. Medication Boundary  — dosage/prescription requests
    3. Diagnostic Boundary  — direct diagnosis solicitation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Keyword Lexicons
# ──────────────────────────────────────────────────────────────────────────────

_EMERGENCY_PATTERNS: List[str] = [
    r"\bchest\s*pain\b", r"\bchest\s*tightness\b",
    r"\bheart\s*attack\b", r"\bstroke\b", r"\bmyocardial\b",
    r"\bshortness\s*of\s*breath\b", r"\bcan'?t\s*breathe\b",
    r"\bdifficulty\s*breathing\b",
    r"\bunconscious\b", r"\bseizure\b", r"\bblacked?\s*out\b",
    r"\bsuicid(e|al)\b", r"\bself\s*harm\b",
    r"\bface\s*drooping\b", r"\barm\s*numb(ness)?\b",
    r"\bslurred\s*speech\b",
    r"\bsevere\s*(head|abdominal|stomach)\s*pain\b",
    r"\boverdos(e|ing)\b", r"\bpoison(ing|ed)?\b",
    r"\bcall\s*(911|999|112|ambulance|emergency)\b",
]

_MEDICATION_PATTERNS: List[str] = [
    r"\bhow\s*much\b.{0,30}\b(mg|milligram|dose|pill|tablet|capsule)\b",
    r"\b(dose|dosage|dosing)\s*of\b",
    r"\bprescri(be|ption|bed)\b",
    r"\bshould\s*i\s*take\b",
    r"\b(metformin|insulin|aspirin|atorvastatin|lisinopril|amlodipine|warfarin|ozempic|jardiance)\b",
    r"\b(antibiotic|statin|beta.?blocker|ace.?inhibitor|diuretic)\b",
    r"\bstop\s*(taking|my)\b.{0,20}\b(medic|pill|tablet|drug)\b",
]

_DIAGNOSTIC_PATTERNS: List[str] = [
    r"\bdo\s*i\s*have\b",
    r"\bam\s*i\s*(diabetic|hypertensive|sick|dying|at\s*risk)\b",
    r"\bdiagnos(e|is|ed)\s*(me|my)\b",
    r"\bwhat\s*(disease|condition|illness)\s*(do\s*i|have\s*i)\b",
    r"\bis\s*this\s*(cancer|diabetes|heart\s*disease|serious|fatal)\b",
    r"\bconfirm\b.{0,20}\b(diagnosis|disease|condition)\b",
]


# ──────────────────────────────────────────────────────────────────────────────
# Response Templates
# ──────────────────────────────────────────────────────────────────────────────

_EMERGENCY_RESPONSE = (
    "🚨 **Immediate Action Required**: Your message mentions symptoms that may "
    "indicate a medical emergency. Please call your local emergency services "
    "(911 / 999 / 112) or go to the nearest emergency room immediately. "
    "Do not rely on this assistant in an emergency. Your safety is the priority."
)

_MEDICATION_RESPONSE = (
    "I'm not able to provide medication dosage, prescription, or advice on "
    "starting, stopping, or adjusting any medication. This falls outside the "
    "safe boundaries of this educational assistant. Please consult your physician "
    "or pharmacist for all medication-related questions.\n\n"
    "*This information is for educational purposes and should not replace "
    "professional medical advice.*"
)

_DIAGNOSTIC_RESPONSE = (
    "I'm not able to provide a diagnosis or confirm a medical condition. This "
    "assistant uses predictive models to estimate statistical risk based on "
    "biomarker patterns — this is NOT equivalent to a clinical diagnosis. "
    "Please consult a qualified healthcare professional for an accurate diagnosis.\n\n"
    "*This information is for educational purposes and should not replace "
    "professional medical advice.*"
)


# ──────────────────────────────────────────────────────────────────────────────
# Data Structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SafetyResult:
    is_safe: bool
    safety_flag: bool
    escalation_required: bool
    triggered_layer: Optional[str]
    redirection_message: Optional[str]
    matched_patterns: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def _scan(text: str, patterns: List[str]) -> List[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def check_safety(user_input: str) -> SafetyResult:
    """
    Run three safety layers on user input (priority order: Emergency → Medication → Diagnostic).
    Returns a SafetyResult with a safe redirection if any layer triggers.
    """
    text = user_input.strip()

    matches = _scan(text, _EMERGENCY_PATTERNS)
    if matches:
        return SafetyResult(
            is_safe=False, safety_flag=True, escalation_required=True,
            triggered_layer="emergency", redirection_message=_EMERGENCY_RESPONSE,
            matched_patterns=matches,
        )

    matches = _scan(text, _MEDICATION_PATTERNS)
    if matches:
        return SafetyResult(
            is_safe=False, safety_flag=True, escalation_required=False,
            triggered_layer="medication", redirection_message=_MEDICATION_RESPONSE,
            matched_patterns=matches,
        )

    matches = _scan(text, _DIAGNOSTIC_PATTERNS)
    if matches:
        return SafetyResult(
            is_safe=False, safety_flag=True, escalation_required=False,
            triggered_layer="diagnostic", redirection_message=_DIAGNOSTIC_RESPONSE,
            matched_patterns=matches,
        )

    return SafetyResult(
        is_safe=True, safety_flag=False, escalation_required=False,
        triggered_layer=None, redirection_message=None,
    )
