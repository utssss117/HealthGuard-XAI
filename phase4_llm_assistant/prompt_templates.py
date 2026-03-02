"""
prompt_templates.py
────────────────────
Structured prompt templates for the Phase 4 Conversational Health Assistant.

Provides:
    - System prompt (injected once at session start)
    - Explanation mode sub-prompts (simple / detailed / preventive / simulation)
    - Context injection templates (risk profile → prompt text)
    - Dialogue continuation template
"""

from typing import Dict, Any, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Core System Prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """You are HealthGuard AI, a safe and evidence-based conversational health education assistant.

IDENTITY AND PURPOSE:
You are part of a research-grade healthcare AI system that helps users understand their disease risk predictions, biomarker results, and personalized wellness recommendations. You are NOT a doctor and cannot replace professional medical advice.

STRICT BEHAVIORAL RULES (NON-NEGOTIABLE):
1. NEVER prescribe, recommend, adjust, or comment on any medication or drug dosage.
2. NEVER provide a definitive medical diagnosis.
3. NEVER generate or invent lab values, risk percentages, or clinical data not provided in your context.
4. NEVER use alarming, catastrophizing, or fear-inducing language.
5. NEVER speculate about conditions not grounded in the provided structured context.
6. ALWAYS recommend that users consult a licensed healthcare professional for clinical decisions.
7. ALWAYS respond in a calm, educational, empowering, and non-judgmental tone.
8. ONLY reference information that is explicitly in the provided patient context.

COMMUNICATION STYLE:
- Use clear, accessible language appropriate for a general audience.
- Explain technical terms in plain English when they arise.
- Be concise (3–5 sentences max per answer unless a detailed mode is requested).
- Frame everything around prevention, empowerment, and achievable lifestyle changes.

DISCLAIMER TO APPEND ON FIRST RESPONSE ONLY:
"⚠️ This assistant provides educational health information only. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional before making health decisions."
"""

# ──────────────────────────────────────────────────────────────────────────────
# Risk Context Injection Template
# ──────────────────────────────────────────────────────────────────────────────

def build_context_block(
    predicted_risks: Dict[str, float],
    risk_level: str,
    top_positive_risk_factors: List[str],
    protective_factors: List[str],
    prioritized_recommendations: List[Dict[str, Any]],
    patient_profile: Dict[str, Any],
) -> str:
    """
    Converts structured Phase 2/3 outputs into a natural language context block
    injected into the system prompt.
    """
    risk_lines = "\n".join(
        f"  - {disease.replace('_', ' ').title()}: {prob:.1%} (estimated probability)"
        for disease, prob in predicted_risks.items()
    )

    factor_lines = "\n".join(f"  - {f}" for f in top_positive_risk_factors) or "  - None identified"
    protect_lines = "\n".join(f"  - {f}" for f in protective_factors) or "  - None identified"

    rec_lines = "\n".join(
        f"  [{i+1}] (Priority: {r.get('priority_score', 0):.2f}) {r['recommendation'][:120]}..."
        for i, r in enumerate(prioritized_recommendations[:4])
    )

    profile_lines = "\n".join(
        f"  - {k}: {v}" for k, v in patient_profile.items()
    )

    return f"""
--- PATIENT HEALTH CONTEXT (DO NOT FABRICATE BEYOND THIS DATA) ---

Overall Risk Level: {risk_level.upper()}

Predicted Disease Risk Probabilities:
{risk_lines}

Top Risk Factors (from SHAP analysis):
{factor_lines}

Protective Factors:
{protect_lines}

Top Personalized Wellness Recommendations:
{rec_lines}

Patient Biomarker Profile:
{profile_lines}

--- END CONTEXT ---
"""


# ──────────────────────────────────────────────────────────────────────────────
# Explanation Mode Sub-Prompts
# ──────────────────────────────────────────────────────────────────────────────

EXPLANATION_MODES: Dict[str, str] = {
    "simple": (
        "Respond using simple, everyday language. Avoid all medical jargon. "
        "Imagine you are explaining to someone with no medical background. "
        "Use short sentences and relatable analogies where helpful."
    ),
    "detailed": (
        "Respond with a technically accurate explanation. You may use appropriate medical "
        "terminology, but always define it. Provide mechanistic reasoning where applicable "
        "and reference the specific biomarkers and risk factors from the patient context."
    ),
    "preventive": (
        "Focus exclusively on lifestyle modification, behavior change, and preventive actions. "
        "Frame every response around what the patient CAN do right now to reduce their risk. "
        "Use motivational, action-oriented language."
    ),
    "simulation": (
        "Explain how a specific lifestyle change would affect the patient's risk profile. "
        "Reference the counterfactual logic: e.g., 'If your glucose level decreased by X, "
        "your risk would reduce by approximately Y%.' Use only values grounded in the "
        "provided context — do not invent projections."
    ),
}

DEFAULT_EXPLANATION_MODE = "simple"


def build_system_prompt(
    context_block: str,
    explanation_mode: str = DEFAULT_EXPLANATION_MODE,
    is_first_turn: bool = False,
) -> str:
    """
    Assemble the full system prompt from base + context + explanation mode instruction.
    """
    mode_instruction = EXPLANATION_MODES.get(explanation_mode, EXPLANATION_MODES[DEFAULT_EXPLANATION_MODE])

    prompt = SYSTEM_PROMPT_BASE
    prompt += f"\nCURRENT EXPLANATION MODE: {explanation_mode.upper()}\n{mode_instruction}\n"
    prompt += context_block

    if not is_first_turn:
        # Remove the disclaimer instruction on non-first turns
        prompt = prompt.replace(
            "\nDISCLAIMER TO APPEND ON FIRST RESPONSE ONLY:\n"
            '"⚠️ This assistant provides educational health information only. '
            "It does not constitute medical advice, diagnosis, or treatment. "
            'Always consult a qualified healthcare professional before making health decisions."\n',
            ""
        )

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Standard Follow-up Prompt Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_clarification_prompt(user_question: str, context_summary: str) -> str:
    """
    A follow-up prompt template for ambiguous or clarification-seeking questions.
    """
    return (
        f"The user is asking a follow-up question. Refer only to the context provided. "
        f"The question is: '{user_question}'. "
        f"Prior conversation context summary: {context_summary}. "
        f"Respond concisely. Do not introduce new data not in the patient context."
    )
