"""
prompt_builder.py
─────────────────
Builds structured, context-aware system and user prompts for the health LLM assistant.
Converts structured patient risk data into an informative LLM system context.
"""

from __future__ import annotations

from typing import Any, Dict, List


_DISCLAIMER = (
    "This information is for educational purposes and should not replace "
    "professional medical advice."
)

_ASSISTANT_PERSONA = """You are a compassionate, knowledgeable Health Guidance Assistant.

Your role is to:
- Clearly explain health risks in plain, accessible language
- Provide evidence-based lifestyle and preventive recommendations
- Use a calm, supportive, and educational tone
- Encourage the user to consult qualified healthcare professionals for clinical decisions

You must NEVER:
- Diagnose a medical condition
- Prescribe or recommend specific medications or dosages
- Use alarming or panic-inducing language
- Claim certainty about future health outcomes

Always end health-related responses with this exact disclaimer:
"{disclaimer}"
""".format(disclaimer=_DISCLAIMER)


def build_system_prompt(patient_data: Dict[str, Any]) -> str:
    """
    Construct the full LLM system prompt incorporating structured patient data.

    Parameters
    ----------
    patient_data : dict matching the standard Phase 4 input schema

    Returns
    -------
    str: Fully formatted system prompt
    """
    risks = patient_data.get("predicted_risks", {})
    risk_level = patient_data.get("risk_level", "Unknown")
    top_risk_factors = patient_data.get("top_risk_factors", [])
    protective_factors = patient_data.get("protective_factors", [])
    profile = patient_data.get("patient_profile", {})

    heart_risk_pct = f"{risks.get('heart_disease', 0.0) * 100:.1f}%"
    diabetes_risk_pct = f"{risks.get('diabetes', 0.0) * 100:.1f}%"

    risk_factors_str = (
        "\n".join(f"  • {f}" for f in top_risk_factors)
        if top_risk_factors else "  • None identified"
    )
    protective_str = (
        "\n".join(f"  • {f}" for f in protective_factors)
        if protective_factors else "  • None identified"
    )

    patient_context = f"""
PATIENT HEALTH PROFILE (Computed Risk Summary):
────────────────────────────────────────────────
  Age              : {profile.get("age", "N/A")}
  BMI              : {profile.get("bmi", "N/A")}
  Blood Pressure   : {profile.get("blood_pressure", "N/A")} mmHg
  Cholesterol      : {profile.get("cholesterol", "N/A")} mg/dL
  Fasting Glucose  : {profile.get("glucose", "N/A")} mg/dL

RISK ASSESSMENT:
  Overall Risk Level   : {risk_level}
  Heart Disease Risk   : {heart_risk_pct}
  Diabetes Risk        : {diabetes_risk_pct}

TOP RISK FACTORS:
{risk_factors_str}

PROTECTIVE FACTORS:
{protective_str}
────────────────────────────────────────────────

Use this profile to provide contextually relevant health guidance. 
Do not repeat raw numbers mechanically — synthesize them into clear, 
actionable insights.
"""

    return _ASSISTANT_PERSONA + "\n" + patient_context


def build_user_message(user_query: str) -> Dict[str, str]:
    """Wrap a user query into the standard message format."""
    return {"role": "user", "content": user_query.strip()}


def build_conversation_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Validate and return conversation history in the expected format."""
    valid_roles = {"user", "assistant"}
    return [m for m in history if m.get("role") in valid_roles and m.get("content")]
