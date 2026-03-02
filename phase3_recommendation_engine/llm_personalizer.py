"""
llm_personalizer.py
────────────────────
LLM-based natural language explanation layer for the RW-AWRM system.

Design:
    - If an API key is provided, calls an LLM (e.g., OpenAI GPT) to produce
      motivational, preventive, plain-language reformulations of each recommendation.
    - If no API key is available (default), applies a local rule-based
      template engine to achieve comparable narrative quality without any
      external dependency.
    - The LLM is strictly INSTRUCTED to:
        • Use an educational, non-diagnostic tone
        • Not prescribe medications
        • Not replace professional medical advice
        • Motivate lifestyle change with evidence-based framing
"""

import os
import textwrap
from typing import List, Dict, Any, Optional

from phase3_recommendation_engine.clinical_thresholds import classify_risk_band


# ──────────────────────────────────────────────────────────────────────────────
# LLM System Prompt (used when API key is available)
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
You are a preventive health educator writing personalized wellness guidance
for a patient-facing healthcare AI system.

Your instructions:
1. Rewrite the given clinical recommendation in warm, motivating, and accessible language.
2. Retain all specific clinical details (exact targets, food types, exercise durations).
3. Frame the message around empowerment, prevention, and achievable steps.
4. Do NOT prescribe medication or suggest drug changes.
5. Do NOT make diagnostic statements.
6. Always end with a brief sentence encouraging the patient to consult their healthcare provider.
7. Keep the response concise (≤ 5 sentences).
""").strip()


# ──────────────────────────────────────────────────────────────────────────────
# Template-based local personalizer (no API key required)
# ──────────────────────────────────────────────────────────────────────────────

_AGE_GROUP_INTROS = {
    "young_adult": "As someone in their younger years, now is the ideal time to build lasting healthy habits.",
    "middle_aged": "At this stage of life, consistent lifestyle adjustments can meaningfully reduce your long-term health risks.",
    "older_adult": "Maintaining your health at this stage requires targeted, consistent effort—and the benefits are significant.",
    "elderly":     "Prioritizing these lifestyle measures now can substantially improve your quality of life and independence.",
}

_RISK_INTROS = {
    "low":    "Your current risk markers are currently mild, but addressing them early is highly effective.",
    "medium": "Your health profile suggests an intermediate level of risk—focused lifestyle changes at this point can be transformative.",
    "high":   "Your risk indicators are elevated, making these lifestyle changes both urgent and highly impactful.",
}

def _local_personalize(
    rec: Dict[str, Any],
    patient_profile: Dict[str, float],
    predicted_risks: Dict[str, float],
    age_group: str,
) -> Dict[str, Any]:
    """Apply template-based narrative enhancement (no API required)."""
    max_risk_val = max(predicted_risks.values()) if predicted_risks else 0.5
    risk_band = classify_risk_band(max_risk_val)

    age_intro  = _AGE_GROUP_INTROS.get(age_group, "")
    risk_intro = _RISK_INTROS.get(risk_band, "")

    # Compose enhanced recommendation
    enhanced_rec = (
        f"{age_intro} {risk_intro} "
        f"{rec['recommendation']} "
        f"Please discuss this guidance with your healthcare provider before making changes."
    )

    # Compose enhanced justification
    enhanced_justification = (
        f"[Risk level: {risk_band.upper()} | Condition: {rec['related_risk']}] "
        f"{rec['justification']}"
    )

    rec = dict(rec)
    rec["llm_recommendation"]  = enhanced_rec
    rec["llm_justification"]   = enhanced_justification
    return rec


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-based personalizer (optional)
# ──────────────────────────────────────────────────────────────────────────────

def _openai_personalize(
    rec: Dict[str, Any],
    patient_profile: Dict[str, float],
    predicted_risks: Dict[str, float],
    age_group: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    Call OpenAI ChatCompletion to produce an enhanced recommendation.
    Falls back to local template if the call fails.
    """
    try:
        import openai
        openai.api_key = api_key

        user_message = textwrap.dedent(f"""
        Patient Age Group: {age_group}
        Related Disease Risk: {rec['related_risk']}
        Predicted Risk Probabilities: {predicted_risks}
        Clinical Recommendation: {rec['recommendation']}
        Clinical Justification: {rec['justification']}

        Please rewrite the recommendation in a warm, motivating, educational tone.
        """).strip()

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        llm_text = response["choices"][0]["message"]["content"].strip()
        rec = dict(rec)
        rec["llm_recommendation"] = llm_text
        rec["llm_justification"]  = (
            f"[LLM-enhanced | Risk level: {classify_risk_band(max(predicted_risks.values()))} "
            f"| Condition: {rec['related_risk']}] {rec['justification']}"
        )
        return rec

    except Exception as e:
        print(f"[llm_personalizer] OpenAI call failed ({e}). Falling back to local template.")
        return _local_personalize(rec, patient_profile, predicted_risks, age_group)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def personalize_with_llm(
    recommendations: List[Dict[str, Any]],
    patient_profile: Dict[str, float],
    predicted_risks: Dict[str, float],
    age_group: str,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Enhance each recommendation with natural language personalization.

    Uses OpenAI GPT if api_key is provided; otherwise applies a high-quality
    deterministic template-based approach.
    """
    enhanced = []
    for rec in recommendations:
        if api_key:
            enhanced.append(_openai_personalize(
                rec, patient_profile, predicted_risks, age_group, api_key
            ))
        else:
            enhanced.append(_local_personalize(
                rec, patient_profile, predicted_risks, age_group
            ))
    return enhanced
