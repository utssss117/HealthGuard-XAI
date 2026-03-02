"""
risk_weighting.py
─────────────────
Risk-Weighted Scoring Algorithm for the RW-AWRM system.

For each rule-engine recommendation, computes:

    raw_score = disease_risk_probability
                × |mean_shap_contribution_for_feature|
                × clinical_importance_weight

Scores are normalized to [0, 1] via min-max normalization across all
recommendations for a given patient, then ranked by priority.
"""

from typing import List, Dict, Any
import numpy as np

from phase3_recommendation_engine.clinical_thresholds import get_clinical_importance, FEATURE_ALIAS_MAP


def _resolve_shap_weight(feature: str, shap_contributions: Dict[str, float]) -> float:
    """
    Return the absolute SHAP contribution for a feature.
    Tries canonical name and aliases; falls back to 0.5 if not present.
    """
    canonical = FEATURE_ALIAS_MAP.get(feature.lower(), feature)
    # Try direct match first
    for key, val in shap_contributions.items():
        if key == canonical or key.lower() == feature.lower():
            return abs(val)
    return 0.5  # conservative fallback weight


def _extract_max_risk(related_risk_str: str, predicted_risks: Dict[str, float]) -> float:
    """
    For a recommendation related to one or more diseases, return the maximum
    predicted probability across those diseases. This ensures multi-risk
    recommendations inherit the highest applicable risk.
    """
    diseases = [d.strip() for d in related_risk_str.split(",")]
    max_risk = 0.0
    for disease in diseases:
        prob = predicted_risks.get(disease, 0.0)
        if prob > max_risk:
            max_risk = prob
    return max_risk if max_risk > 0.0 else 0.3  # neutral fallback


def compute_weighted_scores(
    raw_recommendations: List[Dict[str, Any]],
    predicted_risks: Dict[str, float],
    shap_contributions: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Attach a composite raw score to each recommendation.

    Parameters
    ----------
    raw_recommendations : output from rule_engine.apply_clinical_rules()
    predicted_risks     : {'diabetes': 0.72, 'heart_disease': 0.55}
    shap_contributions  : {'Glucose': 0.31, 'BMI': 0.22, ...}  (mean |SHAP|)

    Returns
    -------
    List of recommendation dicts with added 'raw_score' key.
    """
    scored = []
    for rec in raw_recommendations:
        feature         = rec["feature"]
        disease_risk    = _extract_max_risk(rec["related_risk"], predicted_risks)
        shap_weight     = _resolve_shap_weight(feature, shap_contributions)
        clinical_weight = get_clinical_importance(feature)

        raw_score = disease_risk * shap_weight * clinical_weight
        rec = dict(rec)  # shallow copy
        rec["raw_score"] = raw_score
        scored.append(rec)

    return scored


def normalize_and_rank(scored_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Min-max normalize raw_score → priority_score in [0, 1].
    Sort descending by priority_score.
    """
    if not scored_recommendations:
        return []

    scores = np.array([r["raw_score"] for r in scored_recommendations], dtype=float)
    s_min, s_max = scores.min(), scores.max()

    if s_max == s_min:
        normalized = np.ones_like(scores)
    else:
        normalized = (scores - s_min) / (s_max - s_min)

    for rec, norm_score in zip(scored_recommendations, normalized):
        rec["priority_score"] = round(float(norm_score), 4)

    ranked = sorted(scored_recommendations, key=lambda r: r["priority_score"], reverse=True)
    return ranked


def apply_age_personalization(
    ranked: List[Dict[str, Any]],
    age_group: str,
) -> List[Dict[str, Any]]:
    """
    Apply a multiplicative boost/penalty to priority_score based on age group.
    Older patients get elevated scores for cardiovascular and metabolic risks.
    """
    age_modifiers = {
        "young_adult": 0.85,
        "middle_aged": 1.00,
        "older_adult": 1.15,
        "elderly":     1.25,
    }
    modifier = age_modifiers.get(age_group, 1.0)

    for rec in ranked:
        rec["priority_score"] = min(round(rec["priority_score"] * modifier, 4), 1.0)

    # Re-sort after modifier
    ranked = sorted(ranked, key=lambda r: r["priority_score"], reverse=True)
    return ranked


def apply_severity_boost(
    ranked: List[Dict[str, Any]],
    predicted_risks: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Boost recommendations proportional to highest disease risk probability.
    High-severity patients receive amplified prioritization signals.
    """
    max_risk = max(predicted_risks.values()) if predicted_risks else 0.5

    # Boost = linear scale: 10% boost at max_risk=1.0, no boost at 0.33
    boost_factor = 1.0 + (max(max_risk - 0.33, 0.0) / 0.67) * 0.10

    for rec in ranked:
        rec["priority_score"] = min(round(rec["priority_score"] * boost_factor, 4), 1.0)

    ranked = sorted(ranked, key=lambda r: r["priority_score"], reverse=True)
    return ranked
