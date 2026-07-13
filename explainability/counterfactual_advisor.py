"""
counterfactual_advisor.py
──────────────────────────
Generates counterfactual explanations for HealthGuard-XAI predictions.

"What is the MINIMUM change to a patient's biomarkers that would flip
 their prediction from High Risk → Low Risk?"

Uses a gradient-free iterative search (constrained to clinically 
plausible ranges) to find the closest counterfactual point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

# ── Clinical bounds for each feature ─────────────────────────────────────────
# Defines the minimum plausible and maximum plausible value for each biomarker
# to prevent counterfactuals from suggesting physiologically impossible values.

CLINICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Pregnancies":              (0.0,   15.0),
    "Glucose":                  (70.0,  200.0),
    "BloodPressure":            (50.0,  130.0),
    "SkinThickness":            (5.0,   80.0),
    "Insulin":                  (0.0,   400.0),
    "BMI":                      (15.0,  55.0),
    "DiabetesPedigreeFunction": (0.08,  2.5),
    "Age":                      (18.0,  90.0),
}

# Features that should only decrease (lowering risk) — not increase
MODIFIABLE_FEATURES = [
    "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction",
]

# Step size for each feature in the iterative search
STEP_SIZES: Dict[str, float] = {
    "Pregnancies":              1.0,
    "Glucose":                  2.0,
    "BloodPressure":            2.0,
    "SkinThickness":            1.0,
    "Insulin":                  5.0,
    "BMI":                      0.5,
    "DiabetesPedigreeFunction": 0.05,
    "Age":                      1.0,
}

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def _clip_to_bounds(value: float, feature: str) -> float:
    lo, hi = CLINICAL_BOUNDS[feature]
    return float(np.clip(value, lo, hi))


def _predict_prob(model: Any, scaler: Any, biomarkers: Dict[str, float]) -> float:
    """Get probability from a single biomarker dict."""
    df = pd.DataFrame([biomarkers])[FEATURE_ORDER]
    X_scaled = scaler.transform(df)
    return float(model.predict_proba(X_scaled)[0][1])


def generate_counterfactual(
    model: Any,
    scaler: Any,
    original_biomarkers: Dict[str, float],
    target_risk: float = 0.33,
    max_iterations: int = 200,
) -> Dict[str, Any]:
    """
    Find the minimal biomarker modifications that bring risk below `target_risk`.

    Parameters
    ----------
    model               : Trained sklearn-compatible classifier with predict_proba
    scaler              : Fitted StandardScaler
    original_biomarkers : Dict of feature → value for current patient
    target_risk         : Target probability threshold (default: 0.33 = Low risk)
    max_iterations      : Maximum optimization iterations

    Returns
    -------
    dict with keys:
        - original_risk      : float
        - counterfactual_risk: float
        - changes            : List[dict] — what changed and by how much
        - counterfactual_biomarkers: dict
        - achieved           : bool — whether target was reached
        - risk_reduction     : float
    """
    original_risk = _predict_prob(model, scaler, original_biomarkers)

    # If already below target, no changes needed
    if original_risk <= target_risk:
        return {
            "original_risk":               original_risk,
            "counterfactual_risk":         original_risk,
            "changes":                     [],
            "counterfactual_biomarkers":   original_biomarkers.copy(),
            "achieved":                    True,
            "risk_reduction":              0.0,
            "message":                     "Risk is already within the Low range — no changes needed!",
        }

    # Work on a mutable copy
    cf = original_biomarkers.copy()

    for iteration in range(max_iterations):
        current_risk = _predict_prob(model, scaler, cf)
        if current_risk <= target_risk:
            break

        # Try modifying each modifiable feature and pick the one that reduces risk most
        best_feature   = None
        best_new_val   = None
        best_risk_gain = 0.0

        for feature in MODIFIABLE_FEATURES:
            current_val = cf[feature]
            lo, hi      = CLINICAL_BOUNDS[feature]
            step        = STEP_SIZES[feature]

            # Try decreasing the feature value
            new_val = _clip_to_bounds(current_val - step, feature)
            if new_val == current_val:
                continue  # Already at bound

            trial    = cf.copy()
            trial[feature] = new_val
            trial_risk = _predict_prob(model, scaler, trial)
            gain = current_risk - trial_risk

            if gain > best_risk_gain:
                best_risk_gain = gain
                best_feature   = feature
                best_new_val   = new_val

        if best_feature is None:
            break  # No more improvements possible

        cf[best_feature] = best_new_val

    final_risk = _predict_prob(model, scaler, cf)

    # Build change summary
    changes: List[Dict[str, Any]] = []
    for feature in FEATURE_ORDER:
        orig_val = original_biomarkers[feature]
        new_val  = cf[feature]
        delta    = new_val - orig_val
        if abs(delta) > 1e-6:
            changes.append({
                "feature":        feature,
                "original_value": round(orig_val, 2),
                "new_value":      round(new_val, 2),
                "change":         round(delta, 2),
                "direction":      "↓" if delta < 0 else "↑",
            })

    return {
        "original_risk":             round(original_risk, 4),
        "counterfactual_risk":       round(final_risk, 4),
        "changes":                   changes,
        "counterfactual_biomarkers": cf,
        "achieved":                  final_risk <= target_risk,
        "risk_reduction":            round(original_risk - final_risk, 4),
        "message": (
            f"By making {len(changes)} lifestyle adjustment(s), your estimated risk "
            f"could drop from {original_risk:.1%} → {final_risk:.1%}."
        ) if changes else "No feasible modifications found within clinical bounds.",
    }
