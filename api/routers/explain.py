"""
routers/explain.py
──────────────────
POST /explain — SHAP-based local explainability endpoint.
Returns per-feature SHAP values and top risk/protective factors
for a single patient instance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import PatientBiomarkers, ExplainResponse
from api.dependencies import get_model, get_scaler, biomarkers_to_df

router = APIRouter(prefix="/explain", tags=["Explainability"])

_FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


@router.post("", response_model=ExplainResponse)
def explain_prediction(
    biomarkers: PatientBiomarkers,
    model=Depends(get_model),
    scaler=Depends(get_scaler),
) -> ExplainResponse:
    """
    Compute SHAP feature attributions for a single patient prediction.

    Returns feature importances, top risk factors, and protective factors.
    """
    try:
        import shap
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="SHAP is not installed. Run `pip install shap` to enable /explain.",
        )

    df = biomarkers_to_df(biomarkers.model_dump())
    X_scaled = pd.DataFrame(scaler.transform(df), columns=_FEATURE_ORDER)

    # Select appropriate SHAP explainer
    model_type = type(model).__name__
    if "RandomForest" in model_type or "XGB" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        sv = shap_vals[0]
    elif "Logistic" in model_type or "Linear" in model_type:
        explainer = shap.LinearExplainer(model, X_scaled)
        shap_vals = explainer.shap_values(X_scaled)
        sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
    else:
        background = shap.sample(X_scaled, min(20, len(X_scaled)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_vals = explainer.shap_values(X_scaled)
        sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    feature_importances = {
        f: float(round(v, 6)) for f, v in zip(_FEATURE_ORDER, sv)
    }

    # Separate positive (risk-increasing) vs. negative (protective) factors
    contributions = sorted(zip(_FEATURE_ORDER, df.iloc[0].values, sv), key=lambda x: x[2])
    protective     = [f"{f} ({v:.2f})" for f, v, s in contributions if s < 0][:5]
    risk_factors   = [f"{f} ({v:.2f})" for f, v, s in reversed(contributions) if s > 0][:5]

    return ExplainResponse(
        feature_importances=feature_importances,
        top_positive_risk_factors=risk_factors,
        protective_factors=protective,
    )
