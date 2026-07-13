"""
routers/explain.py
──────────────────
POST /explain — SHAP-based local explainability endpoint.
Returns per-feature SHAP values and top risk/protective factors
for a single patient instance.
"""

from __future__ import annotations

import json
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from backend.schemas import PatientBiomarkers, ExplainResponse
from backend.dependencies import get_model, get_scaler, biomarkers_to_df
from backend.auth import get_current_user

router = APIRouter(prefix="/explain", tags=["Explainability"])

_FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]




_explain_cache = {}
MAX_CACHE_SIZE = 500

import os

def get_background_data(scaler, n_samples=50):
    try:
        # explain.py is in backend/routers/ -> go up three levels to reach root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(base_dir, "data", "diabetes.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "Outcome" in df.columns:
                df = df.drop(columns=["Outcome"])
            df = df[_FEATURE_ORDER]
            df_sampled = df.sample(n=min(n_samples, len(df)), random_state=42)
            return pd.DataFrame(scaler.transform(df_sampled), columns=_FEATURE_ORDER)
    except Exception:
        pass
    return None

@router.post("", response_model=ExplainResponse)
def explain_prediction(
    biomarkers: PatientBiomarkers,
    model=Depends(get_model),
    scaler=Depends(get_scaler),
    current_user=Depends(get_current_user),
) -> ExplainResponse:
    """
    Compute SHAP feature attributions for a single patient prediction.

    Returns feature importances, top risk factors, and protective factors.
    """
    # Simple in-memory cache lookup
    cache_key = json.dumps(biomarkers.model_dump(), sort_keys=True)
    if cache_key in _explain_cache:
        return _explain_cache[cache_key]

    try:
        import shap
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="SHAP is not installed. Run `pip install shap` to enable /explain.",
        )

    df = biomarkers_to_df(biomarkers.model_dump())
    X_scaled = pd.DataFrame(scaler.transform(df), columns=_FEATURE_ORDER)
    bg = get_background_data(scaler)

    # Select appropriate SHAP explainer
    model_type = type(model).__name__
    if "RandomForest" in model_type or "XGB" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        sv = shap_vals[0]
    elif "Logistic" in model_type or "Linear" in model_type:
        explainer = shap.LinearExplainer(model, bg if bg is not None else X_scaled)
        shap_vals = explainer.shap_values(X_scaled)
        sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
    else:
        background = bg if bg is not None else shap.sample(X_scaled, min(20, len(X_scaled)))
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

    response = ExplainResponse(
        feature_importances=feature_importances,
        top_positive_risk_factors=risk_factors,
        protective_factors=protective,
    )

    if len(_explain_cache) >= MAX_CACHE_SIZE:
        # Simple FIFO eviction
        try:
            first_key = next(iter(_explain_cache))
            _explain_cache.pop(first_key)
        except Exception:
            pass

    _explain_cache[cache_key] = response
    return response
