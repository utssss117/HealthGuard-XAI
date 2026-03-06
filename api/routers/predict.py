"""
routers/predict.py
──────────────────
POST /predict — ML probabilistic risk prediction endpoint.
Loads the trained best model and scaler, accepts patient biomarkers,
and returns a risk probability + risk level.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends

from api.schemas import PatientBiomarkers, PredictResponse
from api.dependencies import get_model, get_scaler, biomarkers_to_df

router = APIRouter(prefix="/predict", tags=["Risk Prediction"])


@router.post("", response_model=PredictResponse)
def predict_risk(
    biomarkers: PatientBiomarkers,
    model=Depends(get_model),
    scaler=Depends(get_scaler),
) -> PredictResponse:
    """
    Predict diabetes/cardiovascular risk probability from patient biomarkers.

    Returns a normalized probability score and a categorical risk level.
    """
    df = biomarkers_to_df(biomarkers.model_dump())
    X_scaled = scaler.transform(df)

    prob = float(model.predict_proba(X_scaled)[0][1])

    if prob <= 0.33:
        risk_level = "Low"
    elif prob <= 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Feature importance (coefficients for LR, feature_importances_ for tree-based)
    feature_names = df.columns.tolist()
    if hasattr(model, "coef_"):
        raw = np.abs(model.coef_[0])
        top_features = dict(zip(feature_names, [float(v) for v in raw]))
    elif hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        top_features = dict(zip(feature_names, [float(v) for v in raw]))
    else:
        top_features = {}

    # Sort descending
    top_features = dict(sorted(top_features.items(), key=lambda x: x[1], reverse=True))

    return PredictResponse(
        risk_probability=prob,
        risk_level=risk_level,
        top_features=top_features,
    )
