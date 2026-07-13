"""
dependencies.py
───────────────
Shared FastAPI dependency injection: loads model and scaler once at startup
and injects them into route handlers via Depends().
"""

from __future__ import annotations
from fastapi import Header, HTTPException

import os
import pickle
import pandas as pd
from functools import lru_cache
from typing import Any, Tuple

_MODEL_PATH  = os.path.join("outputs", "models", "best_model.pkl")
_SCALER_PATH = os.path.join("outputs", "models", "scaler.pkl")

_FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


@lru_cache(maxsize=1)
def load_model_and_scaler() -> Tuple[Any, Any]:
    """Load and cache the trained scikit-learn model and scaler."""
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{_MODEL_PATH}'. "
            "Run `python train.py` from the project root first."
        )
    with open(_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def get_model():
    model, _ = load_model_and_scaler()
    return model


def get_scaler():
    _, scaler = load_model_and_scaler()
    return scaler


def biomarkers_to_df(biomarkers_dict: dict) -> pd.DataFrame:
    """Convert a biomarker dict to a properly ordered DataFrame."""
    return pd.DataFrame([biomarkers_dict])[_FEATURE_ORDER]


async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """
    Validate the incoming request API Key header.
    Enforced only if HEALTHGUARD_API_KEY is configured in the environment.
    """
    expected_key = os.environ.get("HEALTHGUARD_API_KEY")
    if expected_key:
        if not x_api_key or x_api_key != expected_key:
            raise HTTPException(
                status_code=401,
                detail="Authentication required: Invalid or missing API Key"
            )
