"""
schemas.py
──────────
Pydantic request/response models for all HealthGuard-XAI API endpoints.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Shared Input ───────────────────────────────────────────────────────────────

class PatientBiomarkers(BaseModel):
    """Raw patient biomarker features (PIMA Diabetes dataset schema)."""
    Pregnancies:              float = Field(..., example=2)
    Glucose:                  float = Field(..., example=138)
    BloodPressure:            float = Field(..., example=72)
    SkinThickness:            float = Field(..., example=35)
    Insulin:                  float = Field(..., example=0)
    BMI:                      float = Field(..., example=33.6)
    DiabetesPedigreeFunction: float = Field(..., example=0.627)
    Age:                      float = Field(..., example=50)


# ── /predict ──────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    risk_probability: float
    risk_level:       str       # Low | Medium | High
    top_features:     Dict[str, float]


# ── /explain ──────────────────────────────────────────────────────────────────

class ExplainResponse(BaseModel):
    feature_importances:        Dict[str, float]
    top_positive_risk_factors:  List[str]
    protective_factors:         List[str]


# ── /recommend ────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    biomarkers:                 PatientBiomarkers
    predicted_risks:            Dict[str, float]
    top_positive_risk_factors:  List[str] = []
    protective_factors:         List[str] = []
    use_llm:                    bool = False


class RecommendResponse(BaseModel):
    patient_context:              Dict[str, Any]
    prioritized_recommendations:  List[Dict[str, Any]]
    general_wellness_advice:      List[str]
    disclaimer:                   str


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:      str
    patient_data: Dict[str, Any]
    history:      Optional[List[Dict[str, str]]] = []


class ChatResponse(BaseModel):
    assistant_response:  str
    safety_flag:         bool
    escalation_required: bool
