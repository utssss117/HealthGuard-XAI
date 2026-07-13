"""
schemas.py
──────────
Pydantic request/response models with strict validation ranges and Swagger descriptions.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Shared Input ───────────────────────────────────────────────────────────────

class PatientBiomarkers(BaseModel):
    """Raw patient biomarker features (PIMA Diabetes dataset schema) with strict range validation."""
    
    Pregnancies: float = Field(
        ...,
        ge=0,
        le=20,
        description="Number of times pregnant",
        json_schema_extra={"example": 2}
    )
    Glucose: float = Field(
        ...,
        ge=0,
        le=300,
        description="2-hour oral glucose tolerance test value (mg/dL)",
        json_schema_extra={"example": 138}
    )
    BloodPressure: float = Field(
        ...,
        ge=0,
        le=200,
        description="Diastolic blood pressure (mm Hg)",
        json_schema_extra={"example": 72}
    )
    SkinThickness: float = Field(
        ...,
        ge=0,
        le=100,
        description="Triceps skinfold thickness (mm)",
        json_schema_extra={"example": 35}
    )
    Insulin: float = Field(
        ...,
        ge=0,
        le=900,
        description="2-hour serum insulin (mu U/ml)",
        json_schema_extra={"example": 80}
    )
    BMI: float = Field(
        ...,
        ge=0.0,
        le=80.0,
        description="Body mass index (weight in kg/(height in m)^2)",
        json_schema_extra={"example": 33.6}
    )
    DiabetesPedigreeFunction: float = Field(
        ...,
        ge=0.0,
        le=3.0,
        description="Diabetes pedigree function genetic score",
        json_schema_extra={"example": 0.627}
    )
    Age: float = Field(
        ...,
        ge=0,
        le=120,
        description="Age in years",
        json_schema_extra={"example": 50}
    )


# ── /predict ──────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """Response containing risk probability and top diagnostic feature weights."""
    risk_probability: float = Field(..., description="Normalized risk probability score between 0 and 1")
    risk_level: str = Field(..., description="Risk category band: Low, Medium, or High")
    top_features: Dict[str, float] = Field(..., description="Sorted dictionary of key biomarker contributions")


# ── /explain ──────────────────────────────────────────────────────────────────

class ExplainResponse(BaseModel):
    """Response containing SHAP attribution values and categorical risk/protective lists."""
    feature_importances: Dict[str, float] = Field(..., description="Mapping of feature names to calculated SHAP values")
    top_positive_risk_factors: List[str] = Field(..., description="Attributes increasing total health risk")
    protective_factors: List[str] = Field(..., description="Attributes reducing total health risk")


# ── /recommend ────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """Personalized clinical recommendation query parameters."""
    biomarkers: PatientBiomarkers = Field(..., description="Biomarker parameters of the patient")
    predicted_risks: Dict[str, float] = Field(..., description="Calculated risks dictionary, e.g. {'diabetes': 0.72}")
    top_positive_risk_factors: List[str] = Field([], description="SHAP output list of top risk-increasing elements")
    protective_factors: List[str] = Field([], description="SHAP output list of top risk-decreasing elements")
    use_llm: bool = Field(False, description="Toggle Groq LLM clinical language generator wrapper")


class RecommendResponse(BaseModel):
    """Clinical lifestyle recommendations payload."""
    patient_context: Dict[str, Any] = Field(..., description="Context summary generated for this patient's profile")
    prioritized_recommendations: List[Dict[str, Any]] = Field(..., description="List of rule-based clinical guides ordered by priority")
    general_wellness_advice: List[str] = Field(..., description="Default global healthy wellness recommendations")
    disclaimer: str = Field(..., description="Mandatory medical legal disclaimer")


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """AI Health Assistant dialogue query."""
    message: str = Field(..., description="User query message text to HealthGuard assistant")
    patient_data: Dict[str, Any] = Field(..., description="Context patient metrics and risk profile information")
    history: Optional[List[Dict[str, str]]] = Field([], description="Preceding conversation history array")


class ChatResponse(BaseModel):
    """AI Health Assistant dialogue reply."""
    assistant_response: str = Field(..., description="Groq LLaMA clinical response text")
    safety_flag: bool = Field(..., description="Indicates if safety filters flagged this query as containing emergency indicators")
    escalation_required: bool = Field(..., description="Indicates if emergency escalation is required")
