"""
routers/recommend.py
─────────────────────
POST /recommend — Lifestyle recommendation endpoint.
Calls the Phase 3 hybrid rule + LLM recommendation engine.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import RecommendRequest, RecommendResponse
from phase3_recommendation_engine.hybrid_recommender import generate_recommendations

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


@router.post("", response_model=RecommendResponse)
def get_recommendations(body: RecommendRequest) -> RecommendResponse:
    """
    Generate personalized lifestyle recommendations for a patient.

    Combines clinical rule engine + risk-weighted scoring.
    Set use_llm=true to enhance recommendations with an LLM language layer.
    """
    result = generate_recommendations(
        predicted_risks=body.predicted_risks,
        top_positive_risk_factors=body.top_positive_risk_factors,
        protective_factors=body.protective_factors,
        patient_profile=body.biomarkers.model_dump(),
        use_llm=body.use_llm,
    )

    return RecommendResponse(**result)
