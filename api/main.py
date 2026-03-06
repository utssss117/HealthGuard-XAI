"""
main.py
────────
HealthGuard-XAI — Unified FastAPI Application

Exposes:
  POST /predict    — ML probabilistic risk prediction
  POST /explain    — SHAP-based feature explainability
  POST /recommend  — Hybrid lifestyle recommendation engine
  POST /chat       — Groq LLM health guidance assistant

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from api.dependencies import load_model_and_scaler
from api.routers import predict, explain, recommend, chat


# ── Startup lifespan (pre-load model + scaler) ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("HealthGuard-XAI API starting up — loading model and scaler...")
    load_model_and_scaler()   # cache the model at startup
    print("Model loaded successfully. API is ready.")
    yield
    print("HealthGuard-XAI API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HealthGuard-XAI API",
    description=(
        "Research-grade healthcare AI system for multi-disease probabilistic "
        "risk prediction, SHAP explainability, personalized recommendations, "
        "and LLM-powered health guidance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Dashboard ─────────────────────────────────────────────────────────

DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")

if os.path.exists(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(recommend.router)
app.include_router(chat.router)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service":  "HealthGuard-XAI API",
        "version":  "1.0.0",
        "status":   "running",
        "endpoints": ["/predict", "/explain", "/recommend", "/chat", "/docs"],
    }
