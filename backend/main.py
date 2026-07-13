"""
main.py
────────
HealthGuard-XAI — Unified FastAPI Application with Production readiness features.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.dependencies import load_model_and_scaler, verify_api_key
from backend.routers import predict, explain, recommend, chat, auth_router
from backend.database import init_db
from backend.logger import logger

START_TIME = time.time()

# ── Startup lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HealthGuard-XAI API starting up — loading model and scaler...")
    load_model_and_scaler()   # cache the model at startup
    logger.info("Initializing database...")
    init_db()
    logger.info("Model loaded successfully. API is ready.")
    yield
    logger.info("HealthGuard-XAI API shutting down.")


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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handlers ─────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "detail": exc.errors()},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP exception on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# ── Static Dashboard ─────────────────────────────────────────────────────────

DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")

if os.path.exists(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


# ── Routers ───────────────────────────────────────────────────────────────────

# Authentication routes do not require API key headers (called by external providers/webhooks)
app.include_router(auth_router.router, prefix="/api")

# Core predictive diagnostic routes protected via X-API-Key header validation
app.include_router(predict.router, prefix="/api", dependencies=[Depends(verify_api_key)])
app.include_router(explain.router, prefix="/api", dependencies=[Depends(verify_api_key)])
app.include_router(recommend.router, prefix="/api", dependencies=[Depends(verify_api_key)])
app.include_router(chat.router, prefix="/api", dependencies=[Depends(verify_api_key)])


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service":  "HealthGuard-XAI API",
        "version":  "1.0.0",
        "status":   "running",
        "endpoints": ["/predict", "/explain", "/recommend", "/chat", "/docs", "/health"],
    }


@app.get("/health", tags=["Health"])
def health():
    """Check API status, model state, and retrieve service uptime."""
    try:
        model, scaler = load_model_and_scaler()
        model_loaded = model is not None
        scaler_loaded = scaler is not None
        status = "healthy"
    except Exception as e:
        logger.error(f"Health check model/scaler load failed: {e}")
        model_loaded = False
        scaler_loaded = False
        status = "unhealthy"

    return {
        "status": status,
        "model_loaded": model_loaded,
        "scaler_loaded": scaler_loaded,
        "uptime_seconds": time.time() - START_TIME
    }
