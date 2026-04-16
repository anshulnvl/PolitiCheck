"""
api/main.py
===========
PolitiCheck FastAPI application.

Start the server (from the backend/ directory):
    uvicorn api.main:app --reload --port 8000

For local development without Celery workers:
    CELERY_TASK_ALWAYS_EAGER=true uvicorn api.main:app --reload --port 8000

Endpoints
---------
  POST /signals/ml          raw ML signal (from signals.ml_signal)
  POST /api/analyze         main credibility-analysis endpoint
  POST /api/feedback        submit user corrections
  GET  /api/health          liveness + readiness probe
  GET  /docs                auto-generated OpenAPI UI (Swagger)
  GET  /redoc               ReDoc UI
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure `backend.*` imports work whether uvicorn is launched from project root
# (`uvicorn backend.api.main:app`) or from backend/ (`uvicorn api.main:app`).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.signals.ml_signal import router as ml_router
from backend.api.routers import analyze, feedback, health

app = FastAPI(
    title="PolitiCheck API",
    version="1.0.0",
    description="Credibility analysis API powered by RoBERTa, XGBoost, and external fact-checking.",
)

# CORS configuration from environment, with sensible defaults for development
default_cors_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "null",  # file:// pages send Origin: null
]
cors_origins_env = os.getenv("CORS_ORIGINS")
cors_origins = (
    [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
    if cors_origins_env
    else default_cors_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router)          # POST /signals/ml
app.include_router(analyze.router)     # POST /api/analyze
app.include_router(feedback.router)    # POST /api/feedback
app.include_router(health.router)      # GET  /api/health
