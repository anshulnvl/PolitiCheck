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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from signals.ml_signal import router as ml_router
from api.routers import analyze, feedback, health

app = FastAPI(
    title="PolitiCheck API",
    version="1.0.0",
    description="Credibility analysis API powered by RoBERTa, XGBoost, and external fact-checking.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://politicheck.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router)          # POST /signals/ml
app.include_router(analyze.router)     # POST /api/analyze
app.include_router(feedback.router)    # POST /api/feedback
app.include_router(health.router)      # GET  /api/health
