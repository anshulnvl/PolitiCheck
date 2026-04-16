"""
api/routers/health.py
=====================
GET /api/health — liveness + readiness probe.

Checks:
  • Redis (pipeline cache)
  • PostgreSQL (analyses DB)
  • ML model checkpoint (directory presence)

Returns HTTP 200 with status "ok" when all services are reachable,
or HTTP 503 with status "degraded" when one or more services are down.
Individual service errors are surfaced in the response body so callers
can distinguish which component is unhealthy.
"""

from __future__ import annotations

import os

import redis
import sqlalchemy
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.pipeline.cache import REDIS_URL
from backend.pipeline.db import engine
from backend.signals.ml_signal import CHECKPOINT

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", summary="Service health check")
def health_check() -> JSONResponse:
    """
    Check health of all backend services.
    
    Returns 200 (ok) if all services are reachable, 503 (degraded) otherwise.
    
    Checks:
      - Redis cache (for pipeline)
      - PostgreSQL database
      - ML model checkpoint (optional)
    
    Returns:
      JSONResponse with status and per-service details
    """
    services: dict[str, str] = {}
    overall = "ok"

    # ── Redis ─────────────────────────────────────────────────────────────────
    try:
        r = redis.Redis.from_url(f"{REDIS_URL}/0", socket_connect_timeout=1, socket_timeout=1)
        r.ping()
        services["redis"] = "ok"
    except Exception as exc:
        services["redis"] = f"error: {exc}"
        overall = "degraded"

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        services["postgres"] = "ok"
    except Exception as exc:
        services["postgres"] = f"error: {exc}"
        overall = "degraded"

    # ── ML model checkpoint ───────────────────────────────────────────────────
    # Checkpoint is optional: if missing, health check warns but doesn't fail
    checkpoint_optional = os.getenv("ML_CHECKPOINT_REQUIRED", "false").lower() == "true"
    checkpoint_path = os.path.abspath(CHECKPOINT) if CHECKPOINT else None
    
    if checkpoint_path and os.path.isdir(checkpoint_path):
        services["ml_model"] = "ok"
    elif checkpoint_optional:
        services["ml_model"] = f"checkpoint not found at {checkpoint_path}"
        overall = "degraded"
    else:
        services["ml_model"] = f"not loaded (optional) — expected at {checkpoint_path}"
        # Don't degrade overall status for optional checkpoint


    status_code = 200 if overall == "ok" else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": overall, "services": services},
    )
