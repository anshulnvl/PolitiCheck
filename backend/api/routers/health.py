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

from pipeline.cache import REDIS_URL
from pipeline.db import engine
from signals.ml_signal import CHECKPOINT

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", summary="Service health check")
def health_check() -> JSONResponse:
    services: dict[str, str] = {}
    overall = "ok"

    # ── Redis ─────────────────────────────────────────────────────────────────
    try:
        r = redis.Redis.from_url(f"{REDIS_URL}/0", socket_connect_timeout=2, socket_timeout=2)
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
    checkpoint_path = os.path.abspath(CHECKPOINT)
    if os.path.isdir(checkpoint_path):
        services["ml_model"] = "ok"
    else:
        services["ml_model"] = f"checkpoint not found at {checkpoint_path}"
        overall = "degraded"

    status_code = 200 if overall == "ok" else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": overall, "services": services},
    )
