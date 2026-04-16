"""
celery_app.py
=============
Celery application instance shared across the pipeline.

Broker  : Redis db 0  (task queue)
Backend : Redis db 1  (result store)

Start a worker with:
    celery -A backend.celery_app worker --loglevel=info --concurrency=4

Or for local dev (no separate worker, inline execution):
    CELERY_TASK_ALWAYS_EAGER=true python -m backend.pipeline.orchestrator
"""

import os
import platform

# Must be set before importing numpy, torch, xgboost, or any native library.
# On macOS, multiple OpenMP runtimes (PyTorch + XGBoost) initialised in the
# same process will segfault.  Pinning every threading layer to 1 thread
# prevents the collision; the solo-pool worker is already single-process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
IS_MACOS = platform.system() == "Darwin"

# macOS + prefork can crash with native frameworks (e.g. Metal/MPS, Objective-C)
# when workers fork after runtime initialization. Use a fork-safe default pool.
DEFAULT_WORKER_POOL = "solo" if IS_MACOS else "prefork"
DEFAULT_WORKER_CONCURRENCY = 1 if DEFAULT_WORKER_POOL == "solo" else 4

celery_app = Celery(
    "politicheck",
    broker=f"{REDIS_URL}/0",
    backend=f"{REDIS_URL}/1",
    include=["backend.pipeline.tasks"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Visibility
    task_track_started=True,
    worker_send_task_events=True,

    # Result TTL — keep results for 1 h (orchestrator collects them fast)
    result_expires=3600,

    # Retry defaults for signal tasks
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Worker runtime defaults
    worker_pool=os.getenv("CELERY_WORKER_POOL", DEFAULT_WORKER_POOL),
    worker_concurrency=int(os.getenv("CELERY_WORKER_CONCURRENCY", str(DEFAULT_WORKER_CONCURRENCY))),
    worker_prefetch_multiplier=1,

    # Timezone
    timezone="UTC",
    enable_utc=True,
)
