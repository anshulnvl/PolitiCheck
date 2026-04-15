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
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

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

    # Timezone
    timezone="UTC",
    enable_utc=True,
)
