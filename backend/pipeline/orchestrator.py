"""
pipeline/orchestrator.py
========================
Top-level async entry point that wires Steps 1–9 together.

Step 3  — Redis cache check (short-circuit if already analysed)
Step 4  — Preprocess raw input (URL fetch or text cleaning)
Step 5  — Fan-out: run ML / linguistic / external signals in parallel (Celery group)
Step 6  — Ensemble: XGBoost + SHAP on pre-computed signal scores
Step 8  — Persist: write to Redis cache + PostgreSQL

Usage
-----
    import asyncio
    from backend.pipeline.orchestrator import run_pipeline

    result = asyncio.run(run_pipeline("https://example.com/article"))
    result = asyncio.run(run_pipeline("Some article text …", input_type="text"))

Celery workers must be running for steps 5–6 to execute in separate processes.
For local/testing without workers set CELERY_TASK_ALWAYS_EAGER=true.
"""

import asyncio
import logging
from typing import Literal

from celery import group

from backend.pipeline.cache import check_cache, write_cache
from backend.pipeline.db    import save_to_postgres
from backend.pipeline.tasks import (
    ensemble_task,
    external_signal_task,
    linguistic_signal_task,
    ml_signal_task,
)
from backend.signals.preprocessing import process as _preprocess_sync

log = logging.getLogger(__name__)

InputType = Literal["auto", "url", "text"]

# Fan-out timeout: how long to wait for all three signal workers (seconds)
_SIGNAL_TIMEOUT = int(__import__("os").getenv("SIGNAL_TIMEOUT", 90))


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(text: str, input_type: InputType = "auto") -> dict:
    """
    Analyse *text* (article body or URL) and return a credibility result dict.

    Returns
    -------
    {
        "verdict":           "CREDIBLE" | "FAKE" | "UNCERTAIN",
        "credibility_score": float,          # 0.0 fake → 1.0 credible
        "confidence":        float,          # 0.0 → 1.0
        "signals": {
            "ml":         {"score": float, "features": dict},
            "linguistic": {"score": float, "features": dict, ...},
            "external":   {"score": float, "features": dict},
        },
        "explanation": {
            "top_drivers": [...],            # SHAP top-5
            "base_value":  float,
        },
        "document": {
            "title":       str,
            "source":      str,
            "word_count":  int,
            "language":    str,
        },
        "cached":  bool,
        "input_text": str,                   # original raw input (for DB key)
    }
    """
    # ── Step 3: cache check ───────────────────────────────────────────────────
    cached = check_cache(text)
    if cached:
        log.debug("Cache hit for input (len=%d)", len(text))
        return {**cached, "cached": True}

    # ── Step 4: preprocess ───────────────────────────────────────────────────
    doc = await asyncio.to_thread(_preprocess_sync, text)
    if doc.error:
        return {"error": doc.error, "cached": False}

    article = {
        "text":          doc.body,
        "title":         doc.title or "",
        "source_domain": doc.source_domain or "",
    }

    # ── Step 5: parallel signal fan-out (Celery group) ───────────────────────
    job = group(
        ml_signal_task.s(doc.body),
        linguistic_signal_task.s(doc.body, doc.title or ""),
        external_signal_task.s(doc.body, doc.source_domain or ""),
    )

    async_result = await asyncio.to_thread(
        lambda: job.apply_async()
    )

    ml_result, ling_result, ext_result = await asyncio.to_thread(
        lambda: async_result.get(timeout=_SIGNAL_TIMEOUT, disable_sync_subtasks=False)
    )

    # ── Step 6: ensemble ─────────────────────────────────────────────────────
    final = await asyncio.to_thread(
        lambda: ensemble_task.delay(article, ml_result, ling_result, ext_result)
                              .get(timeout=30)
    )

    # ── Assemble full result ──────────────────────────────────────────────────
    result = {
        "verdict":           final["verdict"],
        "credibility_score": final["credibility_score"],
        "confidence":        final["confidence"],
        "signals": {
            "ml":         {"score": ml_result.get("score"),   "features": ml_result.get("features", {})},
            "linguistic": {
                "score":           ling_result.get("score"),
                "features":        ling_result.get("features", {}),
                "category_scores": ling_result.get("category_scores", {}),
                "flags":           ling_result.get("flags", []),
            },
            "external":   {"score": ext_result.get("score"),  "features": ext_result.get("features", {})},
        },
        "explanation": final.get("explanation", {}),
        "document": {
            "title":      doc.title,
            "source":     doc.source_domain,
            "word_count": doc.word_count,
            "language":   doc.language,
        },
        "cached":     False,
        "input_text": text,   # kept for DB keying; strip before returning to end users if needed
    }

    # ── Step 8: persist ───────────────────────────────────────────────────────
    write_cache(text, result)
    await asyncio.to_thread(save_to_postgres, result)

    return result
