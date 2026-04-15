"""
pipeline/tasks.py
=================
Celery tasks wrapping each inference module.

Signal tasks (ml / linguistic / external) are fire-and-forget leaves that
run in parallel via a Celery group.  Each returns a plain JSON-serialisable
dict so the result backend can store it without pickling:

    {"score": float, "features": dict, "error": str | None}

ensemble_task is a thin leaf that:
  1. Builds the 19-feature vector using pre-computed signal scores (no
     second pass through the signal modules).
  2. Runs XGBoost predict_proba + SHAP explanation.
  3. Returns the full credibility result dict.

Importing this module registers all tasks with the shared celery_app.
"""

import asyncio
import logging

from backend.celery_app import celery_app

log = logging.getLogger(__name__)

# ── Retry policy shared across signal tasks ───────────────────────────────────
_SIGNAL_RETRY = dict(max_retries=2, default_retry_delay=5)


# ─────────────────────────────────────────────────────────────────────────────
# ML signal
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="ml_signal_task", **_SIGNAL_RETRY)
def ml_signal_task(self, text: str) -> dict:
    """Run the fine-tuned RoBERTa classifier on *text*."""
    try:
        from backend.signals.ml_signal import compute
        result = compute(text)
        return {"score": result.score, "features": result.features, "error": result.error}
    except Exception as exc:
        log.error("ml_signal_task error: %s", exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"score": 0.5, "features": {}, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Linguistic signal
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="linguistic_signal_task", **_SIGNAL_RETRY)
def linguistic_signal_task(self, text: str, title: str = "") -> dict:
    """Detect aggressive / manipulative language patterns."""
    try:
        from backend.signals.linguistic_signal import compute
        result = compute(text, title)
        return {
            "score":           result.score,
            "features":        result.features,
            "category_scores": result.category_scores,
            "flags":           result.flags,
            "error":           result.error,
        }
    except Exception as exc:
        log.error("linguistic_signal_task error: %s", exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"score": 0.5, "features": {}, "category_scores": {}, "flags": [], "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# External signal
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="external_signal_task", **_SIGNAL_RETRY)
def external_signal_task(self, text: str, domain: str = "") -> dict:
    """Google FactCheck + source reputation (async internally)."""
    try:
        from backend.signals.external_signal import compute
        # compute() is an async coroutine — run it synchronously inside the worker
        result = asyncio.run(compute(text, domain))
        return {"score": result.score, "features": result.features, "error": result.error}
    except Exception as exc:
        log.error("external_signal_task error: %s", exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"score": 0.5, "features": {}, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble task
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="ensemble_task", max_retries=1, default_retry_delay=2)
def ensemble_task(
    self,
    article: dict,
    ml_result: dict,
    ling_result: dict,
    ext_result: dict,
) -> dict:
    """
    Build the 19-feature vector from pre-computed signal scores and run the
    XGBoost ensemble.  Returns the full credibility result dict.

    Uses build_feature_row_precomputed() so signal modules are NOT called again.
    """
    try:
        import numpy as np
        from backend.ensemble.ensemble import get_ensemble
        from backend.ensemble.feature_builder import build_feature_row_precomputed

        features = build_feature_row_precomputed(
            article,
            ml_score=ml_result.get("score", 0.5),
            linguistic_score=ling_result.get("score", 0.5),
            external_score=ext_result.get("score", 0.5),
        )

        ens = get_ensemble()
        feature_cols = ens.feature_cols
        x = np.array(
            [features[c] for c in feature_cols],
            dtype=np.float32,
        ).reshape(1, -1)

        prob = float(ens.model.predict_proba(x)[0, 1])   # P(credible)

        if   prob >= 0.65: verdict = "CREDIBLE"
        elif prob <= 0.35: verdict = "FAKE"
        else:              verdict = "UNCERTAIN"

        explanation = ens.explainer.explain(features)

        return {
            "credibility_score": round(prob, 4),
            "verdict":           verdict,
            "confidence":        round(abs(prob - 0.5) * 2, 4),
            "features":          features,
            "explanation":       explanation,
        }

    except Exception as exc:
        log.error("ensemble_task error: %s", exc)
        # Fallback: weighted average of signal scores (same as legacy pipeline)
        fallback_score = (
            ml_result.get("score", 0.5)   * 0.30 +
            ling_result.get("score", 0.5) * 0.30 +
            ext_result.get("score", 0.5)  * 0.40
        )
        # signal scores are fake-probability (0=real, 1=fake);
        # credibility_score must be P(credible) = 1 - P(fake)
        fake_score   = round(fallback_score, 4)
        credibility  = round(1.0 - fake_score, 4)
        if   credibility >= 0.65: verdict = "CREDIBLE"
        elif credibility <= 0.35: verdict = "FAKE"
        else:                     verdict = "UNCERTAIN"

        return {
            "credibility_score": credibility,
            "verdict":           verdict,
            "confidence":        round(abs(credibility - 0.5) * 2, 4),
            "features":          {},
            "explanation":       {"error": str(exc)},
        }
