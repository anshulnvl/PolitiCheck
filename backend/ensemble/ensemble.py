import json
import os
import threading
import logging
import numpy as np
import xgboost as xgb
from .feature_builder import build_feature_row
from .shap_explainer import EnsembleExplainer

_THIS_DIR = os.path.dirname(__file__)


def _resolve_model_path() -> str:
    """
    Resolve ensemble checkpoint path from environment or known defaults.

    Search order:
      1) ENSEMBLE_MODEL_PATH (if set)
      2) backend/checkpoints/ensemble_xgb.json
      3) project-root/checkpoints/ensemble_xgb.json
    """
    env_path = os.getenv("ENSEMBLE_MODEL_PATH")
    candidates = []
    if env_path:
        candidates.append(os.path.abspath(env_path))

    candidates.extend([
        os.path.abspath(os.path.join(_THIS_DIR, "../checkpoints/ensemble_xgb.json")),
        os.path.abspath(os.path.join(_THIS_DIR, "../../checkpoints/ensemble_xgb.json")),
    ])

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Keep deterministic behavior for error messages when files are missing.
    return candidates[0]


DEFAULT_MODEL_PATH = _resolve_model_path()

_instance_lock = threading.Lock()  # Thread-safe singleton access
log = logging.getLogger(__name__)


class Ensemble:
    """
    Thread-safe singleton wrapper for XGBoost ensemble model.
    
    Uses double-checked locking pattern for lazy initialization.
    Supports SHAP explanations for model interpretability.

    Usage:
        ensemble = Ensemble()                     # loads from default path
        result   = ensemble.predict(article_dict)
    """

    _instance = None  # module-level singleton

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        model_path = os.path.abspath(model_path)
        meta_path = model_path.replace(".json", "_meta.json")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "Ensemble model checkpoint not found. Checked path: "
                f"{model_path}. Set ENSEMBLE_MODEL_PATH to override."
            )
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                "Ensemble model metadata not found. Expected: "
                f"{meta_path}."
            )

        # Load XGBoost model
        # Force single-threaded inference to avoid OpenMP segfaults on macOS
        # (XGBoost's internal threading conflicts with forked/spawned processes).
        self.model = xgb.XGBClassifier(nthread=1)
        self.model.load_model(model_path)
        self.model.set_params(nthread=1)

        # Load feature column order (MUST match training order)
        with open(meta_path) as f:
            self.feature_cols = json.load(f)["feature_cols"]

        # OMP_NUM_THREADS=1 (set in celery_app.py) prevents the OpenMP clash
        # between PyTorch and XGBoost that previously caused crashes on macOS,
        # so SHAP TreeExplainer is now safe to use on all platforms.
        self.enable_shap = os.getenv("ENSEMBLE_ENABLE_SHAP", "true").lower() == "true"
        self.explainer = None

        if self.enable_shap:
            try:
                self.explainer = EnsembleExplainer(self.model, self.feature_cols)
            except Exception as exc:
                # Never fail inference if explanation backend is unavailable.
                log.warning("SHAP disabled after initialization error: %s", exc)
                self.enable_shap = False

    def explain(self, features: dict) -> dict:
        if not self.enable_shap or self.explainer is None:
            return {
                "top_drivers": [],
                "shap_values": {},
                "base_value": 0.0,
                "warning": "SHAP explanation disabled",
            }

        try:
            return self.explainer.explain(features)
        except Exception as exc:
            return {
                "top_drivers": [],
                "shap_values": {},
                "base_value": 0.0,
                "error": f"SHAP explanation failed: {exc}",
            }

    def predict(self, article: dict) -> dict:
        """
        article: {"text": str, "title": str, "source_domain": str}
        Returns:
          {
            "credibility_score": float,     # 0.0 (fake) → 1.0 (credible)
            "verdict": "CREDIBLE"|"FAKE"|"UNCERTAIN",
            "confidence": float,            # distance from 0.5
            "features": {name: value},      # raw feature row
            "explanation": {
                "top_drivers": [...],       # SHAP top-5
                "shap_values": {...},
                "base_value": float,
            },
          }
        """
        features = build_feature_row(article)
        x = np.array(
            [features[c] for c in self.feature_cols],
            dtype=np.float32,
        ).reshape(1, -1)

        prob = float(self.model.predict_proba(x)[0, 1])  # P(credible)

        if   prob >= 0.65: verdict = "CREDIBLE"
        elif prob <= 0.35: verdict = "FAKE"
        else:              verdict = "UNCERTAIN"

        explanation = self.explain(features)

        return {
            "credibility_score": prob,
            "verdict":           verdict,
            "confidence":        abs(prob - 0.5) * 2,  # 0→1
            "features":          features,
            "explanation":       explanation,
        }


def get_ensemble() -> Ensemble:
    """
    Lazy-load singleton ensemble instance (thread-safe).
    
    Double-checked locking pattern ensures only one instance exists
    even under concurrent access from multiple threads/tasks.
    
    Returns:
        Ensemble: Cached singleton instance
    """
    if Ensemble._instance is None:
        with _instance_lock:
            if Ensemble._instance is None:
                Ensemble._instance = Ensemble()
    return Ensemble._instance