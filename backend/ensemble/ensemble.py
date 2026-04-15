import json
import os
import threading
import numpy as np
import xgboost as xgb
from .feature_builder import build_feature_row
from .shap_explainer import EnsembleExplainer

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../../checkpoints/ensemble_xgb.json"
)

_instance_lock = threading.Lock()  # Thread-safe singleton access


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
        meta_path = model_path.replace(".json", "_meta.json")

        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        # Load feature column order (MUST match training order)
        with open(meta_path) as f:
            self.feature_cols = json.load(f)["feature_cols"]

        self.explainer = EnsembleExplainer(self.model, self.feature_cols)

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

        explanation = self.explainer.explain(features)

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