import shap
import numpy as np

class EnsembleExplainer:
    """
    Wraps a trained XGBClassifier and produces SHAP explanations.
    Use a single shared instance per process (explainer init is slow).
    """

    def __init__(self, model, feature_cols: list[str]):
        self.model        = model
        self.feature_cols = feature_cols
        # TreeExplainer is exact & fast for XGBoost
        self.explainer    = shap.TreeExplainer(model)

    def explain(self, feature_row: dict) -> dict:
        """
        feature_row: output of build_feature_row()
        Returns:
          {
            "shap_values": {feature_name: shap_value},  # all features
            "top_drivers": [                            # top 5, sorted by |shap|
                {"feature": str, "shap": float, "value": float},
            ],
            "base_value": float,      # model expected value (log-odds)
          }
        """
        x = np.array(
            [feature_row[c] for c in self.feature_cols],
            dtype=np.float32
        ).reshape(1, -1)

        sv = self.explainer.shap_values(x)  # shape (1, n_features)
        # For binary XGBoost, shap_values returns a 2D array
        shap_arr = sv[0] if isinstance(sv, list) else sv[0]

        shap_dict = {
            col: float(val)
            for col, val in zip(self.feature_cols, shap_arr)
        }

        top = sorted(
            [
                {"feature": k, "shap": v, "value": feature_row[k]}
                for k, v in shap_dict.items()
            ],
            key=lambda d: abs(d["shap"]),
            reverse=True,
        )[:5]

        return {
            "shap_values": shap_dict,
            "top_drivers": top,
            "base_value": float(self.explainer.expected_value),
        }