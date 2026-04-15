"""
Backend package for PolitiCheck.

Subpackages:
  - api: FastAPI application and routers
  - ensemble: XGBoost ensemble model and SHAP explainer
  - pipeline: Orchestrator, tasks, caching, and database
  - signals: ML, linguistic, and external fact-checking signals
  - training: Model training and evaluation utilities
"""

from .celery_app import celery_app

__all__ = ["celery_app"]
