"""
ml_signal.py
============
ML-based fake news signal using the fine-tuned RoBERTa model trained in
backend/training/train.py.

Score convention (matches linguistic_signal and external_signal):
  0.0 = likely real / trustworthy
  1.0 = likely fake / suspicious

Label convention from training:
  0 = fake  →  logit index 0
  1 = real  →  logit index 1

So:  score = 1.0 - P(label == 1)
"""

import os
import threading
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# ── Checkpoint path (relative to project root) ───────────────────────────────
# Use absolute path resolution for robustness in Docker/production environments
CHECKPOINT = os.getenv(
    "ML_CHECKPOINT",
    None  # Will compute below with proper fallback
)

# If env var not set, compute absolute path from this file's location
if not CHECKPOINT:
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to project root (backend/signals -> backend -> . )
    project_root = os.path.dirname(os.path.dirname(current_dir))
    CHECKPOINT = os.path.join(project_root, "checkpoints", "roberta-politicheck")

MAX_LEN = 256

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_tokenizer = None
_model     = None
_device    = None
_load_lock = threading.Lock()


def _load_model():
    """Load tokenizer + model once; reuse on subsequent calls."""
    global _tokenizer, _model, _device

    if _model is not None:
        return  # already loaded

    with _load_lock:
        if _model is not None:
            return

        checkpoint = os.path.abspath(CHECKPOINT)
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(
                f"ML checkpoint not found at {checkpoint}. "
                "Run backend/training/train.py first to generate the model."
            )

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        _tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        _model     = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        _model.eval()

        # Device selection: MPS → CUDA → CPU
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")

        _model = _model.to(_device)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MLSignalResult:
    """
    score    : 0.0 (likely real) → 1.0 (likely fake)
    features : raw model outputs for transparency
    error    : populated only on failure
    """
    score: float
    features: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str]     = None


# ── Main entry point ──────────────────────────────────────────────────────────

def compute(text: str) -> MLSignalResult:
    """
    Run the RoBERTa fake-news classifier on *text*.

    Parameters
    ----------
    text : article body (or any string ≥ 30 chars)

    Returns
    -------
    MLSignalResult
    """
    if not text or len(text.strip()) < 30:
        return MLSignalResult(score=0.5, error="Text too short for ML analysis")

    try:
        _load_model()
    except FileNotFoundError as exc:
        return MLSignalResult(score=0.5, error=str(exc))

    try:
        inputs = _tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = _model(**inputs).logits          # shape: (1, 2)

        logits_np = logits.cpu().numpy()[0]           # [logit_fake, logit_real]
        probs     = _softmax(logits_np)               # [p_fake, p_real]

        p_fake = float(probs[0])
        p_real = float(probs[1])

        # score = probability the article is fake (0 = real, 1 = fake)
        score = round(p_fake, 4)

        return MLSignalResult(
            score=score,
            features={
                "p_fake":        p_fake,
                "p_real":        p_real,
                "logit_fake":    float(logits_np[0]),
                "logit_real":    float(logits_np[1]),
                "model_verdict": "FAKE" if p_fake >= 0.5 else "REAL",
                "confidence":    round(max(p_fake, p_real), 4),
                "checkpoint":    CHECKPOINT,
            },
        )

    except Exception as exc:
        return MLSignalResult(score=0.5, error=f"Inference error: {exc}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ── FastAPI router ────────────────────────────────────────────────────────────

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/signals", tags=["signals"])


class _MLRequest(BaseModel):
    text: str


class _MLResponse(BaseModel):
    score: float
    features: dict
    error: str | None = None


@router.post("/ml", summary="Run ML fake-news signal on raw text")
def ml_signal_endpoint(req: _MLRequest) -> _MLResponse:
    result = compute(req.text)
    return _MLResponse(score=result.score, features=result.features, error=result.error)
