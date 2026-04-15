# backend/training/utils.py

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_metrics(eval_pred):
    """Called by HuggingFace Trainer after each eval step."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)[:, 1]  # prob of class 1 (real)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1":       f1_score(labels, predictions, average="binary"),
        "auc_roc":  roc_auc_score(labels, probs),
    }

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)