# backend/training/evaluate.py

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)

CHECKPOINT = "./checkpoints/roberta-politicheck"
DATA_PATH  = "./backend/training/merged_dataset.csv"
MAX_LEN    = 512
BATCH_SIZE = 32


def predict_batch(texts, model, tokenizer, device):
    inputs = tokenizer(
        texts, truncation=True, padding=True,
        max_length=MAX_LEN, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def evaluate():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT).to(device)
    model.eval()

    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    test_df = df.sample(frac=0.1, random_state=99)  # use 10% as held-out test

    all_probs = []
    texts = test_df["text"].tolist()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        all_probs.extend(predict_batch(batch, model, tokenizer, device))

    all_probs  = np.array(all_probs)
    preds      = np.argmax(all_probs, axis=-1)
    labels     = test_df["label"].values

    print("\n── Evaluation Results ──────────────────────────")
    print(f"Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"F1 Score : {f1_score(labels, preds):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(labels, all_probs[:, 1]):.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Fake", "Real"]))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    print("────────────────────────────────────────────────")

    # Check against targets
    f1  = f1_score(labels, preds)
    auc = roc_auc_score(labels, all_probs[:, 1])
    acc = accuracy_score(labels, preds)

    print("\n── Target Check ────────────────────────────────")
    print(f"F1  > 0.82  : {'✅ PASS' if f1 > 0.82 else '❌ FAIL'} ({f1:.4f})")
    print(f"AUC > 0.88  : {'✅ PASS' if auc > 0.88 else '❌ FAIL'} ({auc:.4f})")
    print(f"Acc > 84%   : {'✅ PASS' if acc > 0.84 else '❌ FAIL'} ({acc:.4f})")
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    evaluate()