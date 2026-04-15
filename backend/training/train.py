# backend/training/train.py

import os
import sys
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_BASE    = "roberta-base"
CHECKPOINT    = "./checkpoints/roberta-politicheck"
DATA_PATH     = "./backend/training/merged_dataset.csv"
MAX_LEN       = 256      # lower memory usage on Apple MPS
BATCH_SIZE    = 16       # safer default on Apple MPS
GRAD_ACCUM    = 16       # keeps effective batch size reasonable
EPOCHS        = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
# ─────────────────────────────────────────────────────────────────────────────


# ── Device Setup (M2 Pro / MPS) ──────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) detected — using MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ CUDA GPU detected — using CUDA")
        return torch.device("cuda")
    else:
        print("⚠️  No GPU found — falling back to CPU (this will be slow)")
        return torch.device("cpu")


# ── Metrics ──────────────────────────────────────────────────────────────────
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)[:, 1]  # probability of class 1 (real news)

    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="binary")
    auc = roc_auc_score(labels, probs)

    print(f"\n  → Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}")
    return {
        "accuracy": acc,
        "f1":       f1,
        "auc_roc":  auc,
    }


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_and_split(path):
    if not os.path.exists(path):
        print(f"❌ Dataset not found at: {path}")
        print("   Run dataset.py first to generate the merged dataset.")
        sys.exit(1)

    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    # Validate labels
    unique_labels = df["label"].unique()
    assert set(unique_labels).issubset({0, 1}), f"Unexpected labels: {unique_labels}"

    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # 80% train / 10% val / 10% test
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    print(f"\nSplit → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return (
        DatasetDict({
            "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
        }),
        test_df  # keep a copy for final reporting
    )


# ── Tokenisation ─────────────────────────────────────────────────────────────
def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()

    # 1. Load and split dataset
    dataset, test_df = load_and_split(DATA_PATH)

    # 2. Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_BASE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    # 3. Tokenize all splits
    print("Tokenizing dataset (this may take a few minutes)...")
    tokenized = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    tokenized.set_format("torch")

    # 4. Load model
    print(f"\nLoading model: {MODEL_BASE}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE,
        num_labels=2,
    )
    model = model.to(device)

    # 5. Training arguments — tuned for M2 Pro
    os.makedirs(CHECKPOINT, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    args = TrainingArguments(
        output_dir=CHECKPOINT,

        # Training schedule
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        # Optimizer
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.1,                   # linear warmup for first 10% of steps
        lr_scheduler_type="linear",

        # Evaluation
        eval_strategy="epoch",              # evaluate after every epoch
        save_strategy="epoch",              # save checkpoint after every epoch
        load_best_model_at_end=True,        # restore best checkpoint when done
        metric_for_best_model="f1",
        greater_is_better=True,

        # Logging
        logging_dir="./logs",
        logging_steps=100,
        logging_first_step=True,

        # M2 Pro specific — NO fp16
        fp16=False,                         # MPS does not support fp16
        bf16=False,                         # not supported on MPS either

        # Misc
        seed=42,
        report_to="none",                   # set to "wandb" if you want experiment tracking
        save_total_limit=2,                 # only keep the 2 best checkpoints to save disk space
        dataloader_num_workers=0,           # set to 0 for MPS — avoids multiprocessing issues on Mac
        dataloader_pin_memory=False,        # MPS doesn't support pinned host memory
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
            # stops training early if F1 doesn't improve for 2 epochs
        ],
    )

    # 7. Train
    print("\n🚀 Starting training...")
    print(f"   Model     : {MODEL_BASE}")
    print(f"   Device    : {device}")
    print(f"   Epochs    : {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Grad accum: {GRAD_ACCUM}")
    print(f"   Max len   : {MAX_LEN}")
    print(f"   LR        : {LEARNING_RATE}")
    print("─" * 50)

    train_result = trainer.train()

    # 8. Save final model + tokenizer
    print(f"\n💾 Saving model to {CHECKPOINT}...")
    trainer.save_model(CHECKPOINT)
    tokenizer.save_pretrained(CHECKPOINT)

    # 9. Print training summary
    print("\n── Training Summary ─────────────────────────────")
    print(f"Total steps    : {train_result.global_step}")
    print(f"Training loss  : {train_result.training_loss:.4f}")
    print(f"Runtime        : {train_result.metrics.get('train_runtime', 0):.0f}s")
    print("─────────────────────────────────────────────────")

    # 10. Final evaluation on held-out test set
    print("\n📊 Final evaluation on test set...")
    test_results = trainer.evaluate(tokenized["test"])

    f1  = test_results.get("eval_f1", 0)
    auc = test_results.get("eval_auc_roc", 0)
    acc = test_results.get("eval_accuracy", 0)

    print("\n── Final Test Results ───────────────────────────")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print("\n── Target Check ─────────────────────────────────")
    print(f"F1  > 0.82 : {'✅ PASS' if f1  > 0.82 else '❌ FAIL'} ({f1:.4f})")
    print(f"AUC > 0.88 : {'✅ PASS' if auc > 0.88 else '❌ FAIL'} ({auc:.4f})")
    print(f"Acc > 84%  : {'✅ PASS' if acc > 0.84 else '❌ FAIL'} ({acc:.4f})")
    print("─────────────────────────────────────────────────")

    if f1 > 0.82 and auc > 0.88 and acc > 0.84:
        print("\n🎉 All targets met! Model is ready.")
        print(f"   Checkpoint saved at: {CHECKPOINT}")
    else:
        print("\n⚠️  Some targets not met. Try:")
        if f1 <= 0.82:
            print("   - Add more data or increase epochs to 5")
        if auc <= 0.88:
            print("   - Switch MODEL_BASE to 'roberta-large'")
        if acc <= 0.84:
            print("   - Lower learning rate to 1e-5")


if __name__ == "__main__":
    main()