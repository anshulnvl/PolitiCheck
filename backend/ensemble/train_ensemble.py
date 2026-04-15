"""
Run from project root:
    python3 -m backend.ensemble.train_ensemble \
        --data backend/combined.jsonl \
        --out  backend/checkpoints/ensemble_xgb.json
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy  as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score
)
from .feature_builder import build_feature_row

from tqdm import tqdm

def _feature_cache_path(data_path: str, offline: bool) -> Path:
    data_file = Path(data_path)
    suffix = ".offline.features.jsonl" if offline else ".features.jsonl"
    return data_file.with_suffix(suffix)


def _build_feature_row_from_json(line: str, offline: bool) -> dict:
    obj = json.loads(line)
    feats = build_feature_row(obj, offline=offline)
    feats["label"] = obj["label"]
    return feats


def load_or_build_features(
    data_path: str,
    workers: int | None = None,
    offline: bool = True,
) -> pd.DataFrame:
    cache_path = _feature_cache_path(data_path, offline=offline)

    if cache_path.exists() and cache_path.stat().st_mtime >= Path(data_path).stat().st_mtime:
        print(f"Loading cached features from {cache_path}...")
        return pd.read_json(cache_path, lines=True)

    print("Building features (first run only)...")
    with open(data_path) as f:
        lines = f.readlines()

    worker_count = 1 if workers is None else max(1, workers)
    if worker_count == 1:
        records = [
            _build_feature_row_from_json(line, offline=offline)
            for line in tqdm(lines, desc="Building features")
        ]
    else:
        print(f"  Using {worker_count} workers")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            records = list(
                tqdm(
                    executor.map(lambda line: _build_feature_row_from_json(line, offline=offline), lines),
                    total=len(lines),
                    desc="Building features",
                )
            )

    df = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(cache_path, orient="records", lines=True)
    return df

def load_dataset(path: str) -> pd.DataFrame:
    return load_or_build_features(path)

def train(data_path: str, out_path: str, workers: int | None = None, offline: bool = True):
    print("[1/4] Loading dataset...")
    df = load_or_build_features(data_path, workers=workers, offline=offline)
    print(f"  → {len(df)} articles, class balance: {df['label'].value_counts().to_dict()}")

    FEATURE_COLS = [c for c in df.columns if c != "label"]
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[2/4] Training XGBoost (this takes ~1-2 min on CPU)...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        tree_method="hist",       # fast even on CPU
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print("[3/4] Evaluating...")
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["fake", "credible"]))
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  PR-AUC   : {average_precision_score(y_test, y_prob):.4f}")

    print(f"[4/4] Saving model → {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save_model(out_path)
    # Also save feature column order — critical for consistent inference
    meta_path = out_path.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"feature_cols": FEATURE_COLS}, f, indent=2)
    print(f"  Feature metadata → {meta_path}")
    print("Done ✓")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="backend/checkpoints/ensemble_xgb.json")
    p.add_argument("--workers", type=int, default=None, help="Parallel feature-building workers")
    p.add_argument("--live-signals", action="store_true", help="Use live network-backed external signals during feature building")
    args = p.parse_args()
    train(args.data, args.out, workers=args.workers, offline=not args.live_signals)