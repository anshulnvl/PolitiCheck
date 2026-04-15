# backend/training/dataset.py

from datasets import load_dataset
import pandas as pd
import os
import json

OUTPUT_PATH = "./backend/training/merged_dataset.csv"

def load_liar():
    """Load LIAR directly from downloaded TSV files — no HuggingFace needed"""
    print("Loading LIAR dataset from TSV files...")

    # Column names for the LIAR dataset (it has no header row)
    columns = [
        "id", "label", "statement", "subject", "speaker",
        "job_title", "state", "party", "barely_true_count",
        "false_count", "half_true_count", "mostly_true_count",
        "pants_fire_count", "context"
    ]

    label_map = {
        "true": 1, "mostly-true": 1, "half-true": 1,
        "barely-true": 0, "false": 0, "pants-fire": 0
    }

    rows = []
    for filename in ["train.tsv", "valid.tsv", "test.tsv"]:
        path = f"./backend/training/raw/liar/{filename}"
        df = pd.read_csv(path, sep="\t", names=columns, header=None)
        for _, row in df.iterrows():
            label = label_map.get(row["label"], None)
            if label is not None:
                rows.append({"text": row["statement"], "label": label})

    result = pd.DataFrame(rows)
    print(f"  Loaded {len(result)} rows from LIAR")
    return result

def load_isot():
    """ISOT: 44,000 articles — True/Fake split"""
    print("Loading ISOT dataset...")

    true_path = "./backend/training/raw/isot/True.csv"
    fake_path = "./backend/training/raw/isot/Fake.csv"

    # Check both files exist before proceeding
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        print("  ⚠️  ISOT not found — skipping. Download from Kaggle.")
        return pd.DataFrame()

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df["label"] = 1
    fake_df["label"] = 0

    combined = pd.concat([true_df, fake_df], ignore_index=True)

    # Rename 'text' column first to avoid conflict when building the combined text
    combined = combined.rename(columns={"text": "body"})
    combined["text"] = combined["title"].fillna("") + " " + combined["body"].fillna("")

    result = combined[["text", "label"]]
    print(f"  Loaded {len(result)} rows from ISOT")
    return result

def load_fakenewsnet():
    print("Loading FakeNewsNet...")
    base = "./backend/training/raw/fakenewsnet"
    frames = []
    for filename, label in [
        ("politifact_fake.csv", 0),
        ("politifact_real.csv", 1),
        ("gossipcop_fake.csv",  0),
        ("gossipcop_real.csv",  1),
    ]:
        path = f"{base}/{filename}"
        if not os.path.exists(path):
            print(f"  ⚠️  {filename} not found — skipping")
            continue
        df = pd.read_csv(path)
        # these CSVs have a 'title' column — use it as the text
        df = df[["title"]].dropna()
        df = df.rename(columns={"title": "text"})
        df["label"] = label
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(result)} rows from FakeNewsNet")
    return result

def load_fever():
    print("Loading FEVER...")
    base = "./backend/training/raw/fever"
    rows = []

    # FEVER labels: SUPPORTS=real(1), REFUTES=fake(0), NOT ENOUGH INFO=skip
    label_map = {"SUPPORTS": 1, "REFUTES": 0}

    for filename in ["train.jsonl", "shared_task_dev.jsonl"]:
        path = f"{base}/{filename}"
        if not os.path.exists(path):
            print(f"  ⚠️  {filename} not found — skipping")
            continue
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                label = label_map.get(item.get("label"), None)
                if label is not None:
                    rows.append({"text": item["claim"], "label": label})

    result = pd.DataFrame(rows)
    print(f"  Loaded {len(result)} rows from FEVER")
    return result

def load_welfake():
    """WELFake: 72,000 articles from 4 sources"""
    print("Loading WELFake dataset...")
    # Download from: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
    path = "./backend/training/raw/welfake/WELFake_Dataset.csv"

    if not os.path.exists(path):
        print("  ⚠️  WELFake not found — skipping. Download from Kaggle.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.rename(columns={"Label": "label"})
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    # WELFake: 1=fake, 0=real — flip to match our convention
    df["label"] = df["label"].apply(lambda x: 0 if x == 1 else 1)
    return df[["text", "label"]]


def build_merged_dataset():
    frames = []

    # LIAR (auto-downloads via HuggingFace)
    frames.append(load_liar())

    # ISOT and WELFake (manual download required)
    frames.append(load_isot())

    frames.append(load_fakenewsnet())
    frames.append(load_fever())
    frames.append(load_welfake())

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    # Clean up
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 20]   # drop very short entries
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    print(f"  Removed {before - after} duplicate rows")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"\n✅ Merged dataset: {len(df)} rows")
    print(df["label"].value_counts())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(df["label"].value_counts(normalize=True))
    return df


if __name__ == "__main__":
    build_merged_dataset()