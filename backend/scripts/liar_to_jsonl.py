import csv
import json
import pathlib

# LIAR fine-grained labels → binary
# pants-fire, false, barely-true → label=0 (fake)
# mostly-true, true              → label=1 (credible)
# half-true                      → skipped (ambiguous)
FAKE_LABELS = {"pants-fire", "false", "barely-true"}
REAL_LABELS = {"mostly-true", "true"}

RAW_DIR = pathlib.Path(__file__).parent.parent / "training" / "raw" / "liar"
OUT_FILE = pathlib.Path(__file__).parent.parent / "training" / "liar_labelled.jsonl"

SPLITS = {
    "train":      RAW_DIR / "train.tsv",
    "validation": RAW_DIR / "valid.tsv",
    "test":       RAW_DIR / "test.tsv",
}

# TSV columns (no header row in raw LIAR files)
COLS = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
]

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

written = 0
with OUT_FILE.open("w") as f:
    for split, path in SPLITS.items():
        with path.open() as tsv:
            reader = csv.DictReader(tsv, fieldnames=COLS, delimiter="\t")
            for row in reader:
                liar_label = row["label"].strip()
                if liar_label in FAKE_LABELS:
                    binary = 0
                elif liar_label in REAL_LABELS:
                    binary = 1
                else:
                    continue    # skip half-true

                statement = row["statement"].strip()
                obj = {
                    "text":          statement,
                    "title":         statement[:80],
                    "source_domain": "politifact.com",
                    "label":         binary,
                    "pub_date":      "",
                }
                f.write(json.dumps(obj) + "\n")
                written += 1

print(f"Done → {OUT_FILE}  ({written} rows)")