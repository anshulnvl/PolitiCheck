"""
Merge N jsonl files, deduplicate by title hash, balance classes.
Usage:
  python3 scripts/merge_datasets.py \
    --inputs backend/training/raw/isot/labelled_articles.jsonl backend/training/raw/liar/liar_labelled.jsonl \
    --out    data/combined.jsonl \
    --max_per_class 15000
"""
import argparse, json, hashlib, random
from collections import defaultdict

def title_hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()

def merge(inputs, out_path, max_per_class):
    seen   = set()
    buckets = defaultdict(list)

    for path in inputs:
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                h   = title_hash(obj.get("title", obj.get("text", "")[:80]))
                if h in seen:
                    continue
                seen.add(h)
                buckets[obj["label"]].append(obj)

    for label, items in buckets.items():
        random.shuffle(items)
        buckets[label] = items[:max_per_class]

    all_items = buckets[0] + buckets[1]
    random.shuffle(all_items)

    with open(out_path, "w") as f:
        for obj in all_items:
            f.write(json.dumps(obj) + "\n")

    print(f"Merged → {out_path}")
    print(f"  fake={len(buckets[0])}  real={len(buckets[1])}  total={len(all_items)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out",    default="data/combined.jsonl")
    p.add_argument("--max_per_class", type=int, default=15000)
    args = p.parse_args()
    merge(args.inputs, args.out, args.max_per_class)
