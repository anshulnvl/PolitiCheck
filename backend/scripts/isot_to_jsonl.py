"""
Usage:
  

"""
import argparse, json, re, hashlib
import pandas as pd
from urllib.parse import urlparse

def extract_domain(text: str) -> str:
    # ISOT "text" column sometimes starts with "CITY, Month DD (Reuters) -"
    m = re.match(r'^([A-Z]+(?:/[A-Z]+)?)\s*\([Rr]euters\)', text)
    if m:
        return "reuters.com"
    return "unknown.com"

def convert(true_path, fake_path, out_path):
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df["label"]         = 1
    fake_df["label"]         = 0
    true_df["source_domain"] = true_df["text"].apply(extract_domain)
    fake_df["source_domain"] = "unknown.com"

    df = pd.concat([true_df, fake_df], ignore_index=True).sample(frac=1, random_state=42)

    written = 0
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            text  = str(row.get("text",  "")).strip()
            title = str(row.get("title", "")).strip()
            if len(text) < 50:   # skip stubs
                continue
            obj = {
                "text":          text,
                "title":         title,
                "source_domain": str(row.get("source_domain", "unknown.com")),
                "label":         int(row["label"]),
                "pub_date":      str(row.get("date", "")),
            }
            f.write(json.dumps(obj) + "\n")
            written += 1

    fake_ct = df[df.label == 0].shape[0]
    real_ct = df[df.label == 1].shape[0]
    print(f"Written {written} articles → {out_path}")
    print(f"  fake={fake_ct}  real={real_ct}  ratio={fake_ct/max(real_ct,1):.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--true", required=True)
    p.add_argument("--fake", required=True)
    p.add_argument("--out",  default="data/labelled_articles.jsonl")
    args = p.parse_args()
    convert(args.true, args.fake, args.out)