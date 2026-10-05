"""Encoder ablation: compare frozen multilingual encoders for MASSIVE intent classification.

Tests multiple sentence encoders with the same LR pipeline to isolate
encoder contribution to classification accuracy.

Encoders tested:
  1. paraphrase-multilingual-MiniLM-L12-v2 (384-dim) — current Jeffy encoder
  2. paraphrase-multilingual-mpnet-base-v2 (768-dim) — larger paraphrase model
  3. LaBSE (768-dim) — Google's language-agnostic BERT

Output: data/encoder_ablation.csv
"""

import csv
import time
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data"

ENCODERS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("paraphrase-multilingual-mpnet-base-v2", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    ("LaBSE", "sentence-transformers/LaBSE"),
]

LANGS = [
    "af","am","ar","az","bn","cy","da","de","el","en","es","fa","fi","fr",
    "he","hi","hu","hy","id","is","it","ja","jv","ka","km","kn","ko","lv",
    "ml","mn","ms","my","nb","nl","pl","pt","ro","ru","sl","sq","sv","sw",
    "ta","te","th","tl","tr","ur","vi","zh_cn","zh_tw"
]

DATASET_LANG_MAP = {
    "zh_cn": "zh-CN", "zh_tw": "zh-TW",
}

MAX_TRAIN = 10000

results = []

for enc_name, enc_path in ENCODERS:
    print(f"\n{'='*60}")
    print(f"ENCODER: {enc_name}")
    print(f"{'='*60}")

    t0 = time.time()
    encoder = SentenceTransformer(enc_path)
    dim = encoder.get_sentence_embedding_dimension()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s, dim={dim}")

    for lang in LANGS:
        ds_lang = DATASET_LANG_MAP.get(lang, lang)
        try:
            train_ds = load_dataset("mteb/amazon_massive_intent", ds_lang, split="train")
            test_ds = load_dataset("mteb/amazon_massive_intent", ds_lang, split="test")
        except Exception as e:
            print(f"  SKIP {lang}: {e}")
            results.append({
                "encoder": enc_name, "dim": dim, "lang": lang,
                "test_acc": None, "train_acc": None, "error": str(e)
            })
            continue

        train_texts = train_ds["text"][:MAX_TRAIN]
        train_labels = train_ds["label"][:MAX_TRAIN]
        test_texts = test_ds["text"]
        test_labels = test_ds["label"]

        t1 = time.time()
        train_emb = encoder.encode(train_texts, show_progress_bar=False, batch_size=128)
        test_emb = encoder.encode(test_texts, show_progress_bar=False, batch_size=128)
        encode_time = time.time() - t1

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_emb)
        test_scaled = scaler.transform(test_emb)

        clf = LogisticRegression(C=0.01, solver="newton-cg", max_iter=200, n_jobs=-1)
        t2 = time.time()
        clf.fit(train_scaled, train_labels)
        train_time = time.time() - t2

        train_acc = clf.score(train_scaled, train_labels)
        test_acc = clf.score(test_scaled, test_labels)

        results.append({
            "encoder": enc_name, "dim": dim, "lang": lang,
            "test_acc": round(test_acc, 4), "train_acc": round(train_acc, 4),
            "encode_time_s": round(encode_time, 1), "train_time_s": round(train_time, 1),
            "n_train": len(train_texts), "n_test": len(test_texts),
            "error": ""
        })
        print(f"  {lang}: test={test_acc:.3f} train={train_acc:.3f} ({encode_time:.1f}s encode, {train_time:.1f}s train)")

    del encoder
    print(f"  Encoder {enc_name} complete.")

# Save results
out_path = OUTPUT_DIR / "encoder_ablation.csv"
fields = ["encoder","dim","lang","test_acc","train_acc","encode_time_s","train_time_s","n_train","n_test","error"]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in results:
        row = {k: r.get(k, "") for k in fields}
        writer.writerow(row)
print(f"\nSaved results to {out_path}")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for enc_name, _ in ENCODERS:
    enc_results = [r for r in results if r["encoder"] == enc_name and r.get("test_acc") is not None]
    if enc_results:
        accs = [r["test_acc"] for r in enc_results]
        mean_acc = np.mean(accs)
        median_acc = np.median(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        dim = enc_results[0]["dim"]
        print(f"\n{enc_name} ({dim}-dim):")
        print(f"  Mean:   {mean_acc:.3f}")
        print(f"  Median: {median_acc:.3f}")
        print(f"  Range:  {min_acc:.3f} - {max_acc:.3f}")

print("\nDone!")
