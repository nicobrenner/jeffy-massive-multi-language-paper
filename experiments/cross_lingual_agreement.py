"""Cross-lingual classifier agreement analysis.

Feed the same English test texts to all 51 MASSIVE language classifiers
and measure pairwise agreement rates.

Output:
  - agreement_matrix.csv: 51x51 pairwise agreement rates
  - predictions.csv: per-text predictions from all classifiers
  - agreement_summary.md: markdown summary with key findings
"""

import json
import csv
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = Path("/home/nico.linux/work/jeffy/src/jeffy/pack")
OUTPUT_DIR = REPO_ROOT / "data"

# All 51 MASSIVE language pack prefixes
MASSIVE_PACKS = sorted([
    p.name for p in PACK_DIR.iterdir()
    if p.is_dir() and p.name.startswith("massive_intent_")
])

print(f"Found {len(MASSIVE_PACKS)} MASSIVE language packs")

# Step 1: Load encoder
print("Loading multilingual encoder...")
t0 = time.time()
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(f"  Encoder loaded in {time.time()-t0:.1f}s")

# Step 2: Load all classifiers
print("Loading classifiers...")
from jeffy.model_pack import load_artifact

classifiers = {}
for pack_name in MASSIVE_PACKS:
    lang_code = pack_name.replace("massive_intent_", "")
    pack_path = PACK_DIR / pack_name
    try:
        clf, scaler, manifest = load_artifact(pack_path)
        classifiers[lang_code] = {"clf": clf, "scaler": scaler, "manifest": manifest}
    except Exception as e:
        print(f"  SKIP {lang_code}: {e}")

print(f"  Loaded {len(classifiers)} classifiers")
lang_codes = sorted(classifiers.keys())

# Step 3: Load English MASSIVE test set
print("Loading English MASSIVE test set...")
from datasets import load_dataset
ds = load_dataset("mteb/amazon_massive_intent", "en", split="test")
texts = ds["text"]
true_labels = ds["label"]

print(f"  {len(texts)} test examples, {len(set(true_labels))} intents")

# Step 4: Encode all texts once
print("Encoding texts...")
t0 = time.time()
embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=128)
print(f"  Encoded {len(embeddings)} texts in {time.time()-t0:.1f}s")

# Step 5: Predict with all classifiers
print("Running predictions across all classifiers...")
all_predictions = {}  # lang_code -> list of predicted labels

for lang_code in lang_codes:
    c = classifiers[lang_code]
    scaled = c["scaler"].transform(embeddings)
    preds = c["clf"].predict(scaled)
    all_predictions[lang_code] = list(preds)

    # Also compute accuracy on English test set
    correct = sum(1 for p, t in zip(preds, true_labels) if p == t)
    acc = correct / len(preds)
    classifiers[lang_code]["en_accuracy"] = acc

print(f"  Done. {len(lang_codes)} classifiers × {len(texts)} texts = {len(lang_codes)*len(texts)} predictions")

# Step 6: Compute pairwise agreement matrix
print("Computing agreement matrix...")
n = len(lang_codes)
agreement_matrix = np.zeros((n, n))

for i in range(n):
    preds_i = all_predictions[lang_codes[i]]
    for j in range(i, n):
        preds_j = all_predictions[lang_codes[j]]
        agree = sum(1 for a, b in zip(preds_i, preds_j) if a == b)
        rate = agree / len(preds_i)
        agreement_matrix[i][j] = rate
        agreement_matrix[j][i] = rate

# Step 7: Save agreement matrix as CSV
matrix_path = OUTPUT_DIR / "agreement_matrix.csv"
with open(matrix_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([""] + lang_codes)
    for i, lang in enumerate(lang_codes):
        writer.writerow([lang] + [f"{agreement_matrix[i][j]:.4f}" for j in range(n)])
print(f"  Saved agreement matrix to {matrix_path}")

# Step 8: Save all predictions
preds_path = OUTPUT_DIR / "predictions.csv"
with open(preds_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text_idx", "true_label"] + lang_codes)
    for idx in range(len(texts)):
        row = [idx, true_labels[idx]] + [all_predictions[lc][idx] for lc in lang_codes]
        writer.writerow(row)
print(f"  Saved predictions to {preds_path}")

# Step 9: Analysis
print("\n=== ANALYSIS ===\n")

# Per-language accuracy on English test set
print("--- Accuracy on English test set (cross-lingual transfer) ---")
en_accs = [(lc, classifiers[lc]["en_accuracy"]) for lc in lang_codes]
en_accs.sort(key=lambda x: -x[1])
for lc, acc in en_accs:
    marker = " ←" if lc == "en" else ""
    print(f"  {lc:>8}: {acc*100:.1f}%{marker}")

# Overall agreement stats
off_diag = []
for i in range(n):
    for j in range(i+1, n):
        off_diag.append(agreement_matrix[i][j])
off_diag = np.array(off_diag)
print(f"\n--- Agreement stats (off-diagonal) ---")
print(f"  Mean: {off_diag.mean():.4f}")
print(f"  Median: {np.median(off_diag):.4f}")
print(f"  Min: {off_diag.min():.4f}")
print(f"  Max: {off_diag.max():.4f}")
print(f"  Std: {off_diag.std():.4f}")

# Most and least agreeing pairs
pairs = []
for i in range(n):
    for j in range(i+1, n):
        pairs.append((lang_codes[i], lang_codes[j], agreement_matrix[i][j]))
pairs.sort(key=lambda x: -x[2])

print(f"\n--- Top 15 most agreeing pairs ---")
for a, b, rate in pairs[:15]:
    print(f"  {a:>8} - {b:<8}: {rate:.4f}")

print(f"\n--- Top 15 least agreeing pairs ---")
for a, b, rate in pairs[-15:]:
    print(f"  {a:>8} - {b:<8}: {rate:.4f}")

# Per-intent disagreement analysis
print(f"\n--- Intent disagreement analysis ---")
intent_disagreement = Counter()
intent_counts = Counter(true_labels)

for idx in range(len(texts)):
    preds_for_text = [all_predictions[lc][idx] for lc in lang_codes]
    unique_preds = len(set(preds_for_text))
    if unique_preds > 1:
        intent_disagreement[true_labels[idx]] += 1

print("Intents with most cross-classifier disagreement:")
for intent, count in intent_disagreement.most_common(15):
    total = intent_counts[intent]
    print(f"  {intent:>30}: {count}/{total} texts have disagreement ({count/total*100:.0f}%)")

print("\nIntents with least disagreement:")
for intent, count in sorted(intent_disagreement.items(), key=lambda x: x[1])[:10]:
    total = intent_counts[intent]
    print(f"  {intent:>30}: {count}/{total} texts have disagreement ({count/total*100:.0f}%)")

# Full agreement (all 51 classifiers agree)
full_agree = 0
for idx in range(len(texts)):
    preds = set(all_predictions[lc][idx] for lc in lang_codes)
    if len(preds) == 1:
        full_agree += 1
print(f"\n--- Full agreement (all 51 classifiers predict same label) ---")
print(f"  {full_agree}/{len(texts)} texts ({full_agree/len(texts)*100:.1f}%)")

# Majority vote accuracy
print(f"\n--- Majority vote accuracy ---")
majority_correct = 0
for idx in range(len(texts)):
    preds = [all_predictions[lc][idx] for lc in lang_codes]
    majority = Counter(preds).most_common(1)[0][0]
    if majority == true_labels[idx]:
        majority_correct += 1
print(f"  {majority_correct}/{len(texts)} ({majority_correct/len(texts)*100:.1f}%)")

# Save summary
summary_path = OUTPUT_DIR / "agreement_summary.md"
with open(summary_path, "w") as f:
    f.write("# Cross-Lingual Classifier Agreement Analysis\n\n")
    f.write(f"Date: 2026-10-05\n")
    f.write(f"Test set: English MASSIVE test split ({len(texts)} examples, {len(set(true_labels))} intents)\n")
    f.write(f"Classifiers: {len(lang_codes)} languages\n")
    f.write(f"Encoder: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)\n\n")

    f.write("## Key Findings\n\n")
    f.write(f"- **Pairwise agreement:** mean {off_diag.mean():.3f}, median {np.median(off_diag):.3f}, std {off_diag.std():.3f}\n")
    f.write(f"- **Full agreement (all 51):** {full_agree}/{len(texts)} ({full_agree/len(texts)*100:.1f}%)\n")
    f.write(f"- **Majority vote accuracy:** {majority_correct/len(texts)*100:.1f}%\n\n")

    f.write("## Per-Language Accuracy on English Test Set\n\n")
    f.write("| Language | Accuracy |\n|----------|----------|\n")
    for lc, acc in en_accs:
        f.write(f"| {lc} | {acc*100:.1f}% |\n")

    f.write("\n## Most Agreeing Pairs\n\n")
    f.write("| Lang A | Lang B | Agreement |\n|--------|--------|----------|\n")
    for a, b, rate in pairs[:20]:
        f.write(f"| {a} | {b} | {rate:.4f} |\n")

    f.write("\n## Least Agreeing Pairs\n\n")
    f.write("| Lang A | Lang B | Agreement |\n|--------|--------|----------|\n")
    for a, b, rate in pairs[-20:]:
        f.write(f"| {a} | {b} | {rate:.4f} |\n")

    f.write("\n## Most Disputed Intents\n\n")
    f.write("| Intent | Disagreement Rate |\n|--------|------------------|\n")
    for intent, count in intent_disagreement.most_common(20):
        total = intent_counts[intent]
        f.write(f"| {intent} | {count}/{total} ({count/total*100:.0f}%) |\n")

print(f"\n  Saved summary to {summary_path}")
print("\nDone!")
