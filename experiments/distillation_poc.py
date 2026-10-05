"""Knowledge distillation PoC: train Jeffy LR on XLM-R soft labels.

Uses a fine-tuned XLM-R checkpoint as teacher to generate soft probability
distributions over intents, then trains LR on those soft labels instead of
hard ground-truth labels.

Teacher: cartesinus/xlm-r-base-amazon-massive-intent (HuggingFace)
Student: StandardScaler + LogisticRegression (same as Jeffy)
Encoder: paraphrase-multilingual-MiniLM-L12-v2 (frozen, same as Jeffy)

PoC subset: 6 languages spanning high/mid/low accuracy tiers.
Full 51-language run can follow if results are promising.

Output: data/distillation_poc.csv
"""

import csv
import time
import sys
import numpy as np
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data"

# PoC languages: 2 high-tier, 2 mid-tier, 2 low-tier
POC_LANGS = [
    ("en", "en-US"),
    ("fr", "fr-FR"),
    ("de", "de-DE"),
    ("ko", "ko-KR"),
    ("ja", "ja-JP"),
    ("sw", "sw-KE"),
]

MAX_TRAIN = 10000

print("Loading student encoder (multilingual-MiniLM)...")
t0 = time.time()
student_encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(f"  Loaded in {time.time()-t0:.1f}s")

print("Loading teacher model (XLM-R fine-tuned on MASSIVE)...")
t0 = time.time()
teacher_name = "cartesinus/xlm-r-base-amazon-massive-intent"
tokenizer = AutoTokenizer.from_pretrained(teacher_name)
teacher = AutoModelForSequenceClassification.from_pretrained(teacher_name)
teacher.eval()
# Get label mapping from teacher
id2label = teacher.config.id2label
label2id = teacher.config.label2id
num_labels = teacher.config.num_labels
print(f"  Loaded in {time.time()-t0:.1f}s, {num_labels} labels")
print(f"  Sample labels: {list(id2label.values())[:5]}")

def get_teacher_soft_labels(texts, batch_size=32):
    """Get teacher's probability distributions for a list of texts."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = teacher(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)

def get_teacher_hard_predictions(probs):
    """Convert soft labels to hard predictions."""
    pred_ids = np.argmax(probs, axis=1)
    return [id2label[i] for i in pred_ids]

results = []

for lang, ds_lang in POC_LANGS:
    print(f"\n{'='*60}")
    print(f"LANGUAGE: {lang} ({ds_lang})")
    print(f"{'='*60}")

    train_ds = load_dataset("mteb/amazon_massive_intent", lang, split="train")
    test_ds = load_dataset("mteb/amazon_massive_intent", lang, split="test")

    train_texts = train_ds["text"][:MAX_TRAIN]
    train_labels = train_ds["label"][:MAX_TRAIN]
    test_texts = test_ds["text"]
    test_labels = test_ds["label"]

    # Encode with student encoder
    print("  Encoding with student encoder...")
    t0 = time.time()
    train_emb = student_encoder.encode(train_texts, show_progress_bar=False, batch_size=128)
    test_emb = student_encoder.encode(test_texts, show_progress_bar=False, batch_size=128)
    encode_time = time.time() - t0
    print(f"  Encoded in {encode_time:.1f}s")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_emb)
    test_scaled = scaler.transform(test_emb)

    # --- Baseline: LR on hard ground-truth labels ---
    print("  Training baseline (hard labels)...")
    clf_hard = LogisticRegression(C=0.01, solver="newton-cg", max_iter=200, n_jobs=-1)
    clf_hard.fit(train_scaled, train_labels)
    hard_test_acc = clf_hard.score(test_scaled, test_labels)
    print(f"  Baseline accuracy: {hard_test_acc:.4f}")

    # --- Get teacher soft labels ---
    print("  Getting teacher soft labels on training data...")
    t0 = time.time()
    train_soft = get_teacher_soft_labels(list(train_texts), batch_size=32)
    teacher_time = time.time() - t0
    print(f"  Teacher inference: {teacher_time:.1f}s")

    # Teacher's own accuracy on test set
    print("  Getting teacher predictions on test data...")
    test_soft = get_teacher_soft_labels(list(test_texts), batch_size=32)
    teacher_preds = get_teacher_hard_predictions(test_soft)
    teacher_test_acc = sum(1 for p, t in zip(teacher_preds, test_labels) if p == t) / len(test_labels)
    print(f"  Teacher test accuracy: {teacher_test_acc:.4f}")

    # --- Distillation: LR on soft labels ---
    # Use teacher's hard predictions as labels (hard distillation)
    teacher_train_preds = get_teacher_hard_predictions(train_soft)
    print("  Training distilled model (teacher hard predictions)...")
    clf_distill_hard = LogisticRegression(C=0.01, solver="newton-cg", max_iter=200, n_jobs=-1)
    clf_distill_hard.fit(train_scaled, teacher_train_preds)
    distill_hard_acc = clf_distill_hard.score(test_scaled, test_labels)
    print(f"  Distilled (hard) accuracy: {distill_hard_acc:.4f}")

    # Soft distillation: train on soft probability targets
    # LR doesn't natively support soft targets, so we use a weighted approach:
    # For each sample, create multiple weighted copies based on top-k teacher probs
    print("  Training distilled model (soft labels, top-5 expansion)...")
    top_k = 5
    expanded_X = []
    expanded_y = []
    expanded_w = []
    for idx in range(len(train_texts)):
        probs = train_soft[idx]
        top_indices = np.argsort(probs)[-top_k:]
        for ti in top_indices:
            if probs[ti] > 0.01:
                expanded_X.append(train_scaled[idx])
                expanded_y.append(id2label[ti])
                expanded_w.append(probs[ti])

    expanded_X = np.array(expanded_X)
    expanded_w = np.array(expanded_w)

    clf_distill_soft = LogisticRegression(C=0.01, solver="newton-cg", max_iter=200, n_jobs=-1)
    clf_distill_soft.fit(expanded_X, expanded_y, sample_weight=expanded_w)
    distill_soft_acc = clf_distill_soft.score(test_scaled, test_labels)
    print(f"  Distilled (soft) accuracy: {distill_soft_acc:.4f}")

    # Summary
    improvement_hard = distill_hard_acc - hard_test_acc
    improvement_soft = distill_soft_acc - hard_test_acc
    recovery_hard = (distill_hard_acc - hard_test_acc) / (teacher_test_acc - hard_test_acc) * 100 if teacher_test_acc > hard_test_acc else 0
    recovery_soft = (distill_soft_acc - hard_test_acc) / (teacher_test_acc - hard_test_acc) * 100 if teacher_test_acc > hard_test_acc else 0

    print(f"\n  SUMMARY for {lang}:")
    print(f"    Baseline (hard labels):      {hard_test_acc:.4f}")
    print(f"    Distilled (teacher hard):     {distill_hard_acc:.4f}  ({improvement_hard:+.4f}, {recovery_hard:.0f}% gap recovery)")
    print(f"    Distilled (teacher soft):     {distill_soft_acc:.4f}  ({improvement_soft:+.4f}, {recovery_soft:.0f}% gap recovery)")
    print(f"    Teacher (XLM-R fine-tuned):   {teacher_test_acc:.4f}")

    results.append({
        "lang": lang,
        "baseline_acc": round(hard_test_acc, 4),
        "distill_hard_acc": round(distill_hard_acc, 4),
        "distill_soft_acc": round(distill_soft_acc, 4),
        "teacher_acc": round(teacher_test_acc, 4),
        "improvement_hard": round(improvement_hard, 4),
        "improvement_soft": round(improvement_soft, 4),
        "gap_recovery_hard_pct": round(recovery_hard, 1),
        "gap_recovery_soft_pct": round(recovery_soft, 1),
        "teacher_inference_s": round(teacher_time, 1),
    })

# Save
out_path = OUTPUT_DIR / "distillation_poc.csv"
fields = list(results[0].keys())
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved to {out_path}")

# Overall summary
print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
print(f"{'Lang':<6} {'Baseline':>9} {'Dist-Hard':>10} {'Dist-Soft':>10} {'Teacher':>9} {'Gap Recov':>10}")
for r in results:
    print(f"{r['lang']:<6} {r['baseline_acc']:>9.3f} {r['distill_hard_acc']:>10.3f} {r['distill_soft_acc']:>10.3f} {r['teacher_acc']:>9.3f} {r['gap_recovery_soft_pct']:>9.1f}%")

mean_baseline = np.mean([r["baseline_acc"] for r in results])
mean_distill = np.mean([r["distill_soft_acc"] for r in results])
mean_teacher = np.mean([r["teacher_acc"] for r in results])
print(f"\n{'Mean':<6} {mean_baseline:>9.3f} {'':>10} {mean_distill:>10.3f} {mean_teacher:>9.3f}")
print(f"\nDistillation closes {((mean_distill - mean_baseline) / (mean_teacher - mean_baseline)) * 100:.0f}% of the gap on average.")
print("\nDone!")
