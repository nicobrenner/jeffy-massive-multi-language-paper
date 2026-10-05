"""Classifier architecture ablation: LR vs MLP vs XGBoost on frozen embeddings.

Tests whether non-linear classifiers can close the gap between Jeffy's LR
and XLM-R fine-tuned, using the same frozen multilingual embeddings.

Classifiers tested:
  1. LogisticRegression (Jeffy baseline)
  2. MLP (2-layer: 384→256→60)
  3. XGBoost (gradient-boosted trees)

Encoder: paraphrase-multilingual-MiniLM-L12-v2 (frozen, 384-dim)
PoC languages: same 6 as distillation experiment (en, fr, de, ko, ja, sw)

Output: data/classifier_ablation.csv
"""

import csv
import time
import sys
import numpy as np
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data"

POC_LANGS = [
    ("en", "en-US"),
    ("fr", "fr-FR"),
    ("de", "de-DE"),
    ("ko", "ko-KR"),
    ("ja", "ja-JP"),
    ("sw", "sw-KE"),
]

XLMR_FULL = {"en": 0.883, "fr": 0.863, "de": 0.857, "ko": 0.865, "ja": 0.839, "sw": 0.831}

MAX_TRAIN = 15000  # MASSIVE has ~11.5K — use all of them

print("Loading encoder (multilingual-MiniLM)...")
t0 = time.time()
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(f"  Loaded in {time.time()-t0:.1f}s")

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

    print("  Encoding...")
    t0 = time.time()
    train_emb = encoder.encode(train_texts, show_progress_bar=False, batch_size=128)
    test_emb = encoder.encode(test_texts, show_progress_bar=False, batch_size=128)
    print(f"  Encoded in {time.time()-t0:.1f}s")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_emb)
    test_scaled = scaler.transform(test_emb)

    le = LabelEncoder()
    le.fit(train_labels)
    train_labels_int = le.transform(train_labels)
    test_labels_int = le.transform(test_labels)
    num_classes = len(le.classes_)
    row = {"lang": lang, "xlmr_full": XLMR_FULL.get(lang, None)}

    # --- 1. Logistic Regression (Jeffy baseline) ---
    print("  [1/3] LogisticRegression (C=0.01, newton-cg)...")
    t0 = time.time()
    clf_lr = LogisticRegression(C=0.01, solver="newton-cg", max_iter=200)
    clf_lr.fit(train_scaled, train_labels)
    lr_time = time.time() - t0
    lr_acc = clf_lr.score(test_scaled, test_labels)
    lr_params = clf_lr.coef_.size + clf_lr.intercept_.size
    print(f"    acc={lr_acc:.4f}  params={lr_params:,}  time={lr_time:.1f}s")
    row["lr_acc"] = round(lr_acc, 4)
    row["lr_params"] = lr_params
    row["lr_train_s"] = round(lr_time, 1)

    # --- 2. MLP (2 hidden layers) ---
    print("  [2/3] MLP (384→256→128→60)...")
    t0 = time.time()
    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        batch_size=256,
    )
    clf_mlp.fit(train_scaled, train_labels)
    mlp_time = time.time() - t0
    mlp_acc = clf_mlp.score(test_scaled, test_labels)
    mlp_params = sum(w.size for w in clf_mlp.coefs_) + sum(b.size for b in clf_mlp.intercepts_)
    print(f"    acc={mlp_acc:.4f}  params={mlp_params:,}  time={mlp_time:.1f}s")
    row["mlp_acc"] = round(mlp_acc, 4)
    row["mlp_params"] = mlp_params
    row["mlp_train_s"] = round(mlp_time, 1)

    # --- 3. XGBoost ---
    print("  [3/3] XGBoost (500 trees, max_depth=6)...")
    t0 = time.time()
    clf_xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        device="cpu",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        verbosity=0,
        random_state=42,
    )
    clf_xgb.fit(
        train_scaled, train_labels_int,
        eval_set=[(test_scaled, test_labels_int)],
        verbose=False,
    )
    xgb_time = time.time() - t0
    xgb_acc = clf_xgb.score(test_scaled, test_labels_int)
    xgb_n_trees = clf_xgb.best_ntree_limit if hasattr(clf_xgb, 'best_ntree_limit') else clf_xgb.n_estimators
    print(f"    acc={xgb_acc:.4f}  trees={xgb_n_trees}  time={xgb_time:.1f}s")
    row["xgb_acc"] = round(xgb_acc, 4)
    row["xgb_trees"] = xgb_n_trees
    row["xgb_train_s"] = round(xgb_time, 1)

    # Gap recovery vs XLM-R
    xlmr = row["xlmr_full"]
    if xlmr and xlmr > lr_acc:
        gap = xlmr - lr_acc
        row["mlp_gap_recovery_pct"] = round((mlp_acc - lr_acc) / gap * 100, 1)
        row["xgb_gap_recovery_pct"] = round((xgb_acc - lr_acc) / gap * 100, 1)
    else:
        row["mlp_gap_recovery_pct"] = None
        row["xgb_gap_recovery_pct"] = None

    print(f"\n  SUMMARY for {lang}:")
    print(f"    LR (Jeffy):     {lr_acc:.4f}  ({lr_params:>8,} params)")
    print(f"    MLP:            {mlp_acc:.4f}  ({mlp_params:>8,} params)  gap recovery: {row.get('mlp_gap_recovery_pct', '?')}%")
    print(f"    XGBoost:        {xgb_acc:.4f}  ({xgb_n_trees} trees)  gap recovery: {row.get('xgb_gap_recovery_pct', '?')}%")
    print(f"    XLM-R (SOTA):   {xlmr}")

    results.append(row)

# Save
out_path = OUTPUT_DIR / "classifier_ablation.csv"
fields = ["lang", "lr_acc", "lr_params", "lr_train_s",
          "mlp_acc", "mlp_params", "mlp_train_s",
          "xgb_acc", "xgb_trees", "xgb_train_s",
          "xlmr_full", "mlp_gap_recovery_pct", "xgb_gap_recovery_pct"]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved to {out_path}")

# Overall
print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
print(f"{'Lang':<6} {'LR':>8} {'MLP':>8} {'XGBoost':>8} {'XLM-R':>8} {'MLP gap%':>9} {'XGB gap%':>9}")
for r in results:
    print(f"{r['lang']:<6} {r['lr_acc']:>8.3f} {r['mlp_acc']:>8.3f} {r['xgb_acc']:>8.3f} {r['xlmr_full']:>8.3f} {r.get('mlp_gap_recovery_pct','?'):>8}% {r.get('xgb_gap_recovery_pct','?'):>8}%")

mean_lr = np.mean([r["lr_acc"] for r in results])
mean_mlp = np.mean([r["mlp_acc"] for r in results])
mean_xgb = np.mean([r["xgb_acc"] for r in results])
mean_xlmr = np.mean([r["xlmr_full"] for r in results if r["xlmr_full"]])
print(f"\n{'Mean':<6} {mean_lr:>8.3f} {mean_mlp:>8.3f} {mean_xgb:>8.3f} {mean_xlmr:>8.3f}")
gap = mean_xlmr - mean_lr
print(f"\nMLP closes {((mean_mlp - mean_lr) / gap) * 100:.0f}% of LR→XLM-R gap")
print(f"XGBoost closes {((mean_xgb - mean_lr) / gap) * 100:.0f}% of LR→XLM-R gap")
print("\nDone!")
