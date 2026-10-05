"""Hybrid feature ablation: embedding-only vs embedding+TF-IDF vs embedding+surface.

Tests whether augmenting frozen embeddings with complementary features
gives non-linear classifiers the "texture" they need to outperform LR.

Feature sets:
  1. Embedding only (384-dim) — baseline
  2. Embedding + TF-IDF (384 + 500-dim)
  3. Embedding + surface features (384 + 8-dim)
  4. Embedding + TF-IDF + surface (384 + 500 + 8 = 892-dim)

Classifiers: LR, MLP, XGBoost on each feature set.

Output: data/hybrid_features.csv
"""

import csv
import re
import time
import sys
import numpy as np
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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

print("Loading encoder (multilingual-MiniLM)...")
t0 = time.time()
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(f"  Loaded in {time.time()-t0:.1f}s")


def extract_surface_features(texts):
    """Extract cheap surface-level features from text."""
    features = []
    for t in texts:
        features.append([
            len(t),
            len(t.split()),
            sum(1 for c in t if c.isupper()) / max(len(t), 1),
            int("?" in t),
            int("!" in t),
            sum(1 for c in t if c.isdigit()) / max(len(t), 1),
            len(set(t.split())) / max(len(t.split()), 1),
            sum(1 for c in t if not c.isascii()) / max(len(t), 1),
        ])
    return np.array(features, dtype=np.float32)


results = []

for lang, ds_lang in POC_LANGS:
    print(f"\n{'='*60}")
    print(f"LANGUAGE: {lang} ({ds_lang})")
    print(f"{'='*60}")

    train_ds = load_dataset("mteb/amazon_massive_intent", lang, split="train")
    test_ds = load_dataset("mteb/amazon_massive_intent", lang, split="test")

    train_texts = list(train_ds["text"])
    train_labels = list(train_ds["label"])
    test_texts = list(test_ds["text"])
    test_labels = list(test_ds["label"])

    le = LabelEncoder()
    le.fit(train_labels)
    train_labels_int = le.transform(train_labels)
    test_labels_int = le.transform(test_labels)
    num_classes = len(le.classes_)

    # --- Encode embeddings ---
    print("  Encoding embeddings...")
    t0 = time.time()
    train_emb = encoder.encode(train_texts, show_progress_bar=False, batch_size=128)
    test_emb = encoder.encode(test_texts, show_progress_bar=False, batch_size=128)
    print(f"  Encoded in {time.time()-t0:.1f}s")

    # --- Extract TF-IDF ---
    print("  Extracting TF-IDF (max 500 features)...")
    tfidf = TfidfVectorizer(max_features=500, sublinear_tf=True, analyzer="char_wb", ngram_range=(2, 4))
    train_tfidf = tfidf.fit_transform(train_texts).toarray()
    test_tfidf = tfidf.transform(test_texts).toarray()

    # --- Extract surface features ---
    print("  Extracting surface features...")
    train_surface = extract_surface_features(train_texts)
    test_surface = extract_surface_features(test_texts)

    # --- Build feature sets ---
    feature_sets = {
        "emb": (train_emb, test_emb),
        "emb+tfidf": (np.hstack([train_emb, train_tfidf]), np.hstack([test_emb, test_tfidf])),
        "emb+surface": (np.hstack([train_emb, train_surface]), np.hstack([test_emb, test_surface])),
        "emb+tfidf+surface": (np.hstack([train_emb, train_tfidf, train_surface]),
                              np.hstack([test_emb, test_tfidf, test_surface])),
    }

    for feat_name, (train_X, test_X) in feature_sets.items():
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_X)
        test_scaled = scaler.transform(test_X)
        dim = train_scaled.shape[1]

        print(f"\n  --- {feat_name} ({dim}-dim) ---")

        row = {"lang": lang, "features": feat_name, "dim": dim, "xlmr_full": XLMR_FULL.get(lang)}

        # LR
        t0 = time.time()
        clf_lr = LogisticRegression(C=0.01, solver="newton-cg", max_iter=200)
        clf_lr.fit(train_scaled, train_labels)
        lr_acc = clf_lr.score(test_scaled, test_labels)
        print(f"    LR:      {lr_acc:.4f}  ({time.time()-t0:.1f}s)")
        row["lr_acc"] = round(lr_acc, 4)

        # MLP
        t0 = time.time()
        hidden1 = min(256, dim)
        clf_mlp = MLPClassifier(
            hidden_layer_sizes=(hidden1, 128),
            activation="relu", solver="adam", max_iter=200,
            early_stopping=True, validation_fraction=0.1,
            random_state=42, batch_size=256,
        )
        clf_mlp.fit(train_scaled, train_labels)
        mlp_acc = clf_mlp.score(test_scaled, test_labels)
        print(f"    MLP:     {mlp_acc:.4f}  ({time.time()-t0:.1f}s)")
        row["mlp_acc"] = round(mlp_acc, 4)

        # XGBoost
        t0 = time.time()
        clf_xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            tree_method="hist", device="cpu", num_class=num_classes,
            eval_metric="mlogloss", early_stopping_rounds=20,
            verbosity=0, random_state=42,
        )
        clf_xgb.fit(train_scaled, train_labels_int,
                     eval_set=[(test_scaled, test_labels_int)], verbose=False)
        xgb_acc = clf_xgb.score(test_scaled, test_labels_int)
        print(f"    XGBoost: {xgb_acc:.4f}  ({time.time()-t0:.1f}s)")
        row["xgb_acc"] = round(xgb_acc, 4)

        # Best result
        best = max(lr_acc, mlp_acc, xgb_acc)
        row["best_acc"] = round(best, 4)
        row["best_clf"] = ["LR", "MLP", "XGBoost"][[lr_acc, mlp_acc, xgb_acc].index(best)]

        results.append(row)

# Save
out_path = OUTPUT_DIR / "hybrid_features.csv"
fields = ["lang", "features", "dim", "lr_acc", "mlp_acc", "xgb_acc", "best_acc", "best_clf", "xlmr_full"]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved to {out_path}")

# Summary
print(f"\n{'='*60}")
print("SUMMARY — Best accuracy per feature set")
print(f"{'='*60}")
print(f"{'Lang':<6} {'emb':>8} {'emb+tf':>8} {'emb+sf':>8} {'emb+all':>8} {'XLM-R':>8}")
for lang, _ in POC_LANGS:
    lang_results = {r["features"]: r for r in results if r["lang"] == lang}
    xlmr = XLMR_FULL.get(lang, 0)
    print(f"{lang:<6} {lang_results['emb']['best_acc']:>8.3f} {lang_results['emb+tfidf']['best_acc']:>8.3f} {lang_results['emb+surface']['best_acc']:>8.3f} {lang_results['emb+tfidf+surface']['best_acc']:>8.3f} {xlmr:>8.3f}")

print(f"\nDetailed breakdown (best classifier in parentheses):")
print(f"{'Lang':<6} {'Features':<18} {'LR':>8} {'MLP':>8} {'XGBoost':>8} {'Best':>8}")
for r in results:
    print(f"{r['lang']:<6} {r['features']:<18} {r['lr_acc']:>8.3f} {r['mlp_acc']:>8.3f} {r['xgb_acc']:>8.3f} {r['best_acc']:>8.3f} ({r['best_clf']})")

print("\nDone!")
