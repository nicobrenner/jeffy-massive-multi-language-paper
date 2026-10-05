# Paper Outline

## Working Title

**"Frozen Multilingual Embeddings as Universal Intent Classifier Foundations: 51 Languages, 100KB Per Task"**

Alternative: *"How Far Can Frozen Embeddings Go? Lightweight Multilingual Intent Classification at 10,000× Compression"*

---

## Target Venue

**Primary:** EMNLP 2027 (Empirical Methods in Natural Language Processing) — strong fit for empirical benchmarking + analysis paper. Submission deadline typically ~June 2027.

**Alternatives:**
- ACL 2027 (main conference or Findings) — if the cross-lingual analysis angle is strong enough
- EACL 2027 — European venue, good fit for multilingual work
- ACL System Demonstrations track — if we lean into the Jeffy tooling/catalog angle
- NAACL 2027 — if EMNLP timeline doesn't work

---

## Contributions

1. **First systematic benchmark** of frozen multilingual sentence embeddings + per-language logistic regression across all 51 MASSIVE languages, establishing the accuracy–size Pareto frontier against fine-tuned baselines (XLM-R, mT5).
2. **Cross-lingual classifier agreement analysis** — a novel evaluation methodology that uses per-language classifier agreement/disagreement patterns as a probe for multilingual encoder quality, revealing encoder biases invisible to standard per-language accuracy metrics.
3. **Practical deployment analysis** showing that the frozen-embedding approach achieves 75.6% mean accuracy at ~100KB per task (vs. ~1GB for fine-tuned models), trainable in 22 minutes for all 51 languages on commodity CPU hardware with no GPU.
4. *(If distillation experiment works)* **Knowledge distillation from SOTA models** into the lightweight architecture, measuring how much of the accuracy gap can be closed while preserving the size/speed advantage.

---

## Abstract Sketch (3–4 sentences)

We evaluate a lightweight approach to multilingual intent classification: frozen multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2, 384-dim) paired with per-language logistic regression classifiers, benchmarked across all 51 languages in the Amazon MASSIVE dataset. Our classifiers achieve a mean test accuracy of 75.6% (median 78.5%) at ~100KB per language — approximately 10,000× smaller than fine-tuned XLM-R baselines (~88% accuracy, ~1GB per model). We introduce cross-lingual classifier agreement analysis, a novel evaluation methodology that feeds identical inputs to all 51 per-language classifiers, revealing systematic patterns in how shared embedding spaces encode (and fail to encode) cross-lingual semantic equivalence. Our analysis shows that accuracy strongly correlates with encoder pretraining data availability, that classifier disagreement clusters by language family, and that [distillation result if applicable].

---

## 1. Introduction

- The multilingual NLP gap: most deployed classifiers are English-only; adding a new language typically requires fine-tuning a large model (GPU, hours, expertise)
- The practical need: real users want local, CPU-only, fast classification in their language (cite the motivating use case — a Polish developer building a personal assistant)
- The hypothesis: modern multilingual sentence encoders already encode cross-lingual semantic similarity; a simple linear classifier on top should be sufficient for many practical tasks
- Our approach: one frozen multilingual encoder shared across all languages + one logistic regression per language (~100KB each)
- Preview of results: 51 languages in 22 minutes on CPU, 75.6% mean accuracy, and a novel analysis method using cross-lingual agreement
- Paper structure overview

> **Figure 1:** Architecture diagram — one shared frozen encoder feeding 51 independent per-language LR classifiers. Contrast with the fine-tuned approach (one large model per language or one multilingual model).

---

## 2. Related Work

### 2.1 Multilingual Intent Classification
- MASSIVE dataset and official baselines: XLM-R Base ~88.3%, mT5 Base 85–89% (FitzGerald et al., ACL 2023)
- XLM-R and cross-lingual transfer (Conneau et al., 2020); XeroAlign improvements (Gritta & Iacobacci, 2021)
- Zero-shot cross-lingual transfer: training on English only, evaluating on other languages — XLM-R zero-shot ~70.6% on MASSIVE

### 2.2 Sentence Embeddings for Classification
- Sentence-BERT and sentence-transformers (Reimers & Gurevych, 2019)
- SetFit: few-shot classification via contrastive fine-tuning of sentence encoders + LR head (Tunstall et al., 2022) — differs from our work in that SetFit fine-tunes the encoder
- Buckmann & Hill (2024): LR on frozen LLM embeddings matches GPT-4 in tens-of-shot regime — validates the core architecture but English-only

### 2.3 Efficient and Edge NLP
- Edge-first feature extraction (SAC 2026)
- Tiny models under mobile constraints (2026)
- Multilingual embeddings + classical classifiers: recent 2024–2025 studies on Turkish/English/Italian, multilingual E5 + LR/XGBoost ensembles

### 2.4 Cross-Lingual Evaluation Methods
- Standard approach: per-language accuracy on held-out test sets
- Embedding space analysis: probing tasks, representational similarity analysis
- **Gap we fill:** no prior work uses per-language classifier agreement as a probe for encoder quality

---

## 3. Method

### 3.1 Architecture
- Encoder: `paraphrase-multilingual-MiniLM-L12-v2` — 12-layer, 384-dim, 50+ languages, 118M params, 470MB
- Preprocessing: StandardScaler on embeddings (per-language, fit on training data)
- Classifier: L2-regularized logistic regression (C=0.01, newton-cg solver)
- One classifier per language, all sharing the same frozen encoder
- Per-task model size: ~100KB (scaler parameters + LR coefficients + intercepts for 60 classes)

> **Table 1:** Architecture comparison — parameters, per-task disk size, training hardware, training time per language. Our approach vs. XLM-R fine-tuned, mT5 fine-tuned, SetFit, zero-shot XLM-R.

### 3.2 Dataset
- Amazon MASSIVE: 1M+ utterances, 60 intents, 51 languages, CC BY 4.0
- Training: 10,000 examples per language (capped); test: held-out split per language
- Same label space across all languages — enables direct cross-lingual comparison

### 3.3 Cross-Lingual Agreement Analysis
- Definition: given input text t, feed t to all 51 per-language classifiers; measure agreement rate (fraction predicting the same label)
- Agreement matrix: for each language pair (i, j), compute agreement rate over a shared test set
- Hypothesis: agreement correlates with (a) language family proximity and (b) encoder representation quality for those languages
- Disagreement analysis: systematic patterns in which intents cause disagreement, whether disagreement is random or structured

### 3.4 Baseline Comparisons
- **[EXPERIMENT NEEDED]** XLM-R Base fine-tuned on MASSIVE (reproduce official baselines)
- **[EXPERIMENT NEEDED]** XLM-R zero-shot (English training only)
- **[EXPERIMENT NEEDED]** SetFit with multilingual-MiniLM (same encoder, contrastive fine-tuning)

### 3.5 Encoder Ablation
- **[EXPERIMENT NEEDED]** Compare frozen encoders: paraphrase-multilingual-MiniLM-L12-v2, paraphrase-multilingual-mpnet-base-v2, multilingual-E5-large, LaBSE
- Same LR setup, same data, different encoders → isolate encoder contribution

### 3.6 Knowledge Distillation (Optional Section)
- **[EXPERIMENT NEEDED]** Train LR on soft labels from a SOTA model (XLM-R fine-tuned or GPT-4) instead of hard MASSIVE labels
- Hypothesis: soft labels transfer inter-class structure, closing part of the accuracy gap while preserving model size

---

## 4. Experiments

### 4.1 Main Results: 51-Language Benchmark

**Data we have:**

> **Table 2 (full page):** Test accuracy for all 51 languages, sorted by accuracy. Columns: Language, ISO code, Language family, Script, Test accuracy (ours), Train accuracy (ours), [XLM-R fine-tuned], [XLM-R zero-shot]. The last two columns are **[EXPERIMENT NEEDED]**.

> **Figure 2:** Bar chart of test accuracy across all 51 languages, ordered by accuracy, with XLM-R baseline as horizontal reference line. Color-coded by language family.

- Summary statistics: mean 75.6%, median 78.5%, std 6.8%, min 59.7% (Javanese), max 86.4% (English)
- 19 languages above 80%, 41 above 70%, 10 below 70%
- Training time: 22 minutes total on CPU for all 51 languages (~26 seconds per language)

### 4.2 Accuracy vs. Model Size

> **Figure 3:** Scatter plot — x-axis: per-task model size (log scale), y-axis: mean accuracy across 51 languages. Points: our LR classifiers (~100KB), XLM-R Base (~1GB), XLM-R Large (~2.2GB), mT5 Base (~1.2GB). Annotated with training time and hardware requirements.

- The Pareto frontier: where does the accuracy-size tradeoff bend?
- At what accuracy threshold does the 10,000× size reduction justify the accuracy drop?

### 4.3 Cross-Lingual Agreement Analysis

**[EXPERIMENT NEEDED — but mechanically straightforward with existing infrastructure]**

> **Figure 4 (key figure):** 51×51 heatmap of pairwise classifier agreement rates, clustered by language family. Dendrograms on both axes.

> **Figure 5:** Agreement rate vs. typological distance (from URIEL/lang2vec features). Scatter plot with regression line.

- Do language families cluster in agreement space?
- Which intents produce the most disagreement? Why?
- Does agreement predict accuracy? (If language A and B agree often, and A is high-accuracy, does that predict B's accuracy?)

### 4.4 Encoder Ablation

**[EXPERIMENT NEEDED]**

> **Table 3:** Mean accuracy across 51 languages for each frozen encoder. Columns: Encoder, Dimensions, Parameters, Disk size, Mean accuracy, Median accuracy, Training time.

- Does a larger encoder (multilingual-mpnet, 768-dim) close the gap significantly?
- Is there a sweet spot on the encoder-size–accuracy curve?

### 4.5 Few-Shot Learning Curves

**[EXPERIMENT NEEDED]**

> **Figure 6:** Accuracy vs. number of training examples (10, 50, 100, 500, 1000, 5000, 10000) for a representative subset of languages (high/mid/low resource). Show where diminishing returns kick in.

- How many examples does LR + frozen embeddings actually need?
- Comparison: SetFit learning curve on same data

### 4.6 Inference Benchmarking

**[EXPERIMENT NEEDED]**

> **Table 4:** Inference latency (ms) and peak memory (MB) on: laptop CPU (x86), Raspberry Pi 4 (ARM), [mobile device if feasible]. Compare: our approach (encoder + LR), XLM-R fine-tuned, distilbert.

- Encoder inference dominates (~200ms); LR is negligible (~0.1ms)
- Total deployment size: 470MB encoder (shared) + ~100KB per language

### 4.7 Knowledge Distillation

**[EXPERIMENT NEEDED — mark as optional / future work if results aren't ready]**

> **Table 5:** Accuracy comparison — hard labels vs. soft labels from XLM-R teacher vs. soft labels from GPT-4 teacher. Same LR architecture.

- Does distillation close the gap from 75.6% mean toward the ~88% XLM-R baseline?
- Which languages benefit most from distillation?

---

## 5. Analysis

### 5.1 What Determines Accuracy?
- Correlation between per-language accuracy and encoder pretraining data (measured by language representation in training corpora)
- Script effects: languages with unique scripts (Khmer, Amharic, Georgian) tend to score lower
- Morphological complexity: German's compound words, Finnish's agglutination, Arabic's morphology
- Language family analysis: Romance languages cluster high, Austronesian cluster low

> **Figure 7:** Scatter plot — accuracy vs. estimated encoder pretraining data size per language (from published corpus statistics). Annotated outliers.

### 5.2 Cross-Lingual Agreement Patterns
- What the agreement heatmap reveals about encoder geometry
- Surprising agreements (e.g., do Persian and Hindi agree more than expected from family distance?)
- Systematic disagreements: which intent pairs are confused across language boundaries?

### 5.3 The "Good Enough" Threshold
- For which use cases is 75% accuracy sufficient? (Routing, triage, pre-filtering)
- Comparison with zero-shot XLM-R (~70.6%) — our per-language LR may actually beat zero-shot transfer
- The deployment advantage: adding a new language takes 26 seconds, no GPU

### 5.4 Failure Modes
- Languages below 65% (Javanese, Filipino, Welsh, Swahili, Icelandic, Amharic): what goes wrong?
- Per-class accuracy breakdown: are certain intents universally hard?
- Confusion matrix analysis for lowest-performing languages

---

## 6. Discussion

- **The frozen-embedding bet:** when does it work and when doesn't it? The encoder is the bottleneck, not the classifier.
- **Implications for practitioners:** if you need >85% accuracy, fine-tune. If you need 51 languages on a laptop in an afternoon, freeze and classify.
- **Implications for encoder development:** cross-lingual agreement as a new evaluation signal — encoder developers could use classifier agreement matrices to diagnose representation weaknesses.
- **Limitations:**
  - Single task (intent classification); results may not generalize to sentiment, NLI, etc.
  - Single encoder; limited encoder ablation
  - LR may underfit for complex decision boundaries
  - No per-class analysis of MASSIVE (some intents may be trivially separable)
- **Ethical considerations:** low-resource language classifiers at lower accuracy could be misapplied; accuracy disparities across languages mirror representation disparities in training data

---

## 7. Conclusion

- Frozen multilingual embeddings + per-language logistic regression is a viable, practical approach for multilingual intent classification
- 51 languages, 22 minutes, ~100KB per task, no GPU — a fundamentally different tradeoff from fine-tuning
- Cross-lingual agreement analysis opens a new evaluation methodology for multilingual encoders
- Future work: distillation from SOTA models, extending to other tasks (sentiment, NER), encoder-specific agreement benchmarks

---

## Appendix

- **A:** Full 51-language results table with per-language detail (already have this data)
- **B:** Hyperparameter sensitivity analysis (C values, solver choice, scaler variants)
- **C:** Complete cross-lingual agreement matrix (51×51)
- **D:** Per-intent accuracy breakdown for selected languages

---

## Figures and Tables Summary

| # | Type | Description | Status |
|---|------|-------------|--------|
| Fig 1 | Diagram | Architecture comparison (ours vs. fine-tuned) | TO DO |
| Tab 1 | Table | Architecture comparison (size, speed, hardware) | TO DO (partial data) |
| Tab 2 | Table | Full 51-language results | **HAVE DATA** |
| Fig 2 | Bar chart | Accuracy across 51 languages, color by family | TO DO (have data) |
| Fig 3 | Scatter | Accuracy vs. model size (Pareto frontier) | TO DO (need baselines) |
| Fig 4 | Heatmap | 51×51 cross-lingual agreement matrix | **NEED EXPERIMENT** |
| Fig 5 | Scatter | Agreement vs. typological distance | **NEED EXPERIMENT** |
| Tab 3 | Table | Encoder ablation results | **NEED EXPERIMENT** |
| Fig 6 | Line chart | Few-shot learning curves | **NEED EXPERIMENT** |
| Tab 4 | Table | Inference latency benchmarks | **NEED EXPERIMENT** |
| Tab 5 | Table | Distillation results | **NEED EXPERIMENT** |
| Fig 7 | Scatter | Accuracy vs. pretraining data per language | TO DO (need corpus stats) |

---

## Experiment Priority Order

1. **Cross-lingual agreement matrix** (51×51) — this is the novel analytical contribution; must have
2. **XLM-R baselines on MASSIVE** — needed for any credible comparison
3. **Encoder ablation** (4 encoders) — shows the approach isn't encoder-specific
4. **Few-shot learning curves** — practical guidance for users with limited data
5. **Inference benchmarking** — supports the deployment story
6. **Distillation** — nice to have, could be future work if time-constrained
