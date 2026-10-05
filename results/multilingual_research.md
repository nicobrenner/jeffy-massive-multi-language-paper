# Multilingual Classifier Research: Prior Art & Paper Angles

## Prior Art

### The MASSIVE Dataset (FitzGerald et al., ACL 2023)
The Amazon MASSIVE dataset — "Multilingual Amazon SLURP for Slot-filling, Intent classification, and Virtual assistant Evaluation" — contains 1M parallel labeled utterances across 51 languages, 18 domains, 60 intents, and 55 slots. The official baselines use XLM-R and mT5:
- **XLM-R Base (full training):** ~88.3% intent accuracy (average across languages)
- **mT5 Base (full training):** 85–89% intent accuracy depending on configuration
- **XLM-R Base (zero-shot, English-only training):** ~70.6% intent accuracy across other languages

These are fine-tuned models with 270M–580M parameters. The paper's per-language results are in Table 8 of the ACL paper.

### Cross-Lingual Transfer with XLM-R
XLM-R (Conneau et al., 2020) is the dominant approach for cross-lingual NLU. XLM-R Large (~560M params) achieves SOTA on cross-lingual benchmarks (XTREME, XNLI). XeroAlign (Gritta & Iacobacci, 2021) improves zero-shot cross-lingual transfer by 5–10 points via alignment techniques. These approaches require fine-tuning the entire transformer — GPU-intensive, hundreds of MB per model.

### SetFit (Tunstall et al., 2022)
SetFit is the closest prior art to Jeffy's approach. It fine-tunes a Sentence Transformer via contrastive learning on few-shot text pairs, then trains a logistic regression head. With just 8 labeled examples per class, SetFit matches fine-tuning RoBERTa Large on full training sets. It supports multilingual models. Key difference from Jeffy: SetFit fine-tunes the encoder itself (contrastive step), while Jeffy uses frozen pretrained encoders.

### Logistic Regression on LLM Embeddings (Buckmann & Hill, Bank of England, 2024)
"Logistic Regression makes small LLMs strong and explainable 'tens-of-shot' classifiers" (arxiv 2408.03414). Penalized logistic regression on embeddings from a small LLM matches or beats GPT-4 in the "tens-of-shot" regime (30+ examples per class). The paper tests bge-large-en-v1.5 (the same encoder Jeffy uses for English). This validates the core architectural bet: frozen embeddings + simple classifier ≈ fine-tuned giant model. Not multilingual though — English only.

### Multilingual Sentence Embeddings + Classical Classifiers
Multiple 2024–2025 studies combine multilingual embeddings (multilingual-E5, LaBSE, paraphrase-multilingual-mpnet) with logistic regression, XGBoost, or KNN for multilingual classification. Generally within 2–5% of fine-tuned BERT accuracy at a fraction of the compute. A 2025 study on Turkish/English/Italian sentiment uses distiluse-base-multilingual-cased-v1 + LR. A 2024 ensemble approach uses multilingual E5 + LR/XGBoost.

### Edge/Efficient Multilingual NLP
"The Edge-First Feature Extractor" (SAC 2026) achieves 12.5× energy-delay-product reduction for multilingual NLP on edge devices using static and distilled features. "Tiny Models, Tough Limits" (2026) benchmarks small language models under mobile CPU constraints. Active research area, but focused on distilled LLMs, not the embedding+classifier architecture.

## How Jeffy Compares

| Aspect | Jeffy | XLM-R Fine-tuned | SetFit | LR + Embeddings (BoE) |
|--------|-------|-------------------|--------|----------------------|
| Encoder | Frozen multilingual-MiniLM (470MB, shared) | Fine-tuned per task (~1GB each) | Fine-tuned per task (~400MB each) | Frozen (English only) |
| Classifier | Logistic regression (few KB) | Transformer head (part of model) | Logistic regression head | Logistic regression |
| Per-language model size | ~100KB | ~1GB | ~400MB | N/A (English only) |
| Training | Minutes, CPU-only | Hours, GPU required | Minutes, GPU helps | Minutes, CPU |
| MASSIVE intent accuracy | 74–83% (varies by lang) | ~88% average | Not benchmarked on MASSIVE | Not tested multilingual |
| Multilingual | Yes (per-language classifiers, shared encoder) | Yes (one model, all languages) | Yes (but requires per-task fine-tuning of encoder) | No |
| Deployment | CPU, local, no internet | GPU recommended | CPU possible but heavier | CPU |

**Key tradeoff:** Jeffy sacrifices ~5–14% accuracy vs. SOTA fine-tuned XLM-R but delivers classifiers that are 10,000× smaller per task, train in minutes on CPU, and share one encoder across all languages. The frozen encoder means adding a new language/task requires only training a logistic regression — seconds to minutes.

## Potential Paper Angles

### 1. "Frozen Multilingual Embeddings as Universal Classifier Foundations"
**Thesis:** A single frozen multilingual sentence encoder + per-language logistic regression achieves practical accuracy (74–83%) on MASSIVE at 1/10,000th the per-task model size of fine-tuned approaches.

**Novel contribution:** Systematic evaluation of the accuracy-vs-size Pareto frontier for multilingual intent classification. No existing paper benchmarks this specific combination (frozen paraphrase-multilingual-MiniLM + LR) across all 51 MASSIVE languages.

**Experiments needed:**
- Train LR classifiers for all 51 MASSIVE languages (not just 6)
- Compare accuracy vs. XLM-R Base, XLM-R Large, mT5 at each language
- Plot accuracy vs. per-task model size across all approaches
- Measure inference latency (ms) and memory footprint per approach

### 2. "Cross-Lingual Classifier Agreement in Shared Embedding Spaces"
**Thesis:** When per-language classifiers share a multilingual encoder, feeding the same text to multiple language classifiers reveals how the shared embedding space encodes cross-lingual semantic similarity — and disagreements reveal encoder biases.

**Novel contribution:** Using classifier agreement/disagreement as a probe for multilingual encoder quality. This is genuinely novel — existing work evaluates encoders via downstream task accuracy, not via cross-classifier agreement patterns.

**Experiments needed:**
- Feed identical text to all 51 language classifiers, measure agreement rates
- Correlate agreement with language family distance (typological features)
- Identify systematic disagreement patterns (e.g., compound-word languages)
- Compare agreement patterns across different multilingual encoders

### 3. "The 80% Threshold: When Frozen Embeddings Are Good Enough"
**Thesis:** For practical deployment (edge, local, resource-constrained), frozen embeddings + LR cross a "good enough" threshold for many classification tasks, and the marginal accuracy gain from fine-tuning doesn't justify the 10,000× cost increase.

**Novel contribution:** Practical deployment-focused evaluation including total system cost (encoder download + classifier size + training time + inference latency) vs. accuracy, across multiple languages and tasks.

**Experiments needed:**
- Benchmark across multiple datasets (MASSIVE, multilingual sentiment, topic classification)
- Measure total deployment cost: download size, disk, RAM, training time, inference latency
- Define and evaluate "good enough" thresholds for different use cases
- Compare: frozen MiniLM, frozen mpnet, frozen E5, SetFit, fine-tuned XLM-R

### 4. "Scaling the Classifier Catalog: A Package Manager for Pretrained Classifiers"
**Thesis:** The "ollama for classifiers" model — a registry of tiny pretrained classifiers that share encoders — is a viable distribution model for NLP, analogous to how ollama distributes LLMs or how package managers distribute libraries.

**Novel contribution:** Systems paper on the architecture, catalog design, and deployment model. Less about ML novelty, more about engineering contribution. Could fit a demo/systems track (EMNLP Demo, ACL System Demonstrations).

## Suggested Experiments (Priority Order)

1. **Expand to all 51 MASSIVE languages** — strongest empirical contribution, straightforward, and produces the accuracy table that anchors any paper
2. **Head-to-head vs. XLM-R on MASSIVE** — use the official MASSIVE evaluation scripts to compare directly against published baselines
3. **Cross-encoder comparison** — test paraphrase-multilingual-MiniLM vs. multilingual-mpnet vs. multilingual-E5 vs. LaBSE as the frozen encoder, same LR setup
4. **Cross-language agreement matrix** — 51×51 matrix of classifier agreement rates, visualized as a heatmap clustered by language family
5. **Few-shot learning curves** — plot accuracy vs. training examples (10, 50, 100, 500, 1000, 5000, 10000) per language
6. **Inference benchmarking** — latency and memory on CPU (laptop, Raspberry Pi, phone) vs. fine-tuned models on GPU

**Most publishable combination:** Angles 1 + 2 together — the accuracy benchmark provides the empirical foundation, and the cross-lingual agreement analysis provides the novel analytical contribution. Target: EMNLP 2027 or ACL 2027 (submission deadlines typically April/June).

## Key References

- FitzGerald et al. (2023). "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset." ACL 2023. https://aclanthology.org/2023.acl-long.235
- Conneau et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale." (XLM-R) https://arxiv.org/abs/1911.02116
- Tunstall et al. (2022). "Efficient Few-Shot Learning Without Prompts." (SetFit) https://arxiv.org/abs/2209.11055
- Buckmann & Hill (2024). "Logistic Regression makes small LLMs strong and explainable 'tens-of-shot' classifiers." https://arxiv.org/abs/2408.03414
- Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." https://arxiv.org/abs/1908.10084
