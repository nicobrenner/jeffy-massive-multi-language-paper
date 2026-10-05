# Frozen Multilingual Embeddings as Universal Intent Classifier Foundations

Research companion to [Jeffy](https://github.com/nicobrenner/jeffy) — exploring how far frozen multilingual sentence embeddings + per-language logistic regression can go for intent classification across 51 languages.

## Key results

- **75.6% mean accuracy** across 51 MASSIVE languages with ~100KB per classifier
- **Beats XLM-R zero-shot** (75.6% vs ~70.6%) while being 10,000× smaller per task
- **22 minutes total training** on commodity CPU (no GPU needed)
- **Cross-lingual agreement analysis**: novel evaluation using classifier disagreement as an encoder probe

## Structure

```
data/                   Experiment outputs (CSVs, matrices)
  training_results.csv  Per-language accuracy + XLM-R baselines
  agreement_matrix.csv  51×51 pairwise classifier agreement rates
  predictions.csv       Per-text predictions from all 51 classifiers
experiments/            Reproducible experiment scripts
  cross_lingual_agreement.py
results/                Analysis summaries and notes
  agreement_summary.md
  experiment_log.md
  xlmr_baseline_research.md
paper/                  Paper and blog outlines
figures/                Generated charts and visualizations
```

## Reproducing

Requires [Jeffy](https://github.com/nicobrenner/jeffy) installed with the MASSIVE language packs:

```bash
pip install jeffy-classify
```

Run the cross-lingual agreement experiment:

```bash
python experiments/cross_lingual_agreement.py
```

## Dataset

[Amazon MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) — 60 voice-command intents, 51 languages, CC BY 4.0.

## References

- FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages" (ACL 2023)
- Encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ languages)

## License

CC BY 4.0 (same as the underlying MASSIVE dataset)
