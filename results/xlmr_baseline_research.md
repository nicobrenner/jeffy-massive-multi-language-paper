# XLM-R Baseline Research for MASSIVE Paper

## Hardware
- **No GPU** (nvidia-smi not found, CUDA: False)
- PyTorch 2.14.1 (CPU), Transformers 5.18.0 installed
- XLM-R Base inference: ~4ms/sample on CPU (fast enough for evaluation, not training)

## Published MASSIVE Baselines (FitzGerald et al., ACL 2023)

Source: arxiv 2204.08582, Table 8 — Intent Accuracy (%)

### Aggregate numbers
| Model | Setup | Mean Accuracy |
|-------|-------|---------------|
| XLM-R Base | Full (all langs) | **85.1** |
| XLM-R Base | Zero-shot (en only) | **70.6** (est) |
| mT5 Enc | Full | **86.1** |
| mT5 T2T | Full | **85.3** |
| **Jeffy LR** | **Full** | **75.6** |

### Per-language: XLM-R Full vs Jeffy LR (selected)
| Language | XLM-R Full | Jeffy LR | Gap |
|----------|-----------|----------|-----|
| en | 88.3 | 86.4 | -1.9 |
| fr | 86.3 | 82.5 | -3.8 |
| sv | 87.9 | 81.1 | -6.8 |
| de | 85.7 | 74.5 | -11.2 |
| ja | 83.9 | 80.8 | -3.1 |
| ko | 86.5 | 74.1 | -12.4 |
| ar | 80.7 | 69.0 | -11.7 |
| cy | 82.6 | 61.2 | -21.4 |
| sw | 83.1 | 60.2 | -22.9 |
| jv | 82.9 | 59.7 | -23.2 |

### Full per-language table (all 51)
| Language | XLM-R Full | XLM-R Zero | Jeffy LR |
|----------|-----------|-----------|----------|
| en-US | 88.3 | — | 86.4 |
| sv-SE | 87.9 | 85.2 | 81.1 |
| nb-NO | 87.3 | 83.6 | 78.5 |
| da-DK | 86.9 | 83.1 | 80.0 |
| ro-RO | 86.9 | 80.8 | 79.6 |
| nl-NL | 86.8 | 82.1 | 81.0 |
| ru-RU | 87.2 | 81.3 | 81.5 |
| id-ID | 87.1 | 83.1 | 82.0 |
| fr-FR | 86.3 | 80.8 | 82.5 |
| it-IT | 86.6 | 76.4 | 80.6 |
| ms-MY | 86.1 | 76.7 | 79.0 |
| es-ES | 86.9 | 78.8 | 81.5 |
| pt-PT | 86.7 | 79.5 | 82.5 |
| fa-IR | 87.0 | 81.1 | 81.1 |
| pl-PL | 85.8 | 80.7 | 81.7 |
| de-DE | 85.7 | 77.6 | 74.5 |
| az-AZ | 86.2 | 70.9 | 72.0 |
| tr-TR | 86.3 | 78.4 | 80.3 |
| ko-KR | 86.5 | 77.0 | 74.1 |
| af-ZA | 85.6 | 71.7 | 71.0 |
| ml-IN | 85.1 | 70.1 | 73.1 |
| sq-AL | 86.4 | 67.6 | 79.2 |
| sl-SL | 86.3 | 69.5 | 79.1 |
| el-GR | 86.2 | 74.0 | 80.3 |
| vi-VN | 86.3 | 79.2 | 79.0 |
| hi-IN | 85.8 | 74.8 | 80.2 |
| hu-HU | 86.2 | 77.1 | 80.2 |
| is-IS | 85.3 | 66.7 | 63.4 |
| fi-FI | 85.5 | 80.2 | 78.1 |
| zh-CN | 84.9 | 61.9 | 82.2 |
| lv-LV | 86.1 | 69.2 | 80.0 |
| th-TH | 84.7 | 77.4 | 79.8 |
| tl-PH | 84.6 | 63.7 | 59.7 |
| mn-MN | 84.3 | 64.4 | 76.5 |
| kn-IN | 84.0 | 63.5 | 70.9 |
| te-IN | 84.5 | 68.2 | 71.4 |
| bn-BD | 84.1 | 66.0 | 67.5 |
| he-IL | 85.9 | 73.2 | 77.0 |
| my-MM | 83.6 | 67.6 | 75.0 |
| jv-ID | 82.9 | 46.5 | 59.7 |
| hy-AM | 84.4 | 71.6 | 76.5 |
| ta-IN | 83.5 | 68.1 | 69.3 |
| ur-PK | 83.2 | 65.6 | 78.0 |
| sw-KE | 83.1 | 46.6 | 60.2 |
| cy-GB | 82.6 | 46.9 | 61.2 |
| ja-JP | 83.9 | 44.8 | 80.8 |
| zh-TW | 83.0 | 60.4 | 78.5 |
| am-ET | 81.7 | 51.9 | 65.0 |
| ar-SA | 80.7 | 62.8 | 69.0 |
| ka-GE | 80.3 | 61.2 | 68.8 |
| km-KH | 77.2 | 61.3 | 66.8 |

## Feasibility Assessment

### What we CAN do on CPU
1. **Cite published baselines** — Table 8 from FitzGerald et al. gives per-language numbers for XLM-R Full, XLM-R Zero, mT5 T2T Full, mT5 Enc Full. This is the standard approach for papers comparing against established benchmarks.
2. **Run XLM-R zero-shot evaluation** — Inference only (no training). Load a pretrained XLM-R + classification head trained on English, evaluate on all 51 test sets. ~12 seconds for 2974 texts. But we'd need a pre-trained checkpoint with the classification head (e.g., `cartesinus/xlm-r-base-amazon-massive-intent` on HuggingFace).
3. **Encoder ablation** — Same LR pipeline with different frozen encoders (mpnet, E5, LaBSE). This is exactly our existing workflow, ~22 min per encoder. Very feasible.

### What we CANNOT do on CPU (practically)
1. **Fine-tune XLM-R** — 51 langs × 10k examples × 3 epochs ≈ 2+ hours even on GPU. On CPU it'd be 50-100+ hours. Impractical.
2. **Fine-tune mT5** — Even larger model, same problem.

## Recommendation

**Cite, don't reproduce.** The FitzGerald et al. baselines are:
- Published in a top venue (ACL 2023)
- On the exact same dataset and splits
- The standard reference everyone else cites

This is normal practice — papers comparing lightweight approaches to SOTA cite the published numbers rather than re-running expensive fine-tuning. Our paper's novelty isn't the baselines but: (a) the accuracy-size Pareto analysis, (b) the cross-lingual agreement methodology.

**One thing we CAN reproduce for extra credibility:** Use the pre-trained `cartesinus/xlm-r-base-amazon-massive-intent` checkpoint from HuggingFace to run evaluation on all 51 test sets ourselves. This gives us verified XLM-R numbers on the same data, and we can report both the published and our reproduced numbers.

**Key narrative angle:** Jeffy LR at 75.6% mean is ~10 points below XLM-R Full (85.1%), but:
- 10,000× smaller per task (~100KB vs ~1GB)
- Trains in 26 seconds per language vs hours with GPU
- No GPU needed at all
- **Jeffy BEATS XLM-R zero-shot** for many languages (mean ~70.6% vs 75.6%)

That last point is huge — our per-language LR with frozen embeddings outperforms XLM-R's zero-shot cross-lingual transfer for most languages, despite being 10,000× smaller.

## Priority experiments (updated with feasibility)

1. **Encoder ablation** (4 encoders, ~90 min total, CPU) — FEASIBLE NOW
2. **Cite XLM-R/mT5 baselines** from Table 8 — NO EXPERIMENT NEEDED
3. **Reproduce XLM-R eval** using HF checkpoint — FEASIBLE (~30 min)
4. **Few-shot learning curves** — FEASIBLE (subsample training data, re-run LR)
5. **Inference benchmarking** — FEASIBLE (just timing)
6. **Distillation** — Would need XLM-R soft labels, FEASIBLE via HF checkpoint inference

Sources:
- [MASSIVE paper (arxiv)](https://arxiv.org/abs/2204.08582)
- [MASSIVE paper (ACL Anthology)](https://aclanthology.org/2023.acl-long.235/)
- [XLM-R MASSIVE checkpoint](https://huggingface.co/cartesinus/xlm-r-base-amazon-massive-intent)
