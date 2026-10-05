# MASSIVE Intent Classification: All 51 Languages

Encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
Classifier: LogisticRegression (C=0.01, newton-cg)
Total training time: 21.9 minutes

| Language | Code | HF Config | Train Acc | Test Acc | Train N | Test N |
|----------|------|-----------|-----------|----------|---------|--------|
| English | en | en | 90.6% | 86.4% | 10000 | 2000 |
| French | fr | fr | 86.5% | 82.5% | 10000 | 2000 |
| Portuguese | pt | pt | 86.7% | 82.5% | 10000 | 2000 |
| Chinese (Simplified) | zh_cn | zh-CN | 86.8% | 82.2% | 10000 | 2000 |
| Indonesian | id | id | 86.3% | 82.0% | 10000 | 2000 |
| Polish | pl | pl | 86.0% | 81.7% | 10000 | 2000 |
| Russian | ru | ru | 86.1% | 81.5% | 10000 | 2000 |
| Spanish | es | es | 86.2% | 81.5% | 10000 | 2000 |
| Persian | fa | fa | 86.0% | 81.1% | 10000 | 2000 |
| Swedish | sv | sv | 86.4% | 81.1% | 10000 | 2000 |
| Dutch | nl | nl | 86.1% | 81.0% | 10000 | 2000 |
| Japanese | ja | ja | 86.1% | 80.8% | 10000 | 2000 |
| Italian | it | it | 86.2% | 80.6% | 10000 | 2000 |
| Turkish | tr | tr | 85.6% | 80.3% | 10000 | 2000 |
| Greek | el | el | 85.6% | 80.3% | 10000 | 2000 |
| Hindi | hi | hi | 85.6% | 80.2% | 10000 | 2000 |
| Hungarian | hu | hu | 85.8% | 80.2% | 10000 | 2000 |
| Latvian | lv | lv | 84.8% | 80.0% | 10000 | 2000 |
| Danish | da | da | 85.3% | 80.0% | 10000 | 2000 |
| Thai | th | th | 84.5% | 79.8% | 10000 | 2000 |
| Romanian | ro | ro | 84.3% | 79.6% | 10000 | 2000 |
| Albanian | sq | sq | 85.4% | 79.2% | 10000 | 2000 |
| Slovenian | sl | sl | 85.1% | 79.1% | 10000 | 2000 |
| Malay | ms | ms | 84.4% | 79.0% | 10000 | 2000 |
| Vietnamese | vi | vi | 83.6% | 79.0% | 10000 | 2000 |
| Chinese (Traditional) | zh_tw | zh-TW | 85.4% | 78.5% | 10000 | 2000 |
| Norwegian | nb | nb | 84.3% | 78.5% | 10000 | 2000 |
| Finnish | fi | fi | 83.9% | 78.1% | 10000 | 2000 |
| Urdu | ur | ur | 82.6% | 78.0% | 10000 | 2000 |
| Hebrew | he | he | 83.5% | 77.0% | 10000 | 2000 |
| Armenian | hy | hy | 80.9% | 76.5% | 10000 | 2000 |
| Mongolian | mn | mn | 81.0% | 76.5% | 10000 | 2000 |
| Burmese | my | my | 81.2% | 75.0% | 10000 | 2000 |
| German | de | de | 81.4% | 74.5% | 10000 | 2000 |
| Korean | ko | ko | 79.5% | 74.1% | 10000 | 2000 |
| Malayalam | ml | ml | 77.2% | 73.1% | 10000 | 2000 |
| Azerbaijani | az | az | 77.9% | 72.0% | 10000 | 2000 |
| Telugu | te | te | 76.9% | 71.4% | 10000 | 2000 |
| Afrikaans | af | af | 79.1% | 71.0% | 10000 | 2000 |
| Kannada | kn | kn | 76.7% | 70.9% | 10000 | 2000 |
| Tamil | ta | ta | 73.5% | 69.3% | 10000 | 2000 |
| Arabic | ar | ar | 75.9% | 69.0% | 10000 | 2000 |
| Georgian | ka | ka | 76.1% | 68.8% | 10000 | 2000 |
| Bengali | bn | bn | 72.1% | 67.5% | 10000 | 2000 |
| Khmer | km | km | 72.2% | 66.8% | 10000 | 2000 |
| Amharic | am | am | 72.8% | 65.0% | 10000 | 2000 |
| Icelandic | is | is | 69.5% | 63.4% | 10000 | 2000 |
| Welsh | cy | cy | 66.7% | 61.2% | 10000 | 2000 |
| Swahili | sw | sw | 66.6% | 60.2% | 10000 | 2000 |
| Filipino | tl | tl | 66.7% | 59.7% | 10000 | 2000 |
| Javanese | jv | jv | 67.0% | 59.7% | 10000 | 2000 |

## Stats

- Mean test accuracy: 75.6%
- Median test accuracy: 78.5%
- Min: 59.7%
- Max: 86.4%
- Std: 6.8%
