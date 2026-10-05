# Cross-Lingual Classifier Agreement Analysis

Date: 2026-10-05
Test set: English MASSIVE test split (2974 examples, 59 intents)
Classifiers: 51 languages
Encoder: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)

## Key Findings

- **Pairwise agreement:** mean 0.721, median 0.752, std 0.110
- **Full agreement (all 51):** 348/2974 (11.7%)
- **Majority vote accuracy:** 72.7%

## Per-Language Accuracy on English Test Set

| Language | Accuracy |
|----------|----------|
| en | 74.3% |
| nl | 71.9% |
| pl | 71.9% |
| es | 71.6% |
| fr | 71.5% |
| sv | 71.4% |
| da | 71.4% |
| pt | 71.3% |
| hu | 71.2% |
| tr | 70.9% |
| id | 70.9% |
| nb | 70.8% |
| el | 70.7% |
| hi | 70.5% |
| ru | 70.3% |
| fi | 70.1% |
| zh_cn | 69.9% |
| ro | 69.9% |
| de | 69.8% |
| fa | 69.8% |
| it | 69.7% |
| ms | 69.7% |
| zh_tw | 69.6% |
| sq | 69.2% |
| sl | 69.1% |
| ja | 69.1% |
| vi | 68.8% |
| lv | 68.8% |
| ur | 68.8% |
| he | 68.7% |
| th | 68.6% |
| hy | 68.5% |
| my | 67.8% |
| mn | 67.2% |
| ar | 67.1% |
| ko | 65.9% |
| af | 64.1% |
| az | 63.3% |
| ka | 63.1% |
| ta | 61.4% |
| km | 61.2% |
| kn | 61.1% |
| ml | 60.8% |
| te | 58.5% |
| bn | 56.7% |
| is | 56.0% |
| am | 54.0% |
| tl | 53.0% |
| jv | 52.6% |
| sw | 42.5% |
| cy | 39.7% |

## Most Agreeing Pairs

| Lang A | Lang B | Agreement |
|--------|--------|----------|
| fr | nl | 0.8753 |
| fr | pt | 0.8705 |
| da | nb | 0.8695 |
| da | nl | 0.8692 |
| nl | sv | 0.8668 |
| es | pt | 0.8662 |
| en | fr | 0.8638 |
| da | sv | 0.8628 |
| en | nl | 0.8628 |
| fr | sv | 0.8611 |
| es | fr | 0.8608 |
| nb | nl | 0.8605 |
| da | fr | 0.8594 |
| es | nl | 0.8591 |
| fr | it | 0.8591 |
| en | tr | 0.8584 |
| hu | nl | 0.8584 |
| hu | tr | 0.8584 |
| nl | pt | 0.8584 |
| zh_cn | zh_tw | 0.8571 |

## Least Agreeing Pairs

| Lang A | Lang B | Agreement |
|--------|--------|----------|
| km | sw | 0.4533 |
| cy | hy | 0.4506 |
| az | cy | 0.4499 |
| kn | sw | 0.4459 |
| cy | my | 0.4442 |
| is | sw | 0.4438 |
| cy | km | 0.4425 |
| cy | ml | 0.4388 |
| cy | is | 0.4351 |
| cy | te | 0.4344 |
| am | sw | 0.4297 |
| cy | ta | 0.4237 |
| cy | jv | 0.4213 |
| cy | kn | 0.4169 |
| bn | sw | 0.4099 |
| am | cy | 0.4089 |
| cy | sw | 0.4069 |
| sw | tl | 0.4059 |
| bn | cy | 0.3934 |
| cy | tl | 0.3914 |

## Most Disputed Intents

| Intent | Disagreement Rate |
|--------|------------------|
| calendar_set | 189/209 (90%) |
| general_quirky | 167/169 (99%) |
| play_music | 157/176 (89%) |
| qa_factoid | 131/141 (93%) |
| calendar_query | 123/126 (98%) |
| weather_query | 120/156 (77%) |
| email_query | 119/119 (100%) |
| email_sendemail | 112/114 (98%) |
| social_post | 81/81 (100%) |
| datetime_query | 68/88 (77%) |
| cooking_recipe | 66/72 (92%) |
| play_podcasts | 58/63 (92%) |
| news_query | 54/124 (44%) |
| play_radio | 54/72 (75%) |
| qa_definition | 54/57 (95%) |
| lists_remove | 52/52 (100%) |
| calendar_remove | 51/67 (76%) |
| lists_query | 51/51 (100%) |
| transport_query | 48/51 (94%) |
| iot_hue_lightoff | 43/43 (100%) |
