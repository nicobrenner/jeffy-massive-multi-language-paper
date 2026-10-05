# Blog Post Outline: Multilingual Classifiers with Jeffy

## Title options (pick one, riff on it)
- "Adding 51 Languages to a Classifier in an Afternoon"
- "74–83% Accuracy, 100KB Per Language, No GPU"
- "A Hacker News Comment Made Me Train 51 Classifiers"
- "Frozen Embeddings, Tiny Classifiers, 51 Languages"

---

## 1. The Spark — A Polish User's Problem
- HN commenter building a Polish-language personal assistant
- Needed to classify messages into 20+ categories
- Every off-the-shelf classifier failed with Polish; one returned "none" for everything
- Fell back to vector search at 60% accuracy
- Wanted: local, CPU-only, under 1 second
- This is a common problem: most NLP classifiers are English-only

> [No graphic needed — keep it personal and quick. Maybe a screenshot of the HN comment if appropriate.]

---

## 2. The Bet — What If the Encoder Already Knows?
- Jeffy's English classifiers use BAAI/bge-large-en-v1.5 (1024-dim, English-specialized)
- But multilingual sentence encoders exist: paraphrase-multilingual-MiniLM-L12-v2
  - 384 dimensions, 50+ languages, 470MB
  - Maps "Turn on the lights" and "Enciende las luces" to nearby points
- The hypothesis: if the encoder already understands meaning across languages, a simple logistic regression on top should work for any language
- No translation needed. No multilingual training tricks. Just swap the encoder.

> [CODE SNIPPET: Show how simple the architecture is — encode → scale → predict]
> ```python
> from sentence_transformers import SentenceTransformer
> encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
> embedding = encoder.encode("Enciende las luces del salón")
> # -> 384-dim vector, same space as English
> prediction = classifier.predict(scaler.transform([embedding]))
> ```

---

## 3. The Dataset — Amazon MASSIVE
- 1M+ labeled utterances, 60 voice-command intents, 51 languages
- Same labels across all languages (alarm_set, weather_query, play_music, etc.)
- Already on HuggingFace: `mteb/amazon_massive_intent`
- Perfect for this experiment: same task, different languages, ready to go

> [GRAPHIC: Small table or visual showing a few example intents with the same phrase in 4-5 languages]

---

## 4. Training — 15 Minutes, 6 Languages, One Laptop
- For each language: load dataset → encode with multilingual encoder → train logistic regression → save
- Each classifier: a few KB of weights (coefficients + intercept + scaler)
- Total training time: ~15 minutes on CPU for 6 languages
- No GPU. No cloud. No fine-tuning.

> [CODE SNIPPET: The actual training loop, simplified]
> ```python
> for lang in ["es", "pl", "pt", "zh-CN", "de", "ja"]:
>     texts, labels = load_massive(lang)
>     embeddings = encoder.encode(texts)
>     scaler = StandardScaler().fit(embeddings)
>     clf = LogisticRegression(max_iter=1000)
>     clf.fit(scaler.transform(embeddings), labels)
> ```

---

## 5. Results — The Accuracy Table
- Present the numbers honestly: 74–83% vs. fine-tuned XLM-R at ~88%

> [GRAPHIC: Bar chart — accuracy by language, with XLM-R baseline as a reference line]

| Language   | Test Accuracy | XLM-R Baseline |
|------------|--------------|----------------|
| English*   | 88.1%        | ~88.3%         |
| Portuguese | 82.5%        | —              |
| Chinese    | 82.2%        | —              |
| Polish     | 81.7%        | —              |
| Spanish    | 81.5%        | —              |
| Japanese   | 80.8%        | —              |
| German     | 74.5%        | —              |

*English uses a different, larger encoder (bge-large-en-v1.5, 1024-dim)

- The tradeoff: ~5-14% less accuracy, but 10,000× smaller per task
- German is lowest — compound words may be the cause (worth investigating)
- English outperforms because it has a specialized larger encoder

> [GRAPHIC: Scatter plot — accuracy vs. per-task model size. Jeffy classifiers as tiny dots in the lower-left, XLM-R/mT5 as big dots in the upper-right. The gap is visible but the size difference is dramatic.]

---

## 6. The Surprising Part — Cross-Language Agreement
- Feed the same text to all 6 classifiers
- They mostly agree on the intent, even though each was trained independently
- "spiel es nochmal ab bitte" (German) → play_music across all classifiers
- Cosine similarity between same intent across languages: 0.83–0.92
- Different intents in the same language: 0.06
- The encoder does the heavy lifting; the classifiers just draw decision boundaries

> [GRAPHIC: Heatmap — cross-language cosine similarity for a few intents. Show that same-intent-different-language is high, different-intent-same-language is low.]

> [INTERACTIVE: Link to the live playground where you can try this yourself — type a phrase, hit "All 6 classifiers"]

---

## 7. Aside — When AI Writes About Your Project
- An AI-generated news site (openai-hub.com) covered Jeffy
- It hallucinated that Jeffy uses "fine-tuned Qwen3.5 and Gemma 4 models"
- Jeffy uses logistic regression. The entire classifier is a matrix multiply.
- The irony: an AI confidently describing an AI project and getting it completely wrong
- This is a good reminder about the state of AI-generated content

> [SCREENSHOT: The hallucinated article, with the wrong claims highlighted]

---

## 8. What's Next — 51 Languages and a Paper
- Expanding from 6 to all 51 MASSIVE languages
- No existing paper benchmarks frozen multilingual embeddings + per-language LR at this scale
- Planned experiments:
  - All 51 languages accuracy table
  - Head-to-head vs. XLM-R on MASSIVE official benchmarks
  - Cross-encoder comparison (MiniLM vs. mpnet vs. E5 vs. LaBSE)
  - Cross-language agreement matrix (51×51 heatmap)
  - Few-shot learning curves (how many examples do you actually need?)
  - Distillation from SOTA models — can a larger model teach a tiny one?
- Target: an actual research paper

> [GRAPHIC: Preview of the 51×51 language agreement heatmap (once we have the data)]

---

## 9. Try It
- `pip install jeffy-classify`
- Link to the live playground
- Link to the multilingual demo specifically
- The code is on GitHub

> [CODE SNIPPET: Minimal usage]
> ```python
> from jeffy import Engine
> engine = Engine()
> engine.load()
> result = engine.predict("massive_intent_es", "enciende las luces del salón")
> print(result["label"])  # iot_hue_lighton
> ```

---

## Graphics/Assets Checklist
1. [ ] Accuracy bar chart (6 languages + English, with XLM-R reference line)
2. [ ] Accuracy vs. model size scatter plot (Jeffy vs. XLM-R/mT5)
3. [ ] Cross-language cosine similarity heatmap
4. [ ] Example intents table (same phrase in multiple languages)
5. [ ] Screenshot of the AI-hallucinated article
6. [ ] Architecture diagram: encoder → scaler → LR → probabilities (optional, keep simple)
7. [ ] Link to live playground multilingual demo

## Tone Notes
- Technical but conversational. Write like you're explaining to a smart friend.
- Be honest about limitations (accuracy gap, German underperformance)
- Let the numbers speak — don't oversell
- The story arc: user request → "can we do this?" → yes, and here's what we learned
- End with an open question (research direction) not a sales pitch
