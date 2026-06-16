# 🧠 SentimentIQ — Amazon Review Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org) [![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io) [![Scikit-learn](https://img.shields.io/badge/Model-Scikit--learn-f7931e?style=flat-square&logo=scikitlearn)](https://scikit-learn.org) [![Live Demo](https://img.shields.io/badge/Live-Demo-2ea44f?style=flat-square)](https://sentiment-analysisapp-qqzxw6exrhfvsby5ygndf2.streamlit.app/)

🌐 **Live Demo:** https://sentiment-analysisapp-qqzxw6exrhfvsby5ygndf2.streamlit.app/

### Snippet of the app

<img width="1350" height="563" alt="image" src="https://github.com/user-attachments/assets/1436ead3-e949-49f1-a736-0249c94ed81c" />

## The Problem

E-commerce platforms collect thousands of reviews a day. Somewhere in that pile is the early signal of a defective product batch, a shipping problem, or a feature customers hate — but nobody can read 4,000 reviews a day to find it.

The usual fix people reach for is "just throw a transformer at it." BERT, fine-tune it, deploy it, done. That works, but it's also slower to train, harder to deploy cheaply, and overkill for a binary positive/negative classification task that doesn't actually need deep contextual language understanding to solve well.

This project asks a more useful question: **how far can disciplined preprocessing and a classical ML model get you, and where exactly does it break?**

---

## What It Does

The app takes an Amazon-style product review — or a whole CSV of them — and returns a sentiment prediction (Positive / Negative) with a confidence score, in real time, through a deployed web interface. No GPU required, no API call to an LLM, sub-second inference.

```
User Review (single text OR batch CSV)
    ↓
Text Preprocessing (lowercase, strip HTML, remove noise)
    ↓
TF-IDF Vectorization (uni-grams + bi-grams, max 1000 features)
    ↓
Logistic Regression
    ↓
Sentiment + Confidence Score + Keyword Signal Breakdown
```

It started as a model comparison exercise. It's now closer to a small but genuinely usable review-triage tool — the kind you could actually hand to someone monitoring product feedback and have them get value from on day one.

---

## The Approach

**Data.** Amazon Product Reviews dataset, binary sentiment labels. Review headline and body were merged into a single text field — headlines alone are often too short to carry sentiment signal reliably, and combining them gave the model more context per example.

**Cleaning.** Lowercasing, HTML tag removal, special character and number stripping, whitespace normalization. Standard, but skipping any one of these steps measurably hurt model performance during testing — TF-IDF is sensitive to vocabulary noise in a way deep models are more robust to.

**Feature engineering.** TF-IDF with uni-grams and bi-grams, capped at 1000 features. The bi-grams matter more than they sound — "not good" and "very good" need to be distinguishable, and a unigram-only model conflates them since both contain "good."

**Class imbalance.** The dataset skewed positive, as most review datasets do. Random oversampling brought the minority class up, which improved recall on negative reviews — the ones that actually matter for catching problems early — at a modest cost of slight overfitting risk, which was monitored via cross-validation.

**Model selection.** Three models were compared head-to-head: Naive Bayes, Logistic Regression, and Linear SVM, tuned via GridSearchCV with 5-fold cross-validation, optimizing for F1-score (chosen over accuracy because of the class imbalance).

**Winner: Logistic Regression.** Not because it's the most powerful model in the comparison — Linear SVM was close — but because it gave the best balance of accuracy, stability across folds, and interpretability. On high-dimensional sparse TF-IDF features, logistic regression's linear decision boundary is enough; the extra complexity of more powerful models bought essentially nothing here.

---

## Where It Breaks

A model is only as trustworthy as your honesty about its limits. Three failure modes showed up clearly in testing:

**Sarcasm and irony.** *"Oh great, another product that stopped working in 2 days."* Every keyword in that sentence reads positive. The model has no mechanism to detect tonal inversion — it's pattern-matching on words, not reasoning about meaning. This is the single biggest gap between this approach and a transformer model.

**Mixed sentiment.** *"The product quality is excellent, but delivery was terrible."* Real reviews are rarely purely one sentiment. A binary classifier forces a single label onto a review that genuinely contains both, and which way it falls often comes down to which clause has more TF-IDF weight rather than which sentiment the reviewer meant to emphasize.

**Domain shift.** The model was trained on Amazon product reviews specifically. Vocabulary, review length, and tone on Amazon don't transfer cleanly to Twitter sentiment, app store reviews, or live customer support chat — all of which have different baseline sentiment vocabularies and informal slang the model has never seen.

None of that gets fixed by a nicer interface. But it does become visible — when the model gets a sarcastic or mixed review wrong, you can usually watch it happen in the keyword highlighting: see it latch onto "excellent" and miss the sarcasm entirely, or weigh one clause more heavily than the other. A wrong answer you can see the reasoning behind is a fundamentally different thing than a wrong answer you just have to take on faith.

---

## What Using It Actually Looks Like

Paste in a review — or click one of the three pre-filled examples (clearly positive, clearly negative, and a deliberately mixed one) — and the model returns a label almost instantly. But the label was never the interesting part of this update. What changed is everything around it.

Instead of a flat confidence percentage, you get an animated bar that fills green or red as it loads — something you read in half a second rather than parsing a number. Underneath it, the review text itself lights up: words like `excellent` or `terrible` get highlighted right where they appear, so you're not just told the verdict, you can see the exact evidence the model used to reach it. That single change is what turns this from a black box into something you can actually audit — if the model gets a review wrong, you can usually tell *why* just by looking at what it latched onto.

Below that sits a quieter layer of detail: a word count, sentence count, and a chip-style list of the most prominent non-stopword terms in the review, useful for skimming something long without reading every line. None of this changes the prediction. It changes whether you trust it.

The real shift, though, is what happens once you stop testing one review at a time. Upload a CSV with a `review` column and the app classifies every row in one pass — positive count, negative count, overall positive rate, and a results file you can download and hand to someone else. That's the difference between a demo and a tool: a single-review classifier proves the model works, but a batch processor that takes a day's worth of customer feedback and returns a triaged spreadsheet is something a small team could actually plug into how they already work, without writing a line of code.

Everything you run in single-review mode also gets logged to a sidebar history with colour-coded badges, so a session builds up a visible trail rather than each review disappearing the moment you move to the next one. And a settings panel lets you collapse any of this away — the confidence bar, the keywords, the stats — so the interface stays clean by default but opens up for anyone who wants more.

---

## Why This Tradeoff Was Worth Making

Transformer models would likely close all three gaps above to some degree. But they come with real costs: heavier infrastructure, slower inference, harder interpretability, and meaningfully more expensive to train and serve at scale.

For a use case like "flag negative reviews for a human to review," the classical pipeline here gets you to a genuinely useful tool — fast, cheap, explainable, and good enough — and the honest documentation of where it fails is itself valuable: it tells you exactly when you'd need to upgrade to something heavier, instead of pretending the simple model handles everything.

---

## Evaluation

Performance was assessed on Accuracy, Precision, Recall, F1-score, and Confusion Matrix, with F1 prioritized given the class imbalance in the underlying dataset.

---

## What's Next

The app is now usable for real review triage. What would still move it further toward production:

- **Negation-aware preprocessing** — explicitly tagging negation scope so "not good" carries distinguishable signal from "good," rather than relying on bi-grams to catch it implicitly
- **Threshold tuning** — moving off the default 0.5 cutoff to a threshold calibrated for the actual business cost of false negatives vs. false positives (missing a real complaint is usually worse than a false alarm)
- **Transformer upgrade path** — DistilBERT as a drop-in replacement for the cases that matter most: sarcasm and mixed-sentiment reviews specifically, potentially as a second-stage model that only runs on reviews the classical model flags as low-confidence
- **Class weighting over oversampling** — testing whether class-weighted loss reduces the overfitting risk that oversampling introduced, without sacrificing the recall gains
- **Persistent batch history** — current analysis history lives in session state and resets on refresh; moving it to a lightweight database would let teams track sentiment trends across sessions, not just within one

---

## Real-World Applications

E-commerce review monitoring, customer feedback triage, brand reputation tracking, social media sentiment monitoring, product quality early-warning systems, and support ticket prioritization. With batch CSV analysis, a small ops or product team could realistically drop in a day's export of reviews and get a triaged, downloadable breakdown without writing any code.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| ML | Scikit-learn (Logistic Regression, Linear SVM, Naive Bayes) |
| Feature Engineering | TF-IDF Vectorization |
| Data Handling | Pandas, NumPy |
| Frontend | Streamlit |

---

## Built By

**Kerubo Bosire** — Actuarial Science | Data Science | Machine Learning | NLP
