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

The keyword highlighting feature doesn't fix any of these three problems, but it does something almost as useful: it makes them visible. When the model gets a sarcastic or mixed review wrong, the keyword breakdown usually shows you exactly why — you can see it latched onto "excellent" and missed the sarcasm, or weighted the positive clause more heavily than the negative one. That transparency turns a silent failure into a legible one, which is most of what you actually want from an interpretable model.

---

## Making the Model Legible

A classifier that just prints `"Positive (87.23% confidence)"` asks you to trust it blindly. That's a bad place to leave a model that — as the next section explains — has real, predictable failure modes. So the interface was built to expose its reasoning, not just its output:

**Confidence bar.** The raw probability score is rendered as an animated bar that fills green or red depending on the call — faster to read at a glance than a percentage, especially when you're scanning many reviews back to back.

**Keyword signal analysis.** The app highlights the specific words it picked up on — `excellent`, `terrible`, and so on — directly in the review text. This is the single most important interface decision in the app: it turns "the model says positive" into "the model says positive *because of these words*," which is the difference between a black box and a tool you can actually audit and trust.

**Top keyword chips.** A cleaned, stopword-filtered list of the most prominent words in the review, displayed as chips. Useful for skimming a long review without reading the whole thing.

**Text statistics panel.** Word count, sentence count, unique word count, average word length. Small, but it signals the tool is actually parsing the text rather than just running it through a black box and printing a label.

**Example buttons.** Three pre-filled examples — clearly positive, mixed, and clearly negative — so a new visitor can see the model in action immediately, including on the mixed case where it's expected to be less confident. That last part matters: showing the model struggle a little on a genuinely ambiguous review is more honest than only ever demoing the easy cases.

**Sidebar settings.** Toggles to show or hide each analysis section (confidence bar, keywords, stats), so the default view stays clean but power users can expand everything.

---

## Batch Analysis

The biggest functional upgrade: you can now upload a CSV with a `review` column and get every row classified in one pass, with summary metrics — total reviews, positive count, negative count, positive rate — and a downloadable results CSV.

This is what moves the project from "demo of a model" to "tool someone could actually use." A single-review classifier is a nice proof of concept; a batch processor that takes a day's worth of reviews and returns a triaged CSV is something a small team could genuinely plug into a feedback workflow without writing a line of code.

**Analysis history.** Every review run through the single-review mode gets logged in session state and shown in a scrollable sidebar history with colour-coded sentiment badges — useful for comparing how the model handled a series of reviews in the same session, and it makes the app feel persistent rather than stateless.

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
