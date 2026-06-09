# 🧠 SentimentIQ — Amazon Review Intelligence Platform

A machine learning-powered web application that classifies Amazon product reviews as **Positive** or **Negative** sentiment using Natural Language Processing — built to demonstrate that strong preprocessing and feature engineering can produce highly effective NLP systems without expensive deep learning infrastructure.

🌐 **Live Demo:** https://sentiment-analysisapp-qqzxw6exrhfvsby5ygndf2.streamlit.app/

### Snippet of the app

<img width="1350" height="563" alt="image" src="https://github.com/user-attachments/assets/1436ead3-e949-49f1-a736-0249c94ed81c" />


## 📌 Problem Statement

E-commerce platforms receive thousands of customer reviews daily, making it difficult for businesses to manually monitor customer satisfaction and identify negative product experiences in real time.

Without automated sentiment analysis:
- Poor customer experiences may go unnoticed
- Product quality issues become harder to detect
- Businesses struggle to analyze customer feedback at scale

This project uses NLP and machine learning to automatically classify Amazon product reviews as positive or negative, enabling faster and more scalable customer feedback analysis.


## ✅ What the App Does

Paste any product review and SentimentIQ instantly returns:

- **Sentiment prediction** (Positive or Negative) with a visual confidence bar
- **Keyword signal detection** — which specific words drove the prediction
- **Text statistics** — word count, sentence count, unique vocabulary, average word length
- **Top keywords** extracted from the review (stopwords removed)
- **Batch CSV analysis** — upload hundreds of reviews, get results with a summary dashboard, download as CSV
- **Session history** — every analysis logged in the sidebar for easy reference

| Feature | Description |
|---|---|
| Single review analysis | Real-time prediction with animated confidence bar |
| Keyword signal detection | Highlights positive and negative trigger words |
| Text statistics panel | Word count, sentences, unique words, avg length |
| Batch CSV analysis | Upload → classify → download results |
| Analysis history | Session history in the sidebar |
| Pre-filled examples | One-click positive, mixed, and negative samples |
| Toggleable panels | Show/hide each section via sidebar settings |


## 🏗️ System Architecture

```
User Review Input
       ↓
Text Preprocessing
(lowercase · strip HTML · remove special chars)
       ↓
TF-IDF Vectorization
(unigrams + bigrams · max 1000 features · stopword removal)
       ↓
Logistic Regression Model
       ↓
Sentiment + Confidence Score
```

## 🧠 Approach

### 1. Data Understanding
- **Dataset:** Amazon Product Reviews
- **Target:** Sentiment label (binary classification)
- **Features:** Review headline + body merged to create richer textual context

### 2. Data Cleaning
- Lowercasing, HTML tag removal, special character stripping, whitespace normalisation
- Combining headline and review body for maximum signal

### 3. Feature Engineering
TF-IDF Vectorization was used to convert text into numerical features:
- Unigrams and bigrams
- Stopword removal
- `max_features=1000` to reduce sparsity

TF-IDF was chosen because it performs exceptionally well on sparse textual classification tasks.

### 4. Handling Class Imbalance
The dataset contained imbalanced sentiment classes. Random oversampling was applied to improve minority class recall — though this introduces a slight overfitting risk.

### 5. Model Selection
Three models were benchmarked:

| Model | Notes |
|---|---|
| Naive Bayes | Fast baseline |
| **Logistic Regression** ✅ | **Best overall — selected** |
| Linear SVM | Competitive but less interpretable |

Hyperparameter tuning via **GridSearchCV** with 5-fold cross-validation, optimising for **F1-score**.

## 🏆 Key Insight

Logistic Regression consistently delivered the best balance of accuracy, stability, generalisation, and interpretability. This is largely because it performs exceptionally well on high-dimensional sparse TF-IDF feature spaces.

## 🤔 Why Traditional ML Instead of Deep Learning?

Although transformer architectures like BERT achieve state-of-the-art NLP performance, this project intentionally uses traditional ML because it offers:

- Faster training times
- Lower computational cost
- Easier interpretability
- Simpler deployment
- Strong performance on structured sentiment classification tasks

> *"Simple models combined with strong preprocessing and feature engineering can outperform unnecessarily complex solutions."*

## ⚠️ Model Limitations

**Sarcasm & irony** — the model reads keyword patterns, not intent.
> *"Oh great, another product that stopped working in 2 days."* → may be misclassified as positive.

**Mixed sentiment** — long reviews with both positive and negative clauses can confuse the classifier.
> *"The product quality is excellent, but delivery was terrible."*

**Domain shift** — trained on Amazon product reviews; performance may drop on tweets, app reviews, or informal slang-heavy text.

## 🔧 Future Improvements

1. **Transformer-based models** — replace TF-IDF + LR with fine-tuned DistilBERT for contextual understanding and sarcasm detection
2. **Improved imbalance handling** — class weighting or focal loss instead of oversampling
3. **Advanced preprocessing** — lemmatisation, negation handling (`"not good"` → strong negative), context-aware tokenisation
4. **Threshold optimisation** — tune the 0.5 decision boundary for precision/recall trade-offs specific to business objectives
5. **Aspect-based sentiment** — separate scores for product quality, delivery, value, etc.

## 📊 Evaluation Metrics

F1-score was prioritised due to class imbalance. Full evaluation used: Accuracy · Precision · Recall · F1-score · Confusion Matrix.

## 🏢 Real-World Applications

- E-commerce review monitoring
- Customer feedback analysis at scale
- Brand reputation tracking
- Social media sentiment monitoring
- Product quality issue detection
- Customer support ticket prioritisation

## 🛠️ Tech Stack

Python · Scikit-learn · Pandas · NumPy · TF-IDF Vectorisation · Logistic Regression · Linear SVM · Naive Bayes · Streamlit


## 🚀 Run Locally

```bash
git clone https://github.com/kerubobosire254/sentiment-analysis_app.git
cd sentiment-analysis_app
pip install -r requirements.txt
streamlit run app.py
```

`model.pkl` must be present in the root directory before running.

## 📁 Project Structure

```
sentiment-analysis_app/
├── app.py                        # Streamlit application
├── model.pkl                     # Trained Logistic Regression pipeline
├── requirements.txt              # Python dependencies
├── sentiment analysis.ipynb      # Training & evaluation notebook
└── Amazon-Product-Reviews.csv    # Training dataset
```

---

## 👩‍💻 Author

**Kerubo Bosire** — Actuarial Science · Data Science · Machine Learning · NLP

[![GitHub](https://img.shields.io/badge/GitHub-kerubobosire254-181717?style=flat&logo=github)](https://github.com/kerubobosire254)
