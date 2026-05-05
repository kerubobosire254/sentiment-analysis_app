# 💬 Sentiment Analysis of Amazon Product Reviews

A machine learning-powered web application that classifies Amazon product reviews as **positive or negative sentiment** 
using Natural Language Processing (NLP) techniques and traditional machine learning models.

Built with:
- Python
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression / SVM / Naive Bayes
- Streamlit (for deployment)

## 🚀 Live Demo
  Local URL: http://localhost:8502
  Network URL: http://192.168.1.184:8502
  
## streamlit cloud
https://sentiment-analysisapp-qqzxw6exrhfvsby5ygndf2.streamlit.app/
  
## 📌 Problem Statement

Customer reviews contain valuable signals about product satisfaction, but manually analyzing them at scale is impossible.

This project builds a classifier that:
- Takes raw product reviews
- Cleans and processes text
- Predicts sentiment (positive / negative)
- Outputs confidence scores for interpretability

## 🧠 Approach

This project follows a classic NLP pipeline:

### 1. Data Understanding
- Dataset: Amazon product reviews
- Target: Sentiment label (binary classification)
- Features: Review headline + review body

### 2. Data Cleaning
Text preprocessing includes:
- Lowercasing
- Removing HTML tags
- Removing special characters and numbers
- Merging headline + body for richer context

### 3. Feature Engineering
- TF-IDF Vectorization
- Uni-grams and bi-grams used
- Stopwords removed
- Feature cap set to reduce sparsity (max_features=1000)


### 4. Handling Class Imbalance
The dataset was imbalanced, so I applied:
- Random oversampling of minority class

This improves recall on underrepresented sentiment classes but introduces slight overfitting risk.


### 5. Model Selection

I compared three models:

- Naive Bayes (baseline, fast, interpretable)
- Logistic Regression (best balance of performance + stability)
- Linear SVM (strong margin-based classifier)

Hyperparameter tuning was done using GridSearchCV with 5-fold cross-validation using F1-score as the evaluation metric.

## 🏆 Key Insight

Logistic Regression consistently performed best due to:
- High-dimensional TF-IDF compatibility
- Good generalization on sparse text data
- Stable performance across folds


## ⚠️ Where the Model Fails

No model is perfect. This one struggles with:

### 1. Sarcasm & irony
Example:
> “Oh great, another product that stopped working in 2 days.”

Model often misclassifies due to literal keyword reliance.

### 2. Context-heavy reviews
Long reviews with mixed sentiment confuse the classifier (positive + negative clauses).

### 3. Domain shift
Trained on Amazon data → may not generalize well to:
- Twitter sentiment
- App reviews
- informal slang-heavy text

## 🔧 How It Can Be Improved

If this were production-grade, I would upgrade it using:

### 1. Transformer-based models
Replace TF-IDF + ML with:
- BERT / DistilBERT
This would significantly improve contextual understanding.

### 2. Better imbalance handling
Instead of oversampling:
- Use class weights (more stable)
- Or focal loss in deep learning models


### 3. Advanced text preprocessing
- Lemmatization
- Negation handling (“not good” → important reversal signal)

### 4. Threshold tuning
Adjust decision thresholds instead of default 0.5 for better recall/precision balance.

## 📊 Evaluation Metrics

Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

F1-score was prioritized due to class imbalance.


## 🌐 Deployment

The model is deployed using Streamlit:

- User inputs a review
- Text is cleaned and vectorized
- Model predicts sentiment
- Confidence score is displayed

## 💡 Key Takeaway

This project is a classic example of:
> “Simple models + good preprocessing can outperform complex models if done correctly.”

However, it also highlights the limitations of traditional NLP approaches and motivates a transition toward transformer-based architectures for production-level sentiment analysis.

## 🧑‍💻 Author

Built by Naomi Kerubo  
Actuarial Science | Data Science | ML Enthusiast
