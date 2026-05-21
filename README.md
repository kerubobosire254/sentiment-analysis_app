# 🧠 NLP-Powered Amazon Review Sentiment Classifier

A machine learning-powered web application that classifies Amazon product reviews as **positive** or **negative** sentiment using Natural Language Processing (NLP) techniques and traditional machine learning models.

This project demonstrates how effective preprocessing, feature engineering, and model selection can build reliable NLP systems without requiring computationally expensive deep learning architectures.

## 🚀 Live Demo

🌐 Streamlit Cloud Deployment:  
https://sentiment-analysisapp-qqzxw6exrhfvsby5ygndf2.streamlit.app/

## 📌 Problem Statement

E-commerce platforms receive thousands of customer reviews daily, making it difficult for businesses to manually monitor customer satisfaction and identify negative product experiences in real time.

Without automated sentiment analysis:
- Poor customer experiences may go unnoticed
- Product quality issues become harder to detect
- Businesses struggle to analyze customer feedback at scale

This project uses Natural Language Processing (NLP) and machine learning to automatically classify Amazon product reviews as positive or negative sentiment, enabling faster and more scalable customer feedback analysis.

## ✅ Solution Overview

The application allows users to input Amazon-style product reviews and instantly receive:

- Sentiment prediction (Positive or Negative)
- Confidence score
- Real-time text classification through a deployed web interface

The system processes raw text using NLP preprocessing techniques and transforms text into numerical representations using TF-IDF vectorization before passing it into trained machine learning models for prediction.

## 🏗️ System Architecture

```text
User Review
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Trained ML Model
    ↓
Sentiment Prediction
```

## 🧠 Approach

This project follows a classic NLP machine learning pipeline:

### 1. Data Understanding

- Dataset: Amazon Product Reviews
- Target Variable: Sentiment Label (Binary Classification)
- Features Used:
  - Review Headline
  - Review Body

The headline and body were merged to create richer textual context for prediction.

### 2. Data Cleaning

Text preprocessing included:

- Lowercasing text
- Removing HTML tags
- Removing special characters and numbers
- Removing unnecessary whitespace
- Combining review headline and body

These steps help reduce noise and improve model consistency.

### 3. Feature Engineering

TF-IDF (Term Frequency–Inverse Document Frequency) Vectorization was used to convert text into numerical features.

Techniques applied:

- Uni-grams and bi-grams
- Stopword removal
- Maximum feature cap (`max_features=1000`) to reduce sparsity

TF-IDF was chosen because it performs exceptionally well on sparse textual classification tasks.

### 4. Handling Class Imbalance

The dataset contained imbalanced sentiment classes.

To improve minority class prediction:
- Random oversampling was applied

This improved recall performance, although it introduced a slight risk of overfitting.

### 5. Model Selection

Three machine learning models were compared:

- Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

Hyperparameter tuning was performed using:

- GridSearchCV
- 5-fold Cross Validation
- F1-score optimization

## 🏆 Key Insight

Logistic Regression consistently delivered the best balance between:

- Accuracy
- Stability
- Generalization
- Interpretability

This is largely because Logistic Regression performs extremely well on high-dimensional sparse TF-IDF features.

## 🤔 Why Traditional Machine Learning Instead of Deep Learning?

Although transformer-based architectures like BERT achieve state-of-the-art NLP performance, this project intentionally focuses on traditional machine learning models because they offer:

- Faster training times
- Lower computational cost
- Easier interpretability
- Simpler deployment
- Strong performance on structured sentiment classification tasks

This project demonstrates that strong preprocessing and feature engineering can still produce highly effective NLP systems without requiring heavy deep learning infrastructure.

## ⚠️ Model Limitations

No model is perfect. This classifier still struggles with:

### 1. Sarcasm & Irony

Example:

> “Oh great, another product that stopped working in 2 days.”

The model may misclassify sarcastic reviews because traditional models rely heavily on keyword patterns rather than contextual understanding.

### 2. Mixed Sentiment Reviews

Long reviews containing both positive and negative clauses can confuse the classifier.

Example:
- “The product quality is excellent, but delivery was terrible.”

### 3. Domain Shift

The model was trained on Amazon product reviews and may not generalize effectively to:

- Twitter sentiment
- Informal slang-heavy text
- App reviews
- Customer support conversations

## 🔧 Future Improvements

If this project were extended to production-level deployment, improvements would include:

### 1. Transformer-Based Models

Replacing TF-IDF + traditional ML with:
- BERT
- DistilBERT

This would significantly improve contextual understanding and sarcasm detection.

### 2. Improved Imbalance Handling

Instead of oversampling:
- Class weighting
- Focal loss techniques

could improve robustness.

### 3. Advanced NLP Preprocessing

Additional improvements:
- Lemmatization
- Negation handling
- Context-aware tokenization

Example:
- “not good” should carry stronger negative meaning.

### 4. Threshold Optimization

Instead of using the default prediction threshold of 0.5, threshold tuning could improve:
- Precision
- Recall
- Business-specific performance objectives

## 📊 Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

F1-score was prioritized because the dataset contained class imbalance.

## 🌐 Deployment

The application was deployed using Streamlit.

Workflow:
1. User inputs a review
2. Text is cleaned and vectorized
3. The trained model predicts sentiment
4. Confidence probability is displayed

## 🏢 Real-World Applications

This type of sentiment analysis system can be applied in:

- E-commerce review monitoring
- Customer feedback analysis
- Brand reputation tracking
- Social media sentiment monitoring
- Product quality monitoring
- Customer support prioritization

## 💡 Key Takeaway

This project highlights an important machine learning principle:

> “Simple models combined with strong preprocessing and feature engineering can outperform unnecessarily complex solutions.”

At the same time, it demonstrates the limitations of traditional NLP systems and the growing importance of transformer-based architectures for production-grade language understanding.

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- TF-IDF Vectorization
- Logistic Regression
- Linear SVM
- Naive Bayes
- Streamlit

## 👩‍💻 Author

Built by Kerubo Bosire

Actuarial Science | Data Science | Machine Learning | NLP
