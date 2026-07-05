![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![License](https://img.shields.io/badge/License-MIT-green)

# 🎬 Movie Review Sentiment Analyzer

An end-to-end Natural Language Processing (NLP) web application that classifies IMDb movie reviews as **Positive** or **Negative** using Machine Learning.

Built using **Python, Scikit-learn, TF-IDF, Logistic Regression, NLTK, and Streamlit**.

---

## 🚀 Live Demo

(https://nlp-sentiment-analysis-6wsumeu3f2bnkjxqfp83re.streamlit.app/)

---

## 📌 Features

- Predicts whether a movie review is Positive or Negative
- Complete NLP preprocessing pipeline
- TF-IDF Vectorization
- Logistic Regression classifier
- Confidence score visualization
- Positive & Negative probability breakdown
- Interactive Streamlit web application
- Professional UI with example reviews

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Joblib

---

## 📊 Model Performance

| Model | Accuracy |
|--------|----------|
| Multinomial Naive Bayes | 84.54% |
| Logistic Regression | **87.98%** ✅ |
| Linear SVM | 88.04% |

Although Linear SVM achieved a slightly higher accuracy, Logistic Regression was selected because it provides probability estimates (`predict_proba`), enabling confidence scores in the web application.

---

## 🧠 NLP Pipeline

Movie Review

↓

Text Preprocessing

- Lowercasing
- HTML Removal
- URL Removal
- Punctuation Removal
- Number Removal
- Stopword Removal
- Lemmatization

↓

TF-IDF Vectorization

↓

Logistic Regression

↓

Prediction

↓

Confidence Score

---

## 📂 Project Structure

```text
nlp-sentiment-analysis/
│
├── app/
│   └── app.py
│
├── assets/
│   ├── home.png
│   ├── positive_prediction.png
│   └── negative_prediction.png
│
├── data/
│   └── raw/
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

---

## 📷 Screenshots

### Home Page

![Home](assets/home.png)

---

### Positive Prediction

![Positive](assets/positive_prediction.png)

---

### Negative Prediction

![Negative](assets/negative_prediction.png)

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/arpitha-25/nlp-sentiment-analysis.git
```

Move into the project

```bash
cd nlp-sentiment-analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

## 🎯 Future Improvements

- Deep Learning using LSTM
- BERT-based sentiment analysis
- Multi-class sentiment classification
- Aspect-based sentiment analysis
- Docker deployment
- CI/CD with GitHub Actions

---

## 👩‍💻 Author

**Arpitha R**

GitHub: https://github.com/arpitha-25
