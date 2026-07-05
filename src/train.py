import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from preprocessing import preprocess_text


# ==========================
# Load Dataset
# ==========================
train_df = pd.read_csv(BASE_DIR / "data" / "raw" / "Train.csv")
valid_df = pd.read_csv(BASE_DIR / "data" / "raw" / "Valid.csv")
test_df = pd.read_csv(BASE_DIR / "data" / "raw" / "Test.csv")


# ==========================
# Remove Duplicates
# ==========================

train_df = train_df.drop_duplicates().reset_index(drop=True)


# ==========================
# Text Preprocessing
# ==========================

print("Preprocessing Training Data...")
train_df["clean_text"] = train_df["text"].apply(preprocess_text)

print("Preprocessing Validation Data...")
valid_df["clean_text"] = valid_df["text"].apply(preprocess_text)

print("Preprocessing Test Data...")
test_df["clean_text"] = test_df["text"].apply(preprocess_text)


# ==========================
# TF-IDF Vectorization
# ==========================

tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(train_df["clean_text"])
X_valid = tfidf.transform(valid_df["clean_text"])
X_test = tfidf.transform(test_df["clean_text"])

y_train = train_df["label"]
y_valid = valid_df["label"]
y_test = test_df["label"]


# ==========================
# Train Logistic Regression
# ==========================

print("Training Logistic Regression...")

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)


# ==========================
# Validation Results
# ==========================

predictions = model.predict(X_valid)

print("\nValidation Results\n")

print("Accuracy :", accuracy_score(y_valid, predictions))
print("Precision:", precision_score(y_valid, predictions))
print("Recall   :", recall_score(y_valid, predictions))
print("F1 Score :", f1_score(y_valid, predictions))

print("\nClassification Report\n")

print(classification_report(y_valid, predictions))


joblib.dump(model, BASE_DIR / "models" / "sentiment_model.pkl")
joblib.dump(tfidf, BASE_DIR / "models" / "tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")