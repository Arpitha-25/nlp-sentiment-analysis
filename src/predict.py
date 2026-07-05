import joblib
from pathlib import Path
from preprocessing import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model and vectorizer
model = joblib.load(BASE_DIR / "models" / "sentiment_model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")


def predict_sentiment(review):
    """
    Predict sentiment of a movie review.
    """

    # Clean the review
    clean_review = preprocess_text(review)

    # Convert to TF-IDF features
    vector = vectorizer.transform([clean_review])

    # Predict sentiment
    prediction = model.predict(vector)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(vector)[0]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😔"

    confidence = probabilities[prediction] * 100

    return sentiment, confidence

if __name__ == "__main__":

    review = "This movie was absolutely fantastic. I loved every second."

    sentiment, confidence = predict_sentiment(review)

    print(sentiment)
    print(f"Confidence: {confidence:.2f}%")