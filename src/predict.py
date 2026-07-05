import joblib
from pathlib import Path
from preprocessing import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model and vectorizer
model = joblib.load(BASE_DIR / "models" / "sentiment_model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")


def predict_sentiment(review):
    """
    Predict the sentiment of a movie review.
    Returns:
        sentiment,
        confidence,
        positive_probability,
        negative_probability
    """

    clean_review = preprocess_text(review)

    vector = vectorizer.transform([clean_review])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    positive_probability = probabilities[1] * 100
    negative_probability = probabilities[0] * 100

    sentiment = "Positive" if prediction == 1 else "Negative"

    confidence = max(positive_probability, negative_probability)

    return (
        sentiment,
        confidence,
        positive_probability,
        negative_probability
    )


if __name__ == "__main__":

    review = "This movie was absolutely fantastic. I loved every second."

    sentiment, confidence = predict_sentiment(review)

    print(sentiment)
    print(f"Confidence: {confidence:.2f}%")