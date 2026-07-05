import streamlit as st
import sys
from pathlib import Path

# =====================================
# Import Prediction Function
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict import predict_sentiment

# =====================================
# Streamlit Configuration
# =====================================

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# Custom CSS
# =====================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* HERO */

.hero{

background:linear-gradient(135deg,#4F46E5,#7C3AED);

padding:40px;

border-radius:18px;

color:white;

text-align:center;

margin-bottom:25px;

box-shadow:0px 8px 30px rgba(0,0,0,0.20);

}

.hero h1{

font-size:45px;

margin-bottom:10px;

}

.hero p{

font-size:20px;

opacity:0.95;

}

/* Result Card */

.result-card{

padding:20px;

border-radius:15px;

margin-top:15px;

margin-bottom:20px;

}

/* Footer */

.footer{

text-align:center;

color:gray;

padding-top:20px;

}

</style>
""", unsafe_allow_html=True)

# =====================================
# Sidebar
# =====================================

with st.sidebar:

    st.title("📌 Project Information")

    st.divider()

    st.subheader("🤖 Model")

    st.success("Logistic Regression")

    st.subheader("📝 Feature Extraction")

    st.info("TF-IDF Vectorizer")

    st.subheader("📚 Dataset")

    st.info("IMDb 50K Reviews")

    st.subheader("📈 Validation Accuracy")

    st.success("87.98%")

    st.divider()

    st.subheader("🛠 Tech Stack")

    st.markdown("""
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Streamlit
""")

    st.divider()

    st.caption("Developed by Arpitha R")

# =====================================
# Hero Section
# =====================================

st.markdown("""

<div class="hero">

<h1>🎬 Movie Review Sentiment Analyzer</h1>

<p>

Analyze IMDb movie reviews using Natural Language Processing

</p>

<p>

<b>TF-IDF • Logistic Regression • Streamlit</b>

</p>

</div>

""", unsafe_allow_html=True)

# =====================================
# Session State
# =====================================

if "review" not in st.session_state:
    st.session_state.review = ""

# =====================================
# Main Input Section
# =====================================

st.subheader("📝 Enter Your Movie Review")

st.write(
    "Paste a movie review below or use one of the sample reviews."
)

positive_review = (
    "This movie was absolutely amazing. The acting was brilliant, "
    "the story was engaging, and I loved every minute of it."
)

negative_review = (
    "Worst movie I have ever watched. The acting was terrible, "
    "the plot was boring, and it was a complete waste of time."
)

col1, col2 = st.columns(2)

with col1:
    if st.button("😊 Load Positive Example", use_container_width=True):
        st.session_state.review = positive_review
        st.rerun()

with col2:
    if st.button("😔 Load Negative Example", use_container_width=True):
        st.session_state.review = negative_review
        st.rerun()

review = st.text_area(
    label="Movie Review",
    value=st.session_state.review,
    height=220,
    placeholder="Type or paste your movie review here..."
)

st.write("")

predict = st.button(
    "🚀 Analyze Sentiment",
    type="primary",
    use_container_width=True
)

# =====================================
# Prediction Logic
# =====================================

if predict:

    if review.strip() == "":

        st.warning("⚠ Please enter a movie review.")

    else:

        (
            sentiment,
            confidence,
            positive_probability,
            negative_probability,
        ) = predict_sentiment(review)

        # =====================================
        # Prediction Result
        # =====================================

        st.divider()

        st.subheader("📊 Prediction Result")

        if sentiment == "Positive":

            st.success("### 🟢 POSITIVE REVIEW")

        else:

            st.error("### 🔴 NEGATIVE REVIEW")

        # =====================================
        # Confidence
        # =====================================

        st.subheader("🎯 Model Confidence")

        st.progress(confidence / 100)

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        # =====================================
        # Probability Breakdown
        # =====================================

        st.subheader("📈 Probability Breakdown")

        col1, col2 = st.columns(2)

        with col1:

            st.write("🟢 POSITIVE")

            st.progress(positive_probability / 100)

            st.write(f"**{positive_probability:.2f}%**")

        with col2:

            st.write("🔴 NEGATIVE")

            st.progress(negative_probability / 100)

            st.write(f"**{negative_probability:.2f}%**")

        # =====================================
        # Review Preview
        # =====================================

        st.subheader("📝 Review Submitted")

        st.text_area(
            "",
            value=review,
            height=180,
            disabled=True
        )

        # =====================================
        # Model Explanation
        # =====================================

        with st.expander("ℹ️ How does the model work?"):

            st.markdown("""
This application uses a **Natural Language Processing (NLP)** pipeline to
classify IMDb movie reviews.

### Workflow

1. Text Preprocessing
   - Lowercasing
   - Remove HTML
   - Remove URLs
   - Remove punctuation
   - Remove numbers
   - Remove stopwords
   - Lemmatization

2. Feature Extraction
   - TF-IDF Vectorization

3. Classification
   - Logistic Regression

The model predicts whether the review expresses a **Positive** or **Negative** sentiment.
""")

# =====================================
# Footer
# =====================================

st.divider()

st.markdown(
    """
<div style='text-align:center;color:gray;'>

Built with ❤️ using Python, Scikit-learn, NLTK and Streamlit

</div>
""",
    unsafe_allow_html=True
)