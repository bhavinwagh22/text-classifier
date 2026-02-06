import streamlit as st
import joblib
import re
import string

# --------------------
# Page config
# --------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# --------------------
# Custom CSS
# --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
}

h1 {
    text-align: center;
    color: #2c3e50;
}

textarea {
    border-radius: 10px !important;
    font-size: 16px !important;
}

.stButton > button {
    background: linear-gradient(90deg, #4b6cb7, #182848);
    color: white;
    border-radius: 25px;
    padding: 10px 25px;
    font-size: 16px;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.03);
}

.result {
    margin-top: 20px;
    padding: 15px;
    border-radius: 12px;
    font-size: 18px;
    text-align: center;
}

.positive {
    background-color: #d4edda;
    color: #155724;
}

.negative {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# Load model & vectorizer
# --------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --------------------
# Text cleaning function
# --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text

# --------------------
# UI
# --------------------
st.title("Sentiment Analysis")

user_review = st.text_area(
    "Enter your review",
    height=150,
    placeholder="Type your review here..."
)

if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_review = clean_text(user_review)
        review_vec = vectorizer.transform([clean_review])
        prediction = model.predict(review_vec)

        if prediction[0] == 1:
            st.markdown(
                "<div class='result positive'>😊 Positive Review</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result negative'>😞 Negative Review</div>",
                unsafe_allow_html=True
            )
