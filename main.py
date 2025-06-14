import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import streamlit as st


word_index = imdb.get_word_index()
reverse_word_index = dict((value, key) for (key, value) in word_index.items())
model = load_model('simple_rnn_imdb.h5')

def decode_review(text):
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    return decoded_review

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 0) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review, encoded_review

def predict_text(text):
    padded_review, encoded_review = preprocess_text(text)
    prediction = model.predict(padded_review)
    sentiment = 'Positive' if prediction[0][0] > 0.7 else 'Negative'
    return sentiment, prediction[0][0], encoded_review

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")
st.title("IMDB Movie Review Sentiment Analysis")
st.markdown(
    "<h4 style='color:gray;'>Enter a movie review below and click <b>Predict Sentiment</b> to see the result.</h4>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Instructions")
    st.write(
        """
        - Type or paste a movie review in the text box.
        - Click **Predict Sentiment**.
        - The app will display the sentiment and confidence score.
        """
    )
    st.info("Model: Simple RNN trained on IMDB dataset.")

user_input = st.text_area("Enter a movie review:", height=150)

if st.button("Predict Sentiment", use_container_width=True):
    if user_input.strip():
        sentiment, confidence, encoded_review = predict_text(user_input)
        emoji = "😊" if sentiment == "Positive" else "😞"
        color = "green" if sentiment == "Positive" else "red"
        col1, col2 = st.columns(2)
        col1.markdown(f"<h3 style='color:{color};'>{emoji} {sentiment}</h3>", unsafe_allow_html=True)
        col2.metric("Confidence", f"{confidence:.2%}")
        st.markdown("---")
        st.subheader("Decoded Review (as processed by model):")
        st.write(decode_review(encoded_review))
    else:
        st.warning("Please enter a movie review before predicting.")
else:
    st.info("Awaiting your input...")