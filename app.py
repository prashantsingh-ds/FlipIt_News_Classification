# Streamlit dashboard for FlipItNews news classification (RNN model)
# Run: streamlit run app.py
# Needs these files in the same folder (made by train.py):
#   rnn_model.keras, tokenizer.json, config.json

import os
import re
import json
import string
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="FlipItNews Classifier", page_icon="📰", layout="wide")

# same stopwords as training
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set(ENGLISH_STOP_WORDS)


# load everything only once
@st.cache_resource
def load_stuff():
    model = load_model("rnn_model.keras")
    with open("tokenizer.json") as f:
        tok = tokenizer_from_json(f.read())
    with open("config.json") as f:
        config = json.load(f)
    return model, tok, config["classes"], config["max_len"]


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def predict(text):
    cleaned = clean_text(text)
    seq = tok.texts_to_sequences([cleaned])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')
    probs = model.predict(seq, verbose=0)[0]
    return probs


# check files first so we get a clear message instead of a big error
needed = ["rnn_model.keras", "tokenizer.json", "config.json"]
missing = [f for f in needed if not os.path.exists(f)]
if missing:
    st.error("Missing files: " + ", ".join(missing) +
             ". Run train.py (or the notebook) first and keep these files next to app.py.")
    st.stop()

model, tok, classes, max_len = load_stuff()

# ---------------- sidebar ----------------
st.sidebar.title("📰 FlipItNews")
st.sidebar.write("Automatic news categorization using an RNN (Bi-LSTM) model.")
page = st.sidebar.radio("Go to", ["Classify Article", "Model Performance", "About"])

# ---------------- page 1 ----------------
if page == "Classify Article":
    st.title("News Article Classifier")
    st.write("Paste a news article below and the model will predict its category.")

    samples = {
        "-- choose a sample --": "",
        "Sports": "India won the cricket match by five wickets after a brilliant century from the captain in the final over at the stadium.",
        "Technology": "The company launched a new smartphone with an advanced processor, better camera and artificial intelligence features for users.",
        "Business": "The stock market rose today as investors reacted to strong quarterly profits and the central bank kept interest rates unchanged.",
        "Politics": "The prime minister addressed parliament about the new bill and opposition leaders demanded a debate before the election.",
        "Entertainment": "The new film starring a famous actor topped the box office on its opening weekend and critics praised the music.",
    }
    choice = st.selectbox("Try a sample", list(samples.keys()))

    text = st.text_area("Article text", value=samples[choice], height=250)

    if st.button("Classify"):
        if len(text.strip().split()) < 5:
            st.warning("Please enter a longer article (at least 5 words).")
        else:
            probs = predict(text)
            idx = int(np.argmax(probs))
            category = classes[idx]

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Predicted Category: **{category}**")
                st.metric("Confidence", f"{probs[idx] * 100:.2f}%")
            with col2:
                prob_df = pd.DataFrame({"Category": classes, "Probability": probs})
                prob_df = prob_df.set_index("Category")
                st.bar_chart(prob_df)

# ---------------- page 2 ----------------
elif page == "Model Performance":
    st.title("Model Performance")

    if os.path.exists("model_comparison.csv"):
        res = pd.read_csv("model_comparison.csv", index_col=0)
        st.subheader("Comparison of all models")
        st.dataframe(res)
        st.bar_chart(res)
    else:
        st.info("model_comparison.csv not found. Run train.py first.")

    c1, c2 = st.columns(2)
    if os.path.exists("category_distribution.png"):
        c1.subheader("Category distribution")
        c1.image("category_distribution.png")
    if os.path.exists("rnn_confusion_matrix.png"):
        c2.subheader("RNN confusion matrix")
        c2.image("rnn_confusion_matrix.png")

# ---------------- page 3 ----------------
else:
    st.title("About")
    st.write("""
    **Problem:** FlipItNews wants to automatically categorize news articles instead of
    labeling them by hand.

    **Approach:**
    - Text cleaning (lowercase, punctuation and stopword removal, tokenization)
    - Features: Bag of Words, TF-IDF, Word2Vec
    - Models: Neural Network and RNN (Bi-LSTM) built with TensorFlow/Keras
    - This dashboard hosts the trained RNN model

    **Benefits for FlipItNews:**
    - Saves manual labeling time
    - Better content organization and search
    - Personalized news feed based on user's favourite categories
    """)
    st.write("Categories:", ", ".join(classes))
