import streamlit as st 
import string
import pandas as pd
import nltk
import pickle
from nltk.stem import PorterStemmer
porter = PorterStemmer()

st.set_page_config(page_icon="🐦", page_title="Twitter Sentiment analysis", layout="wide")
c1, c2 = st.columns([0.7, 2])
c2.subheader("Twitter Sentiment analysis")
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.image("icons/twitter.png", width=158)

with open("Log_model.pkl", "rb")as file:
    model = pickle.load(file)
    
with open("TF_IDF_vectorizer.pkl", "rb")as file:
    vectorizer = pickle.load(file)
        
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = "".join([char for char in text if not char.isdigit()])
    words = nltk.word_tokenize(text)
    text = " ".join([porter.stem(word) for word in words])
    return text

col1, col2 = st.columns([2,1], border=True)
text = col1.text_area("Text:", placeholder="type something..")

if col1.button("Predict"):
    if text:
        text = clean_text(text)
        trans_text = vectorizer.transform([text])
        pred = model.predict(trans_text)[0]
        pred_prob = model.predict_proba(trans_text)[0]
        if pred==0:
            col2.subheader(f"Negative Sentiment:{pred_prob[0]*100:.2f}%")
            col2.image("icons/sad-face.png", width=150)
        elif pred==1:
            col2.subheader(f"Positive Sentiment:{pred_prob[1]*100:.2f}%")
            col2.image("icons/happy-face.png")