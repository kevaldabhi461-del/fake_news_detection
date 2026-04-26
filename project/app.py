import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import torch.nn.functional as F

# Load model
model = BertForSequenceClassification.from_pretrained("bert_model")
tokenizer = BertTokenizer.from_pretrained("bert_model")

API_KEY = "473f6d2d31be4d1194800e3f1ecce3dc"

# Prediction
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()

    return fake_prob, real_prob

# Fetch news
def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}"
    res = requests.get(url).json()
    return [a["title"] for a in res.get("articles", [])[:5]]

# UI
st.title("🤖 Deep Learning Fake News Detector (BERT)")

text = st.text_area("Enter News Text")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Enter text")
    else:
        fake_p, real_p = predict(text)

        if real_p > fake_p:
            st.success(f"🟢 Likely Real ({round(real_p*100,2)}%)")
        else:
            st.error(f"🔴 Likely Fake ({round(fake_p*100,2)}%)")

        st.write("🔍 Related News:")
        news = fetch_news(text)
        for n in news:
            st.write("•", n)