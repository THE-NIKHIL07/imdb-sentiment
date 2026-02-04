import streamlit as st
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import plotly.graph_objects as go

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    body, .stApp, .css-18e3th9 {background-color: white; color: black;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center; font-size:36px; font-weight:bold; color:black;'>IMDB Movie Review Sentiment Analysis 🎬</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:black;'>Enter a review, predict sentiment, and compare with past reviews.</p>", unsafe_allow_html=True)

HF_MODEL_REPO = "THE-NIKHIL07/bert-sentiment-model"

@st.cache_resource
def load_hf_model():
    tokenizer = BertTokenizerFast.from_pretrained(HF_MODEL_REPO)
    model = BertForSequenceClassification.from_pretrained(HF_MODEL_REPO)
    model.eval()
    return tokenizer, model

tokenizer, model = load_hf_model()

review_input = st.text_area("Enter Review:", placeholder="Enter the review here...", height=150)

if 'reviews' not in st.session_state: st.session_state.reviews = []
if 'predictions' not in st.session_state: st.session_state.predictions = []

def predict_sentiment(review):
    tokens = tokenizer(review, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        positive_score = probs[0][1].item() * 100
        sentiment = "Positive 👍" if positive_score >= 50 else "Negative 👎"
        return positive_score, sentiment

if st.button("Predict"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review to predict sentiment.")
    else:
        try:
            score, sentiment = predict_sentiment(review_input)
            label = review_input[:50] + "..." if len(review_input) > 50 else review_input
            st.session_state.reviews.append(label)
            st.session_state.predictions.append(score)
            st.markdown(f"<p style='color:black; font-size:20px;'>Sentiment: <b>{sentiment}</b> (Confidence score: {score:.2f}%)</p>", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(st.session_state.reviews)+1)),
                y=st.session_state.predictions,
                mode='markers+lines',
                marker=dict(size=10, color='black'),
                line=dict(color='green', width=2),
                text=st.session_state.reviews,
                hovertemplate="<b>%{text}</b><br>Confidence: %{y:.2f}%<extra></extra>"
            ))

            fig.update_layout(
                title=dict(text="Sentiment Confidence of Each Review", font=dict(color="black")),
                xaxis_title=dict(text="Sample Number", font=dict(color="black")),
                yaxis_title=dict(text="Positive Sentiment", font=dict(color="black")),
                xaxis=dict(color="black", tickfont=dict(color="black")),
                yaxis=dict(color="black", tickfont=dict(color="black"), range=[0,100]),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color="black"),
                margin=dict(l=40,r=40,t=60,b=60),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
