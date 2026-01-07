import streamlit as st
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import plotly.graph_objects as go


# Model & Data Setup

max_features = 10000
maxlen = 500
model_path = "rnn_imdb_model.h5"

@st.cache_resource
def load_sentiment_model():
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_sentiment_model()

@st.cache_data
def load_word_index():
    word_index = imdb.get_word_index()
    word_index = {k: (v+3) for k,v in word_index.items()}  
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    return word_index

word_index = load_word_index()

# Preprocessing

def preprocess_review(review):
    tokens = review.lower().split()
    seq = []
    for word in tokens:
        idx = word_index.get(word, 2)  
        if idx >= max_features:
            idx = 2
        seq.append(idx)
    padded = pad_sequences([seq], maxlen=maxlen)
    return padded

#css theme
st.markdown("""
    <style>
    body, .stApp, .block-container {
        background-color: #ffffff;
        color: #000000;
    }
    .stTitle {
        color: #000000;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    textarea, .stTextArea>div>div>textarea {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;  
        caret-color: black !important;
        border-radius: 8px;
        padding: 10px;
    }
    ::placeholder {
        color: #888888 !important;
    }
    /* Black Predict button */
    div.stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 8px 20px;
    }
    div.stButton > button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)


# Streamlit UI

st.markdown("<h1 class='stTitle'>IMDB Movie Review Sentiment Analysis 🎬</h1>", unsafe_allow_html=True)
st.markdown("Enter a review, predict sentiment, and compare with past reviews.", unsafe_allow_html=True)

review_input = st.text_area("Enter Review:", placeholder="Enter the review here...", height=150)

# Initialize session state
if 'reviews' not in st.session_state: st.session_state.reviews = []
if 'predictions' not in st.session_state: st.session_state.predictions = []

if st.button("Predict"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review to predict sentiment.")
    else:
        try:
            if model is None:
                st.error("Model not loaded. Cannot make predictions.")
            else:
                # Preprocess and predict
                review_seq = preprocess_review(review_input)
                prediction = model.predict(review_seq)[0][0]
                prediction_percent = prediction * 100
                sentiment = "Positive 👍" if prediction >= 0.5 else "Negative 👎"
                
                # Store review & prediction
                label = review_input[:50] + "..." if len(review_input) > 50 else review_input
                st.session_state.reviews.append(label)
                st.session_state.predictions.append(prediction_percent)
                
                #  prediction
                st.success(f"Predicted Sentiment: **{sentiment}** ({prediction_percent:.2f}%)")

               
                # Dynamic Comparison Chart
              
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=st.session_state.reviews,
                    y=st.session_state.predictions,
                    marker_color='#000000',
                    text=[f"{p:.2f}%" for p in st.session_state.predictions],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Sentiment Comparison (%)",
                    xaxis_title="Reviews",
                    yaxis_title="Positive Sentiment (%)",
                    yaxis=dict(range=[0,100]),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(color="black"),
                    margin=dict(l=40,r=40,t=60,b=60),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        except ValueError as ve:
            st.error(f"Input shape error: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
