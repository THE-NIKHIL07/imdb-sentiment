# IMDB Movie Review Sentiment Analysis 🎬

This is a simple **web application** that predicts whether a movie review is **Positive** or **Negative** using a trained **Recurrent Neural Network (RNN)** model.

The app is built with **Streamlit**, so anyone can run it locally and try different movie reviews.

---

## What does this app do?

- Takes a movie review as text input
- Predicts sentiment:
  - **Positive**
  - **Negative**
- Shows prediction confidence in **percentage**
- Stores multiple predictions during a session
- Displays a **dynamic comparison chart** of all entered reviews

---

## How does it work?

- The model is trained on the **IMDB movie reviews dataset**
- Uses the **top 10,000 most frequent words**
- Reviews are padded to a fixed length of **500**
- An **RNN-based model** processes the text and outputs sentiment probability

---

## How to Run the App

Copy and run the following commands:

```bash
git clone https://github.com/THE-NIKHIL07/imdb-sentiment-rnn.git
cd imdb-sentiment-rnn
pip install -r requirements.txt
streamlit run app.py
