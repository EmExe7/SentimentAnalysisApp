import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a review below:")

user_input = st.text_area("Review")

if st.button("Predict"):
    if user_input:
        # Vectorize the input and predict sentiment
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.write("Please enter a review.")