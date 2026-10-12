import streamlit as st
import pickle

# Page settings
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)

# Load trained model
with open("emotion_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Title
st.title("😊 Emotion Detection from Social Media Text")

st.write(
    "Enter a social media sentence below and the system "
    "will predict the emotion."
)

# Text input
text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am very happy today!"
)

# Emotion information
emotion_info = {
    "happy": ("😊", "This text expresses happiness."),
    "sad": ("😢", "This text expresses sadness."),
    "angry": ("😡", "This text expresses anger."),
    "surprise": ("😲", "This text expresses surprise."),
    "fear": ("😨", "This text expresses fear.")
}

# Prediction
if st.button("🔍 Predict Emotion"):
    if text.strip():

        # Convert text into TF-IDF
        text_vectorized = vectorizer.transform([text])

        # Predict
        prediction = model.predict(text_vectorized)[0]

        # Show result
        emoji, message = emotion_info.get(
            prediction.lower(),
            ("🙂", "Emotion detected.")
        )

        st.success(f"{emoji} Predicted Emotion: {prediction.title()}")
        st.info(message)

    else:
        st.warning("⚠️ Please enter some text first.")