import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page Config
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Sentiment Analysis System")

st.write("Enter any review or tweet")

user_input = st.text_area("Enter Text")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:

        vector = vectorizer.transform([user_input])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        # Debug Output
        st.write("Prediction:", prediction)

        # Convert to lowercase
        pred = str(prediction).lower()

        if pred == "positive":
            st.success("Positive 😊")

        elif pred == "negative":
            st.error("Negative 😠")

        elif pred == "neutral":
            st.info("Neutral 😐")

        else:
            st.warning(f"Other Class Detected: {prediction}")

        st.subheader("Confidence Score")

        scores = {
            cls: round(prob * 100, 2)
            for cls, prob in zip(
                model.classes_,
                probability
            )
        }

        st.write(scores)

        st.bar_chart(scores)

        st.subheader("Model Classes")

        st.write(model.classes_)
