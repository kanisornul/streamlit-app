import streamlit as st
import joblib

# Load the pre-trained model
@st.cache(allow_output_mutation=True)  # Cache the model to avoid reloading it each time
def load_model():
    model = joblib.load("NER_model.joblib")
    return model

# Load the model once when the app starts
model = load_model()

# Set up your app title
st.title("Named Entity Recognition (NER) Prediction App")

# Text input for user data
user_text = st.text_area("Enter text for NER analysis:")

# Button to make predictions
if st.button("Analyze Text"):
    # Assuming your model has a method like `predict()` or `predict_proba()`
    # Adjust this to match your model's prediction method
    prediction = model.predict([user_text])  # Pass user_text inside a list as it expects an array-like input

    # Display the result
    st.write("Prediction results:", prediction)
