import streamlit as st
import joblib

# Load the pre-trained model
@st.cache(allow_output_mutation=True)  # Cache the model to avoid reloading it each time
def load_model():
    model = joblib.load("NER_model.joblib")
    return model

# Load the model once when the app starts
model = load_model()

# Defined words
stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
    word = tokens[i]
    
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }
    
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True
    
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True
    
    return features

def parse(text):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    
    # Make a prediction with the model
    prediction = model.predict([features])[0]
    
    return prediction

# Set up your app title
st.title("Named Entity Recognition (NER) Prediction App")

# Text input for user data
user_text = st.text_area("Enter text for NER analysis:")

# Button to make predictions
if st.button("Analyze Text"):
    # Make prediction using the parse function
    prediction = parse(user_text)  # Call parse with user_text directly

    # Display the result
    st.write("Prediction results:", prediction)
