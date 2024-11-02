import streamlit as st
import pandas as pd
import numpy as np

# Set a title for the app
st.title("Sample Streamlit App")

# Text input for user name
name = st.text_input("Enter your name:", "Guest")

# Display a welcome message
st.write(f"Hello, {name}! Welcome to the Streamlit app.")

# Generate random data for demonstration
data = pd.DataFrame({
    'A': np.random.randn(50),
    'B': np.random.randn(50)
})

# Display the data as a chart
st.line_chart(data)
