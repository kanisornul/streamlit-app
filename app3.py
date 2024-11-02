import streamlit as st
import pandas as pd

# Sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [24, 30, 22, 35, 29],
    'Score': [88, 92, 85, 95, 80]
}
df = pd.DataFrame(data)

# Set the title of the app
st.title("Interactive Data Filter")

# Create a sidebar for filtering
st.sidebar.header("Filter Options")

# Age filter using slider
age_filter = st.sidebar.slider("Select Age Range", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(20, 30))
st.sidebar.write(f"Selected Age Range: {age_filter}")

# Score filter using slider
score_filter = st.sidebar.slider("Select Score Range", min_value=int(df['Score'].min()), max_value=int(df['Score'].max()), value=(80, 95))
st.sidebar.write(f"Selected Score Range: {score_filter}")

# Apply filters to DataFrame
filtered_df = df[(df['Age'].between(age_filter[0], age_filter[1])) & (df['Score'].between(score_filter[0], score_filter[1]))]

# Display the filtered DataFrame
st.write("Filtered Data:")
st.dataframe(filtered_df)

# Show total count of filtered results
st.write(f"Total Entries: {len(filtered_df)}")
