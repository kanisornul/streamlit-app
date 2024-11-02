import streamlit as st

st.title("Container Layout Example")

# Create a container
with st.container():
    st.header("Container 1")
    st.write("This is content inside Container 1")

# Another container
with st.container():
    st.header("Container 2")
    st.write("This is content inside Container 2")