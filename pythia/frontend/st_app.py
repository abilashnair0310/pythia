# streamlit_app.py
import streamlit as st
import requests
import json
import pandas as pd

st.title("DIG Pythia")

backend_url = "http://127.0.0.1:5000/chat_csv"

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    st.write("File uploaded successfully.")

    response_option = st.selectbox(
    "How do you want the model to respond?",
    ("Dataframe", "Code/Analysis"))
    
    # Input for questions
    question = st.text_input("Ask a question about the data")
    
    if st.button("Submit"):
        if question:
            files = {'csv_file': uploaded_file}
            data = {'query': question, 'response_type': response_option}
            response = requests.post(backend_url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                if response_option == 'Dataframe':
                    result = pd.read_json(result['answer'])
                    st.dataframe(result)
                else:
                    result = result['answer']
                    st.write(result)
            else:
                st.error("Failed to get an answer from the backend. " + response.json().get("error", "Unknown error"))
        else:
            st.error("Please enter a question.")

