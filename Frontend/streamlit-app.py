import pandas as pd
import streamlit as st
import os
import requests
import tensorflow as tf

# Function to download the model file
def download_model(url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(response.content)

# Load the trained model
model_url = "https://github.com/Trident09/net-sec-ai-MP/blob/b1aff8fd96776446ae8b5fe959cd5adf7eb65f5c/Frontend/trained_model-neuralnetwork.keras"
model_path = "trained_model-neuralnetwork.keras"
download_model(model_url, model_path)
model = tf.keras.models.load_model(model_path)

# Function to load and preprocess the CSV file
def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop(columns=['Label'], errors='ignore')  # Drop 'Label' column if present
    return df

# Streamlit app
st.title('Traffic Prediction App')
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Preprocess the CSV data
        original_data = preprocess_data(uploaded_file)
        
        # Display the original data
        st.write("### Given Data (Uploaded CSV)")
        st.dataframe(original_data)
        
        # Make predictions
        predictions = model.predict(original_data)
        
        # Display predictions
        st.write("### Predictions")
        st.write(predictions)
    except Exception as e:
        st.error(f"An error occurred: {e}")