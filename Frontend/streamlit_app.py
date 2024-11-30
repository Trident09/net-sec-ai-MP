import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib
import os
import requests

# Function to download the model file
def download_model(url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(response.content)

# Load the trained model
model_url = "https://raw.githubusercontent.com/Trident09/net-sec-ai-MP/e81411df26baf2fa0d731d276d93f973b5cee72d/Frontend/trained_model-randomforest.pkl"
model_path = "trained_model-randomforest.pkl"
download_model(model_url, model_path)
model = joblib.load(model_path)

# Function to load and preprocess the CSV file
def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop(columns=['Label'], errors='ignore')  # Drop 'Label' column if present
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, df_scaled

# Streamlit app
st.title('Traffic Prediction App')
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Preprocess the CSV data
        original_data, processed_data = preprocess_data(uploaded_file)
        
        # Display the original data
        st.write("### Given Data (Uploaded CSV)")
        st.dataframe(original_data)
        
        # Make predictions
        predictions = model.predict(processed_data)
        prediction_probs = model.predict_proba(processed_data)
        
        # Display predictions
        st.write("### Predictions")
        st.write(predictions)
        
        st.write("### Prediction Probabilities")
        st.write(prediction_probs)
    except Exception as e:
        st.error(f"An error occurred: {e}")
