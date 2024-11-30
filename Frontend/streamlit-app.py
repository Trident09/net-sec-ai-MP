import os
import requests
import tensorflow as tf
import streamlit as st
import pandas as pd

# Function to download the model file
def download_model(url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(url)
        response.raise_for_status()  # Ensure successful download
        with open(local_path, 'wb') as f:
            f.write(response.content)
        # st.write(f"Model downloaded to {local_path}")
    else:
        # st.write(f"Model already exists at {local_path}")
        pass

# Set model URL and local path
model_url = "https://raw.githubusercontent.com/Trident09/net-sec-ai-MP/b1aff8fd96776446ae8b5fe959cd5adf7eb65f5c/Frontend/trained_model-neuralnetwork.keras"
model_path = "trained_model-neuralnetwork.keras"

# Download and load the model
download_model(model_url, model_path)

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("Failed to download the model. Please check the URL or file path.")

# Streamlit app
st.title('Traffic Prediction App')
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Preprocess the CSV data
        df = pd.read_csv(uploaded_file).drop(columns=['Label'], errors='ignore')
        
        # Display the original data
        st.write("### Uploaded Data")
        st.dataframe(df)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Display predictions
        st.write("### Predictions")
        st.dataframe(df[['Prediction']])
        
        # Downloadable result
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
