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
        # st.success(f"Model downloaded successfully to `{local_path}`.")
    else:
        # st.info(f"Model already exists at `{local_path}`.")
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

# Define prediction labels
prediction_labels = ["Benign", "DDoS"]

# Streamlit app
st.set_page_config(
    page_title="Traffic Prediction App",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add meta tags and OG image using raw HTML
st.markdown(
    """
    <meta name="description" content="Predict network traffic patterns as either benign or DDoS using AI-powered insights.">
    <meta property="og:image" content="https://i.imgur.com/I7HhUVF.png">
    """,
    unsafe_allow_html=True,
)

# Web app title and description
st.title("Traffic Prediction App 🌐")
st.markdown(
    """
    Welcome to the **Traffic Prediction App**! 
    This tool uses a neural network model to predict whether network traffic is **benign** or **malicious (DDoS)**. 
    Upload your dataset as a CSV file and gain insights into your network traffic.
    """
)

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    try:
        # Preprocess the CSV data
        df = pd.read_csv(uploaded_file).drop(columns=['Label'], errors='ignore')
        
        # Display the original data
        st.write("### Uploaded Data Preview")
        st.dataframe(df)

        # Make predictions
        raw_predictions = model.predict(df)
        mapped_predictions = [prediction_labels[int(round(pred[0]))] for pred in raw_predictions]
        
        # Add predictions to the DataFrame
        df['Prediction'] = mapped_predictions
        
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
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Upload a CSV file to start the prediction process.")
