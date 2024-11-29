import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model
import joblib
model = joblib.load('trained_model.pkl')

# Function to load and preprocess the CSV file
def preprocess_data(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Drop the label column if present (since it's not part of the input features)
    df = df.drop(columns=['Label'], errors='ignore')  # 'Label' might not be present in your input file
    
    # You may need to standardize/scale your input data (match the preprocessing during training)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)  # Apply the scaling
    
    return df, df_scaled

# Streamlit file uploader to load CSV
st.title('Traffic Prediction App')
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Preprocess the CSV data
    original_data, processed_data = preprocess_data(uploaded_file)
    
    # Reshape the input if necessary (make sure it's 2D)
    if processed_data.ndim == 3:  # If it's 3D, reshape it
        processed_data = processed_data.reshape(-1, processed_data.shape[-1])
        
    # Display the original uploaded data
    st.write("### Given Data (Uploaded CSV)")
    st.dataframe(original_data)
    
    # Make predictions using the model
    try:
        predictions = model.predict(processed_data)
        prediction_probs = model.predict_proba(processed_data)
        
        # Display the predictions and probabilities
        st.write("Predicted Labels")
        st.write(predictions)
        
        st.write("Prediction Probabilities (True/False)")
        st.write(prediction_probs)
        
    except Exception as e:
        st.write(f"Error during prediction: {e}")

