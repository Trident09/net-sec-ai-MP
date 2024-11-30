import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('network_traffic_model.keras')

# Define a function for prediction
def predict(features):
    predictions = model.predict(features)
    return ['BENIGN' if pred < 0.5 else 'DDoS' for pred in predictions]

# Streamlit app
st.title('Network Traffic Classification')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.write("Uploaded Data:")
    st.dataframe(data.head())
    
    # Check if all necessary columns are present
    required_columns = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
        'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags'
    ]

    if not all(col in data.columns for col in required_columns):
        st.error("Uploaded CSV does not contain all required columns.")
    else:
        # Select only the required columns
        data = data[required_columns]
        
        # Feature Scaling
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Predict using the uploaded data
        predictions = predict(data_scaled)
        
        # Add predictions to the dataframe
        data['Prediction'] = predictions
        
        # Display the results
        st.write("Predictions:")
        st.dataframe(data)