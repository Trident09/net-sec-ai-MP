import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib  # For saving the trained model
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Argument parsing for dynamic file and model paths
parser = argparse.ArgumentParser(description="Train a Machine Learning Model on CICIDS 2017 Data")
parser.add_argument('--data_dir', type=str, default='MachineLearningCVE', help="Directory containing CSV files")
parser.add_argument('--model_path', type=str, default='model.pkl', help="Path to save the trained model")
args = parser.parse_args()

# Step 1: Load Data
# Step 1: Load Data
def load_data(data_dir):
    logging.info("Loading data...")
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Check if CSV files exist
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")
    
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Strip leading/trailing spaces from column names
            df.columns = df.columns.str.strip()
            
            dataframes.append(df)
            logging.info(f"Loaded {file} with shape {df.shape}")
        except Exception as e:
            logging.error(f"Failed to load {file}. Error: {str(e)}")
            continue
    
    df = pd.concat(dataframes, ignore_index=True)
    logging.info(f"Data loaded. Combined shape: {df.shape}")
    return df


# Step 2: Preprocess Data
# Step 2: Preprocess Data
def preprocess_data(df):
    # Specify the correct label column
    label_column = "Label"  # No leading space

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Check if the label column exists
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the dataset.")

    # Separate feature columns and label column
    feature_columns = [col for col in df.columns if col != label_column]

    # Handle missing values and infinite values
    df.replace([float('inf'), float('-inf')], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

    # Clip extremely large values if necessary (you can adjust the clipping range as needed)
    df = df.clip(-1e6, 1e6)

    # Encode the labels
    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

    # Scale the feature columns
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df, feature_columns, label_encoder



# Step 3: Train Model
def train_model(X_train, y_train):
    logging.info("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model

# Step 4: Evaluate Model
def evaluate_model(model, X_test, y_test, encoder):
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # For ROC-AUC if binary classification

    # Classification Report
    logging.info("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"Confusion Matrix:\n{cm}")
    
    # ROC-AUC (if binary classification)
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_prob)
        logging.info(f"ROC-AUC: {roc_auc:.2f}")
    
    return cm

# Step 5: Save Model
def save_model(model, model_path):
    logging.info(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    logging.info("Model saved.")

# Main Training Loop
if __name__ == "__main__":
    try:
        # Load and preprocess data
        data = load_data(args.data_dir)
        data, feature_columns, label_encoder = preprocess_data(data)
        
        # Split data into train and test sets
        X = data[feature_columns]
        y = data['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, label_encoder)
        
        # Save model
        save_model(model, args.model_path)
        
        logging.info("Training process completed.")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
