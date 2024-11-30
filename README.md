# Traffic Prediction Project

This project aims to predict traffic patterns based on network traffic data using machine learning. The project is organized into three main directories:

1. **Dataset**: Contains the CICIDS dataset used for model training and testing.
2. **Frontend**: Includes the Streamlit app for predicting traffic patterns, the trained model (`trained_model-neuralnetwork.keras`).
3. **Training**: Contains the scripts and notebooks used to train the model, including data preprocessing, feature engineering, model training, and evaluation (mainly in Google Colab and Jupyter notebooks).

## Project Overview

The goal of this project is to predict traffic patterns (either "True" or "False") based on the features of network traffic. The application provides the following:

1. **Training Pipeline**: A training pipeline that processes the dataset, trains a model, and saves the trained model for later predictions.
2. **Prediction App**: A user-friendly web application built with Streamlit that allows users to upload CSV files containing network traffic data and get predictions using the pre-trained model.

## Folder Structure

```plaintext
traffic-prediction-project/
├── Dataset/
│   ├── CICIDS_2023_traffic_data.csv      # Network traffic dataset
│   ├── ...                               # Other data files
│
├── Frontend/
│   ├── streamlit-app.py                  # Streamlit app for predictions
│   ├── trained_model-neuralnetwork.keras # Pre-trained model file
│   ├── other model files                 # Random forest and standardisation model files
│   ├── ...                               # Other frontend assets
│
├── Training/
│   ├── Minor_minor_neuralnetwork.ipynb        # Google Colab/Jupyter notebook for training the model
│   ├── ...                               # Other training scripts
```

## How the Project Works

### 1. **Dataset Folder**:
   - This folder contains the **CICIDS** dataset, which includes network traffic data used to train and test the model. The dataset contains various features such as packet length, flow duration, and flags, which are used to predict traffic anomalies or classify traffic.

### 2. **Training Folder**:

   - This folder contains the scripts and notebooks used for training the model. The data preprocessing steps, feature engineering, model training, and evaluation are implemented here. The trained model and scaler are saved to files (`trained_model-neuralnetwork.keras`), which is then used by the Streamlit frontend for predictions.
      - You can run the Jupyter notebook (`Network_minor_neuralnetwork.ipynb`) to reproduce the training pipeline to train the model from scratch.

### 3. **Frontend Folder**:
   - This folder contains the **Streamlit app** (`app.py`), which is the interface where users can upload CSV files containing traffic data. The app uses the trained model (`trained_model-neuralnetwork.keras`) to predict labels and display prediction probabilities.
   - The app shows the uploaded data and the predictions in real-time, including the predicted labels and the probability for each class.

## Technologies Used
- **Python**: The primary programming language used for this project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and array operations.
- **Scikit-learn**: For machine learning models, preprocessing, and model evaluation.
- **Joblib**: For saving and loading the trained model (if applicable).
- **Streamlit**: For building the web application for predictions.
- **Google Colab/Jupyter Notebooks**: For data preprocessing, feature engineering, and model training.
- **Matplotlib/Seaborn**: For data visualization.
- **TensorFlow**: For building and training the neural network model.
- **Keras**: For defining and managing the neural network layers and training process.
- **Requests**: For making HTTP requests if needed (e.g., for API calls).
- **Scikit-learn (train_test_split, LabelEncoder, StandardScaler)**: For splitting the data, label encoding, and standardizing features.
- **Confusion Matrix (from sklearn.metrics)**: For evaluating model performance.

## Presentation

<a href="https:&#x2F;&#x2F;www.canva.com&#x2F;design&#x2F;DAGXzGEtAIw&#x2F;s2H8wPDRKJofjz-34z51vQ&#x2F;view?utm_content=DAGXzGEtAIw&amp;utm_campaign=designshare&amp;utm_medium=embeds&amp;utm_source=link" target="_blank" rel="noopener">Improve-Network-Security-with-Artificial-Intelligence</a> by Rupam Barui

## How to Set Up the Project

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/Trident09/net-sec-ai-MP.git
```

### 2. Set Up the Environment

It is recommended to use a virtual environment to install the required dependencies.

- Navigate to the project directory:

  ```bash
  cd net-sec-ai-MP
  ```

- Create a virtual environment:

  ```bash
  python -m venv venv
  ```

- Activate the virtual environment:

  - On Windows:

    ```bash
    venv\Scripts\activate
    ```

  - On macOS/Linux:

    ```bash
    source venv/bin/activate
    ```

### 3. Install Dependencies

Install the required dependencies for the project in each folder according to need:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the necessary libraries manually if `requirements.txt` is not provided:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### 4. Train the Model (if not already trained)

If you haven't yet trained the model, you can do so by following these steps:

1. Navigate to the `Training` folder.
2. Open the Jupyter notebook `Network_minor_neuralnetwork.ipynb` 
3. The model will be trained on the CICIDS dataset, and the trained model (`trained_model-neuralnetwork.keras`) will be saved.


The trained files are saved in the `Frontend` folder for use in the Streamlit app.

### 5. Run the Streamlit App

Once everything is set up, you can start the Streamlit app for predictions:

```bash
streamlit run Frontend/streamlit-app.py
```

This will open the app in your web browser.

## How to Use the Traffic Prediction App

1. **Upload a CSV File**: 
   - Click on the "Choose a CSV file" button in the Streamlit app and upload a CSV file containing traffic data. The CSV file should contain network traffic features (e.g., flow duration, packet lengths, flags, etc.).
   
   - The app will display the contents of the uploaded file to ensure the data is correctly loaded.

2. **Get Predictions**: 
   - Once the CSV file is uploaded, the app will use the trained model to predict the traffic labels (True/False).
   
   - The app will display:
     - The original uploaded data.
     - Predicted labels for each row.
     - Prediction probabilities (True/False) for each class.

3. **View Results**:
   - After the predictions are made, the app will show:
     - **Predicted Labels**: Whether the traffic is classified as `True` or `False` based on the model.
     - **Prediction Probabilities (True/False)**: Probabilities for each class indicating the confidence of the prediction.

## Example CSV Format

The CSV file should have the following columns (without the label column):

| Destination Port | Flow Duration | Total Fwd Packets | Total Backward Packets | ... |
|------------------|---------------|-------------------|------------------------|-----|
| 80               | 1500          | 25                | 30                     | ... |
| 443              | 2000          | 40                | 50                     | ... |
| 21               | 1000          | 15                | 20                     | ... |

### Sample Output:

| False Probability | True Probability | Predicted Label |
|-------------------|------------------|-----------------|
| 0.18              | 0.82             | True            |
| 0.09              | 0.91             | True            |
| 0.08              | 0.92             | True            |

## Troubleshooting

- **CSV Format Errors**: Ensure the uploaded CSV contains the necessary features and follows the correct format. The app will drop the label column automatically if present.
- **Model Errors**: Ensure the model and scaler are correctly saved in the `Frontend` folder. The app expects these files to be present for making predictions.

## License

This project is open-source and licensed under the [MIT License](LICENSE).

---