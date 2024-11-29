# Traffic Prediction App

This is a **Streamlit web application** that allows users to upload a CSV file containing traffic data, preprocess it, and make predictions using a trained machine learning model. The app displays the input data along with the predicted labels and prediction probabilities (True/False) based on the model's output.

![Traffic Prediction App](https://i.imgur.com/vUSsa2O.png)

## Features

- Upload a CSV file containing traffic data.
- Preprocess the input data (scaling the features).
- Make predictions using a pre-trained machine learning model.
- Display the original uploaded CSV data.
- Show the predicted labels (True/False).
- Show prediction probabilities for each class (True/False).

## Requirements

- Python 3.x
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib

## Setup Instructions

1. **Clone the repository** or download the project files.

   ```bash
   git clone https://github.com/yourusername/traffic-prediction-app.git
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt` file, install the required packages individually:

   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

5. **Ensure you have the trained model** (`trained_model.pkl`) placed in the project directory. This file should contain the pre-trained model used for predictions.

6. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

   Replace `app.py` with the filename of your Streamlit script if different.

## Usage

Once the app is running:

1. **Upload a CSV file**: 
   - The app will prompt you to upload a CSV file containing traffic data.
   - The CSV should have multiple columns representing traffic features. The column names should match those that the model was trained on.

2. **See the uploaded data**: 
   - After uploading, the app will display the original CSV data in a table so you can verify the input.

3. **View Predictions**: 
   - The model will predict labels (True/False) based on the uploaded data and display them in a table.
   - You will also see prediction probabilities for both classes (True/False).

### Sample Output

- **Input Data (Uploaded CSV)**: A table showing the traffic features from the uploaded CSV file.
- **Predicted Labels**: A list showing whether the prediction for each data point is `True` or `False`.
- **Prediction Probabilities (True/False)**: A table displaying the probability for each class, where `True` is the predicted label and `False` is the alternative.

### Example CSV Data Format:

The CSV file should have the following structure (with traffic-related features):

| Destination Port | Flow Duration | Total Fwd Packets | Total Backward Packets | ... |
|------------------|---------------|-------------------|------------------------|-----|
| 80               | 1500          | 25                | 30                     | ... |
| 443              | 2000          | 40                | 50                     | ... |
| 21               | 1000          | 15                | 20                     | ... |

### Output Example:

| False Probability | True Probability | Predicted Label |
|-------------------|------------------|-----------------|
| 0.18              | 0.82             | True            |
| 0.09              | 0.91             | True            |
| 0.08              | 0.92             | True            |

## Model

This app uses a pre-trained model to make predictions based on traffic data. The model expects the input features to be in the same format and scale as the data used for training. The model is loaded using the `joblib` library from the file `trained_model.pkl`.

## Troubleshooting

- **CSV Format Errors**: Ensure that the uploaded CSV matches the format and feature columns expected by the model. If the CSV contains the `Label` column, it will be dropped automatically.
- **Model Errors**: If you encounter issues related to the model's predictions, verify that the model file (`trained_model.pkl`) is correctly placed in the project directory.

## License

This project is open-source and licensed under the [MIT License](../LICENSE).
