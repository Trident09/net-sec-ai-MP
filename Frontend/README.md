# Traffic Prediction App 🌐

This is a **Streamlit web application** that allows users to upload a CSV file containing traffic data, preprocess it, and make predictions using a pre-trained neural network model. The app categorizes network traffic as **Benign** or **DDoS** based on the model's output and displays user-friendly prediction results.

![Traffic Prediction App](https://i.imgur.com/cx0crtQ.png)

---

## Features

- **Upload CSV Files**: Supports uploading traffic datasets in CSV format.
- **Preprocessing**: Automatically drops unnecessary columns like `Label` and prepares the data for prediction.
- **Predictions**: Uses a neural network model to classify traffic as **Benign** or **DDoS**.
- **Interactive Data Display**:
  - View uploaded data directly in the app.
  - See predictions alongside the original dataset.
- **Download Results**: Provides an option to download the predictions as a CSV file.

---

## Requirements

- Python 3.x
- Streamlit
- pandas
- numpy
- TensorFlow
- requests

---

## Setup Instructions

1. **Clone the Repository**

2. **Navigate to the Project Directory**:

   ```bash
   cd Frontend
   ```

3. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   ```

4. **Activate the Virtual Environment**:

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

6. **Download the Pre-trained Model**:

   The app automatically downloads the pre-trained neural network model (`trained_model-neuralnetwork.keras`) when you run it for the first time.

7. **Run the App**:

   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Upload a CSV File**:
   - The CSV file should contain traffic data features.
   - The `Label` column (if present) will be ignored.

2. **View Uploaded Data**:
   - Once uploaded, the app displays the dataset for review.

3. **Get Predictions**:
   - The app predicts whether the traffic is **Benign** or **DDoS**.
   - Predictions are displayed in the app alongside the input data.

4. **Download Results**:
   - Export the predictions as a CSV file for further analysis.

---

### Example CSV Format

The input CSV should have a structure like this:

| Destination Port | Flow Duration | Total Fwd Packets | Total Backward Packets | ... |
|------------------|---------------|-------------------|------------------------|-----|
| 80               | 1500          | 25                | 30                     | ... |
| 443              | 2000          | 40                | 50                     | ... |
| 21               | 1000          | 15                | 20                     | ... |

---

## Sample Output

| Prediction       | Probability (Benign) | Probability (DDoS) |
|------------------|-----------------------|---------------------|
| Benign           | 0.91                 | 0.09                |
| DDoS             | 0.02                 | 0.98                |
| Benign           | 0.87                 | 0.13                |

---

## Model Details

The app uses a pre-trained neural network model stored as `trained_model-neuralnetwork.keras`. It is downloaded directly from a remote repository during runtime if not already available locally.

---

## Troubleshooting

- **CSV Format Issues**: Ensure the uploaded file has the correct structure and contains valid traffic feature columns. Columns unrelated to the model's input are ignored automatically.
- **Model Loading Issues**: If the app cannot download the model, verify the URL in the code or check your internet connection.

---

## License

This project is licensed under the [MIT License](../LICENSE).