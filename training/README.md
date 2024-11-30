# **Training Scripts Subfolder**

This folder contains all necessary files and scripts to train a neural network model for the project.

---

## **Contents**

1. **`train_neural_network.py`**  
   The main script for training the neural network model. It includes:
   - Data loading from multiple CSV files
   - Data exploration and preprocessing
   - Neural network training using TensorFlow and Keras
   - Model evaluation and visualization
   - Saving the trained model for future use

2. **Dependencies**  
   The following Python libraries are required:
   - `pandas`: For data loading and manipulation
   - `numpy`: For numerical computations
   - `scikit-learn`: For preprocessing, encoding, and splitting the data
   - `tensorflow`: For building and training the neural network
   - `matplotlib` and `seaborn`: For data and result visualization
   - `logging`: For runtime logs and debugging

3. **Expected Input**  
   The script expects a directory containing CSV files. Each file should have the necessary features and target labels. Ensure the data meets these requirements:
   - **File format**: `.csv`
   - **Features and Labels**: Clearly defined columns for predictors and target variables
   - **Missing values**: Handled within the script, but clean data is recommended

4. **Output**  
   The training script produces:
   - A trained model saved as a `.keras` file
   - Visualizations of training and validation accuracy/loss
   - A confusion matrix for evaluating classification performance

---

## **Setup**

### **Step 1: Install Dependencies**
Ensure Python 3.8 or later is installed. Install the required libraries:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

For a virtual environment:
```bash
# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows

# Install dependencies
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

### **Step 2: Prepare the Dataset**
- Place all CSV files containing the dataset in a single directory.
- Ensure the directory path is correctly specified in the script (default: `/content/drive/MyDrive/Colab Notebooks/Files/`).

---

## **How to Run the Script**

Click play on the code cell in the `Network_minor_neuralnetwork.ipynb` notebook to run the training script.

### **Optional Arguments**
- `--data_dir`: Path to the directory containing CSV files (default: `/content/drive/MyDrive/Colab Notebooks/Files/`).

---

## **File Descriptions**

- **`Network_minor_neuralnetwork.ipynb`**:  
  Contains the complete pipeline for loading data, preprocessing, building, training, and evaluating a neural network.
- **`trained_model-neuralnetwork.keras`**:  
  Saved neural network model for reuse.
- **`logs`** (optional):  
  Contains runtime logs if logging is enabled.

---

## **Troubleshooting**

### **Common Errors**
1. **Missing Dependencies**:  
   Ensure required libraries are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **No CSV Files Found**:  
   Verify the directory contains CSV files and the path is correctly specified.

3. **Invalid Data Format**:  
   Check that CSV files have the required features and target labels. Ensure column names are consistent.

### **Debugging**
Use the `logging` module to trace the script's execution. Errors and exceptions are logged to the console.

---

## **Future Enhancements**
1. Add hyperparameter tuning support using tools like `Keras Tuner`.
2. Enable logging of training metrics to a file or dashboard (e.g., TensorBoard).
3. Expand support for multi-label or multi-class classification tasks.
4. Provide options for model architecture customization.

---