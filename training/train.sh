#!/bin/bash

# A shell script to run the training script with specified arguments

# Define default paths
DATA_DIR="../dataset/CICIDS2017_CSVs/MachineLearningCVE/"
MODEL_PATH="trained_model.pkl"

# Run the Python script with the provided paths
python train.py --data_dir "$DATA_DIR" --model_path "$MODEL_PATH"

# Optional: Add execution status check
if [ $? -eq 0 ]; then
    echo "Training completed successfully. The model is saved at $MODEL_PATH."
else
    echo "An error occurred during training."
    exit 1
fi
