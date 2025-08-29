import os
import tqdm
import subprocess
import numpy as np
from itertools import product

# Define available model types
MODEL_CLASSES = ["ptsl2", "ssm", "diag_ssm", "pfsa", "wfsa", "sl2", "sp2", "soft_tsl2"]

# Define hyperparameters for tuning
HYPERPARAMETER_GRID = {
    "batch_size": [32],
    "num_epochs": [100],
    "lr": [0.001],
}

DATA_DIRECTORY = "data/converted_mlregtest/"

def get_directories():
    # Get directories in MLRegTest folder
    directories = []
    for filename in os.listdir(DATA_DIRECTORY):
        full_path = os.path.join(DATA_DIRECTORY, filename)
        if os.path.isdir(full_path):
            directories.append(full_path)
    return directories


# read the training and test files from the mlregtest directory, and then run this script to evaluate the models with different classe.
# plot the results to compare different models and hyperparameters.

# Create output directory if it doesn't exist
OUTPUT_DIR = "output/model_evaluations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to run evaluations for different models and hyperparameters
def run_evaluations(file_dict):
    # Loop through all model classes
    for model_type in MODEL_CLASSES:
        print(f"Evaluating model: {model_type}")

        # Iterate through combinations of hyperparameters
        for batch_size, num_epochs, lr in product(
            HYPERPARAMETER_GRID["batch_size"],
            HYPERPARAMETER_GRID["num_epochs"],
            HYPERPARAMETER_GRID["lr"],
        ):
            output_file = os.path.join(
                OUTPUT_DIR,
                f"{model_type}_bs{batch_size}_ep{num_epochs}_lr{lr}.txt",
            )

            # Run the model evaluation
            command = [
                "python3",
                "eval_model.py",  # Assuming eval_model.py runs training & evaluation
                model_type,
                file_dict['training'],
                file_dict['testing_paired'],
                "--batch_size", str(batch_size),
                "--num_epochs", str(num_epochs),
                "--lr", str(lr),
            ]

            print(f"Running: {command}")
            with open(output_file, "w") as f:
                subprocess.run(command, stdout=f)

            # TODO: can we save the results incrementally?

            print(f"Results saved to: {output_file}")

    print("All model evaluations completed.")

# Function to analyze results
def analyze_results():
    results = []

    for model_type in MODEL_CLASSES:
        for batch_size, num_epochs, lr in product(
            HYPERPARAMETER_GRID["batch_size"],
            HYPERPARAMETER_GRID["num_epochs"],
            HYPERPARAMETER_GRID["lr"],
        ):
            output_file = os.path.join(
                OUTPUT_DIR,
                f"{model_type}_bs{batch_size}_ep{num_epochs}_lr{lr}.txt",
            )

            # Read the loss values from the output files
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    for line in f:
                        if "Loss:" in line:
                            loss_value = float(line.split(":")[-1].strip())
                            results.append(
                                (model_type, batch_size, num_epochs, lr, loss_value)
                            )

    # Find the best model configuration
    best_model = min(results, key=lambda x: x[-1])
    print("\nBest Model Configuration:")
    print(f"Model: {best_model[0]}")
    print(f"Batch Size: {best_model[1]}")
    print(f"Epochs: {best_model[2]}")
    print(f"Learning Rate: {best_model[3]}")
    print(f"Loss: {best_model[4]:.4f}")

if __name__ == "__main__":
    directories = get_directories()
    for directory in directories:
        files = os.listdir(directory)
        file_dict = {}

        for file in files:
            if 'LearningData' in file:
                file_dict['training'] = os.path.join(directory, file)
            elif 'TestingPairs' in file:
                file_dict['testing_paired'] = os.path.join(directory, file)
            elif 'TestingUnpaired' in file:
                file_dict['testing_unpaired'] = os.path.join(directory, file)

        # Step 1: Run evaluations
        run_evaluations(file_dict)

        # Step 2: Analyze results
        analyze_results()
