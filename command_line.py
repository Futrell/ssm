import os
import tqdm
import subprocess
import numpy as np
from itertools import product

# Define available model types
MODEL_CLASSES = ["sl2", "sp2", "soft_tsl2", "ptsl2", "ssm", "pfsa", "wfsa"]

# Define hyperparameters for tuning
HYPERPARAMETER_GRID = {
    "batch_size": [1, 32, 128, 512],
    "num_epochs": [1],
    "lr": [0.001, 0.01, 0.1],
}

training_file = "data/mlregtest/04.04.SL.2.1.0_Dev.txt" # TODO:
test_file = "data/mlregtest/04.04.SL.2.1.0_TestLR.txt" #

# Create output directory if it doesn't exist
OUTPUT_DIR = "output/model_evaluations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to run evaluations for different models and hyperparameters
def run_evaluations():
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
                "python",
                "eval_model.py",  # Assuming eval_model.py runs training & evaluation
                model_type,
                training_file,
                test_file,
                "--batch_size", str(batch_size),
                "--num_epochs", str(num_epochs),
                "--lr", str(lr),
            ]

            print(f"Running: {command}")
            with open(output_file, "w") as f:
                subprocess.run(command, stdout=f)

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
    # Step 1: Run evaluations
    run_evaluations()

    # Step 2: Analyze results
    analyze_results()
