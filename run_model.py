import os
import tqdm
import subprocess
import numpy as np
from itertools import product

# Define available model types
MODEL_CLASSES = [
    "ptsl2",
    "ssm",
    "diag_ssm",
    "pfsa",
    "wfsa",
    "sl2",
    "sp2",
    "soft_tsl2",
]

# Define hyperparameters for tuning
HYPERPARAMETER_GRID = {
    "batch_size": [1, 32, 128, 1024],
    "num_epochs": [10],
    "lr": [0.01, 0.001, 0.0001],
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
    basename = os.path.dirname(file_dict['training']).split('/')[-1]
    # Loop through all model classes
    for model_type in MODEL_CLASSES:
        print(f"Evaluating model: {model_type}")

        # Iterate through combinations of hyperparameters
        for batch_size, num_epochs, lr in product(
            HYPERPARAMETER_GRID["batch_size"],
            HYPERPARAMETER_GRID["num_epochs"],
            HYPERPARAMETER_GRID["lr"],
        ):
            output_folder = os.path.join(OUTPUT_DIR, basename, model_type)
            os.makedirs(output_folder, exist_ok=True)

            model_string = f"{model_type}_bs{batch_size}_ep{num_epochs}_lr{lr}"
            output_file = os.path.join(
                output_folder,
                f"{model_string}.txt",
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
                "--save_checkpoints",
                "--checkpoint_prefix", model_string,
                "--checkpoint_folder", output_folder
            ]

            print(f"Running: {command}")
            with open(output_file, "w") as f:
                # pipe stdout into the file f
                subprocess.run(command, stdout=f)

            print(f"Results saved to: {output_file}")

    print("All model evaluations completed.")


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

        run_evaluations(file_dict)

