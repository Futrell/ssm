import os
import glob
import subprocess

# Default paths
DATA_PATH = "data/mlregtest"
OUTPUT_DIR = "output/mlregtest"

# Prompt for custom data path, otherwise use default
data_path_input = input(f"Enter the path to the directory containing train and test files (default: {DATA_PATH}): ").strip()
if data_path_input:
    DATA_PATH = data_path_input

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all train files (e.g., files ending with '_Train.txt')
TRAIN_FILES = glob.glob(os.path.join(DATA_PATH, "*_Train.txt"))

# Find all test files (e.g., files ending with '_Test*.txt')
TEST_FILES = glob.glob(os.path.join(DATA_PATH, "*_Test*.txt"))

# Iterate over each train file
for train_file in TRAIN_FILES:
    # Extract the train file name without the path
    train_name = os.path.basename(train_file).replace(".txt", "")

    # Iterate over each test file
    for test_file in TEST_FILES:

        # Extract the test file name without the path
        test_name = os.path.basename(test_file).replace(".txt", "")

        # Create a unique output file
        output_file = os.path.join(OUTPUT_DIR, f"{train_name}_{test_name}_output.txt")

        # Run the evaluation command and save the output
        print(f"Running eval_model.py with {train_file} and {test_file}")
        with open(output_file, "w") as outfile:
            subprocess.run(["python", "eval_model.py", "sl2", train_file, test_file], stdout=outfile)

        print(f"Output saved to {output_file}")

print(f"All evaluations completed. Outputs are in the '{OUTPUT_DIR}' folder.")

# TODO: only want paired test files. start with random test files
# also hyperparameter tuning. Find the best hyperparameters for each model based on a randomly chosen train-test pair
# 