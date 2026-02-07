import os
import subprocess
from itertools import product
import argparse
from collections import defaultdict

# Define available model types
MODEL_CLASSES = [
    "ptsl2",
    "diag_ssm",
    "pfsa",
    "wfsa",
    "sl2",
    "sp2",
    "soft_tsl2",
]

# Define hyperparameters for tuning
HYPERPARAMETER_GRID = {
    "batch_size": [1, 2, 4, 16, 32],
    "num_epochs": [10],
    "lr": [0.001, 0.01]
}

DATA_DIRECTORY = "data/converted_mlregtest/"
ORIG_DATA_DIRECTORY = os.path.join("data", "mlregtest")

def get_directories():
    # Get directories in MLRegTest folder
    directories = []
    for filename in os.listdir(DATA_DIRECTORY):
        full_path = os.path.join(DATA_DIRECTORY, filename)
        if os.path.isdir(full_path):
            directories.append(full_path)
    return directories

def combine_test_files(file1, file2, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as outfile:
        for fname in [file1, file2]:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    return output_file


# read the training and test files from the mlregtest directory, and then run this script to evaluate the models with different classe.
# plot the results to compare different models and hyperparameters.

# Create output directory if it doesn't exist
OUTPUT_DIR = "output/model_evaluations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMP_TEST_DIR = "output/temp_tests"
os.makedirs(TEMP_TEST_DIR, exist_ok=True)

# Function to run evaluations for different models and hyperparameters
def run_evaluations(file_dict, model_types=MODEL_CLASSES):
    basename = os.path.dirname(file_dict['training'])
    class_type = 'LSA' if 'LSA' in file_dict['testing_paired'] else 'LSR'
    file_details = os.path.splitext(os.path.basename(file_dict['training']))[0].rsplit('_', 1)[0] + '_' + class_type
    # Loop through all model types
    for model_type in model_types:
        print(f"Evaluating model: {model_type}")

        # Iterate through combinations of hyperparameters
        for batch_size, num_epochs, lr in product(
            HYPERPARAMETER_GRID["batch_size"],
            HYPERPARAMETER_GRID["num_epochs"],
            HYPERPARAMETER_GRID["lr"],
        ):
            output_folder = os.path.join(OUTPUT_DIR, basename, model_type, file_details)
            os.makedirs(output_folder, exist_ok=True)

            model_string = f"{model_type}_bs{batch_size}_ep{num_epochs}_lr{lr}"
            output_file = os.path.join(
                output_folder,
                f"{model_string}.txt",
            )
            # Skip if results already exist
            if os.path.exists(output_file):
                continue  

            # Run the model evaluation
            command = [
                "python",
                "eval_model.py",  # Assuming eval_model.py runs training & evaluation
                model_type,
                file_dict['training'],
                file_dict['testing_paired'],
                "--batch_size", str(batch_size),
                "--num_epochs", str(num_epochs),
                "--lr", str(lr),
                "--save_checkpoints",
                "--report_every", "10",
                "--checkpoint_filename", model_string,
                "--checkpoint_folder", output_folder,
                "--char_separator", '',
                "--dev_file", file_dict['dev']
            ]

            print(f"Running: {command}")
            with open(output_file, "w") as f:
                # pipe stdout into the file f
                subprocess.run(command, stdout=f)

            print(f"Results saved to: {output_file}")

    print("All model evaluations completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run class of phonotactic models on specific datasets")
    parser.add_argument('model_class', type=str, help="Model class to evaluate")
    parser.add_argument('lang_class', type=str, help="Language class for datasets")
    parser.add_argument('alpha_size', type=str,
                    help="Alphabet size for data in string representation of two-digit integer")
    parser.add_argument('window_size', type=str,
                        help="Width of factors in string representation of one-digit integer")
    
    args = parser.parse_args()

    groupings = dict(defaultdict(list))

    alpha_sizes = args.alpha_size.split(',')
    window_sizes = args.window_size.split(',')

    for file in os.listdir(ORIG_DATA_DIRECTORY):
        for alpha_size in alpha_sizes:
            for window_size in window_sizes:
                tokens = file.split('.')
                file_alpha_size, _, file_lang_class, file_window_size, _, ind_data_type, _ = tokens
                file_index, file_split = ind_data_type.split('_')

                if file_alpha_size != alpha_size or file_lang_class != args.lang_class or file_window_size != window_size:
                    continue

                grouping_key = '.'.join([file_lang_class, file_alpha_size, file_window_size, file_index])
                if grouping_key not in groupings:
                    groupings[grouping_key] = defaultdict(list)
                if file_split == "Train":
                    groupings[grouping_key]['training'].append(os.path.join(ORIG_DATA_DIRECTORY, file))
                elif 'Test' in file_split:
                    groupings[grouping_key]['test'].append(os.path.join(ORIG_DATA_DIRECTORY, file))
                elif 'Dev' in file_split:
                    groupings[grouping_key]['dev'].append(os.path.join(ORIG_DATA_DIRECTORY, file))
    
    for setting_str, group in groupings.items():
        train_file = group["training"][0]
        dev_file = group["dev"][0]

        # Identify LA and SA test files
        la_file = [f for f in group["test"] if "TestLA" in f][0]
        sa_file = [f for f in group["test"] if "TestSA" in f][0]

        # Identify LR and SR test files
        lr_file = [f for f in group["test"] if "TestLR" in f][0]
        sr_file = [f for f in group["test"] if "TestSR" in f][0]

        # Construct name for combined file
        idx = setting_str.split(".")[-1]  # extract the final index
        combined_test_file_a = os.path.join(TEMP_TEST_DIR, f"{setting_str}_combined_test_LSA_{idx}.txt")
        combined_test_file_r = os.path.join(TEMP_TEST_DIR, f"{setting_str}_combined_test_LSR_{idx}.txt")

        # Create combined test file
        combine_test_files(la_file, sa_file, combined_test_file_a)
        combine_test_files(lr_file, sr_file, combined_test_file_r)

        for test_file in [combined_test_file_a, combined_test_file_r]:
            file_dict = {
                'training' : train_file, 
                'testing_paired' : test_file,
                'dev': dev_file
            }
            run_evaluations(file_dict, model_types=[args.model_class.lower()])