import os
import subprocess
from itertools import product
import argparse

from run_model import HYPERPARAMETER_GRID, OUTPUT_DIR

TURKISH_DATA_DIRECTORY = os.path.join("data", "turkish")
ORIG_TRAIN_FILE = os.path.join(TURKISH_DATA_DIRECTORY, "turkish_train.txt")
ORIG_DEV_FILE = os.path.join(TURKISH_DATA_DIRECTORY, "turkish_dev.txt")
ORIG_TEST_FILE = os.path.join(TURKISH_DATA_DIRECTORY, "turkish_test_data.csv")
CLEAN_TRAIN_FILE = os.path.join(TURKISH_DATA_DIRECTORY, "turkish_train_clean.txt")
CLEAN_DEV_FILE = os.path.join(TURKISH_DATA_DIRECTORY, "turkish_dev_clean.txt")
CLEAN_TEST_SCORE_FILE = os.path.join(TURKISH_DATA_DIRECTORY, "turkish_test_clean.csv")
CLEAN_TEST_Z_SCORE_FILE = os.path.join(TURKISH_DATA_DIRECTORY, "turkish_test_z_clean.csv")

symbol_mapping = {
    'd͡ʒ': 'dʒ',
    'c': 'k',
    'g': 'ɡ',
    't͡ʃ': 'tʃ',
    'ɫ': 'l',
    'ɾ': 'r'
}

def clean_data_files():
    # Clean training file
    with open(ORIG_TRAIN_FILE, 'r', encoding='utf-8') as infile, open(CLEAN_TRAIN_FILE, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(line.strip() + "\n")
    
    # Clean dev file
    with open(ORIG_DEV_FILE, 'r', encoding='utf-8') as infile, open(CLEAN_DEV_FILE, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Skip header line
            if line.strip() == 'form':
                continue
            outfile.write(line.strip() + "\n")

    # Clean test file and split into mean score and mean z-score files
    with open(ORIG_TEST_FILE, 'r', encoding='utf-8') as infile, \
        open(CLEAN_TEST_SCORE_FILE, 'w', encoding='utf-8') as outfile1, \
        open(CLEAN_TEST_Z_SCORE_FILE, 'w', encoding='utf-8') as outfile2:
        for line in infile:
            tokens = line.strip().split(',')
            word, z_score, score = tokens
            for old, new in symbol_mapping.items():
                word = word.replace(old, new)
            # Write score to one file and z-score to another file, both with the word as the first column
            outfile1.write(f"{word},{score}\n")
            outfile2.write(f"{word},{z_score}\n")

def run_turkish_evaluations(model_type, use_z_score=False, redo_clean=False):
    for file in [CLEAN_TRAIN_FILE, CLEAN_DEV_FILE, CLEAN_TEST_SCORE_FILE, CLEAN_TEST_Z_SCORE_FILE]:
        if redo_clean or not os.path.exists(file):
            clean_data_files()
            break

    output_folder = os.path.join(OUTPUT_DIR, TURKISH_DATA_DIRECTORY, model_type, "z_score" if use_z_score else "score")
    os.makedirs(output_folder, exist_ok=True)

    for batch_size, num_epochs, lr in product(
        HYPERPARAMETER_GRID["batch_size"],
        HYPERPARAMETER_GRID["num_epochs"],
        HYPERPARAMETER_GRID["lr"],
    ):
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
            CLEAN_TRAIN_FILE,
            CLEAN_TEST_Z_SCORE_FILE if use_z_score else CLEAN_TEST_SCORE_FILE,
            "--batch_size", str(batch_size),
            "--num_epochs", str(num_epochs),
            "--lr", str(lr),
            "--save_checkpoints",
            "--report_every", "10",
            "--checkpoint_filename", model_string,
            "--checkpoint_folder", output_folder,
            "--char_separator", ' ',
            "--col_separator", ',',
            "--dev_file", CLEAN_DEV_FILE,
            "--numerical_eval",
            "--no_header",
            "--no_filter_training_grammatical"
        ]

        print(f"Running: {command}")
        with open(output_file, "w") as f:
            # pipe stdout into the file f
            subprocess.run(command, stdout=f)

        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run class of phonotactic models on specific natural language datasets")
    parser.add_argument('model_class', type=str, help="Model class to evaluate")
    parser.add_argument('--dataset', type=str, default='turkish', help="Dataset to use for running model")
    
    args = parser.parse_args()

    if args.dataset == 'turkish':
        run_turkish_evaluations(args.model_class.lower(), use_z_score=False)
        run_turkish_evaluations(args.model_class.lower(), use_z_score=True)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented for natural language evaluations yet.")