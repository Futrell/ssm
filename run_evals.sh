#!/usr/bin/env bash
set -euo pipefail

# --- config mirroring your Python script ---
DATA_DIRECTORY="data/converted_mlregtest"
OUTPUT_DIR="output/model_evaluations"

# Model classes
MODEL_CLASSES=(
  "ptsl2"
  "diag_ssm"
  "pfsa"
  "wfsa"
  "sl2"
  "sp2"
  "soft_tsl2"
)

# Hyperparameter grid
BATCH_SIZES=(32 128 1024)
NUM_EPOCHS=(10)
LRS=(0.001)

# Path to the evaluator script (adjust if needed)
EVAL_SCRIPT="eval_model.py"

mkdir -p "$OUTPUT_DIR"

echo "Scanning datasets under: $DATA_DIRECTORY"
echo

# Iterate over immediate subdirectories of DATA_DIRECTORY
# The glob pattern */ gives only directories; it’s fine if none exist.
shopt -s nullglob
for DIR in $DATA_DIRECTORY/*/; do
  # Normalize DIR (remove trailing slash for basename)
  DIR="${DIR%/}"
  BASENAME="$(basename "$DIR")"
  echo ">> Dataset: $BASENAME"

  # Find the first (alphabetically) file matching each pattern
  # Note: This avoids mapfile and NUL-delimited pipelines; robust enough for typical filenames.
  training="$(find "$DIR" -maxdepth 1 -type f -name '*LearningData*'  | LC_ALL=C sort | sed -n '1p')"
  testing_paired="$(find "$DIR" -maxdepth 1 -type f -name '*TestingPairs*' | LC_ALL=C sort | sed -n '1p')"
  testing_unpaired="$(find "$DIR" -maxdepth 1 -type f -name '*TestingUnpaired*' | LC_ALL=C sort | sed -n '1p')"

  if [[ -z "$training" || -z "$testing_paired" ]]; then
    echo "   Skipping: required files not found."
    echo "   training:        ${training:-<none>}"
    echo "   testing_paired:  ${testing_paired:-<none>}"
    echo
    continue
  fi

  for model_type in "${MODEL_CLASSES[@]}"; do
    echo "   Model: $model_type"

    outdir="$OUTPUT_DIR/$BASENAME/$model_type"
    mkdir -p "$outdir"

    for bs in "${BATCH_SIZES[@]}"; do
      for ne in "${NUM_EPOCHS[@]}"; do
        for lr in "${LRS[@]}"; do
          model_string="${model_type}_bs${bs}_ep${ne}_lr${lr}"
          outfile="$outdir/${model_string}.txt"

          cmd=(python "$EVAL_SCRIPT"
               "$model_type"
               "$training"
               "$testing_paired"
               --batch_size "$bs"
               --num_epochs "$ne"
               --lr "$lr"
               --save_checkpoints
               --checkpoint_filename "$model_string"
               --checkpoint_folder "$outdir"
          )

          echo "      Running: ${cmd[*]}"
          # Capture stdout to file; keep stderr on console. Use 2>&1 to capture both if preferred.
          "${cmd[@]}" >"$outfile"
          echo "      -> Saved to: $outfile"
        done
      done
    done
    echo
  done

  echo ">> Completed dataset: $BASENAME"
  echo
done
shopt -u nullglob

echo "All model evaluations completed."
