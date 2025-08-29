#!/usr/bin/env bash
#set -euo pipefail

# --- config that mirrors your Python script ---
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

# Create output base directory
mkdir -p "$OUTPUT_DIR"

echo "Scanning datasets under: $DATA_DIRECTORY"
echo

# Iterate over immediate subdirectories of DATA_DIRECTORY
# Use -print0/-d '' to be robust to spaces/newlines in names
while IFS= read -r -d '' DIR; do
  # Skip if not a directory (belt & suspenders)
  [[ -d "$DIR" ]] || continue

  BASENAME="$(basename "$DIR")"
  echo ">> Dataset: $BASENAME"

  training=""
  testing_paired=""
  testing_unpaired=""

  # Find files by the same name patterns as in Python script
  # We prefer deterministic selection if multiple match: pick the first alphabetically
  mapfile -d '' matches < <(find "$DIR" -maxdepth 1 -type f -name '*LearningData*' -print0 | sort -z)
  if (( ${#matches[@]} > 0 )); then training="${matches[0]}"; fi

  mapfile -d '' matches < <(find "$DIR" -maxdepth 1 -type f -name '*TestingPairs*' -print0 | sort -z)
  if (( ${#matches[@]} > 0 )); then testing_paired="${matches[0]}"; fi

  mapfile -d '' matches < <(find "$DIR" -maxdepth 1 -type f -name '*TestingUnpaired*' -print0 | sort -z)
  if (( ${#matches[@]} > 0 )); then testing_unpaired="${matches[0]}"; fi

  # Basic checks (the Python version assumed training + testing_paired exist)
  if [[ -z "$training" || -z "$testing_paired" ]]; then
    echo "   Skipping: required files not found."
    echo "   training:        $training"
    echo "   testing_paired:  $testing_paired"
    echo
    continue
  fi

  # Loop over models and hyperparameters
  for model_type in "${MODEL_CLASSES[@]}"; do
    echo "   Model: $model_type"

    outdir="$OUTPUT_DIR/$BASENAME/$model_type"
    mkdir -p "$outdir"

    for bs in "${BATCH_SIZES[@]}"; do
      for ne in "${NUM_EPOCHS[@]}"; do
        for lr in "${LRS[@]}"; do
          model_string="${model_type}_bs${bs}_ep${ne}_lr${lr}"
          outfile="$outdir/${model_string}.txt"

          # Assemble the command (quote paths/args!)
          # Note: char separator is a single space argument by default
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
	       --report_every 10
          )

          echo "      Running: ${cmd[*]}"
          # Redirect stdout to the log file; keep stderr on console (or redirect with 2>&1 if desired)
          "${cmd[@]}" >"$outfile"
          echo "      -> Saved to: $outfile"
        done
      done
    done
    echo
  done

  echo ">> Completed dataset: $BASENAME"
  echo
done < <(find "$DATA_DIRECTORY" -mindepth 1 -maxdepth 1 -type d -print0)

echo "All model evaluations completed."
