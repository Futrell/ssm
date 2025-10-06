#!/usr/bin/env bash
# One-window-per-model launcher for eval_model.py  (fixed: include test_file)
set -euo pipefail
shopt -s nullglob

# ── Config (same knobs as your original) ────────────────────────────────────────
DATA_DIRECTORY="${DATA_DIRECTORY:-data/converted_mlregtest}"
OUTPUT_DIR="${OUTPUT_DIR:-output/model_evaluations}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_model.py}"
PYBIN="${PYBIN:-python3.13}"

MODEL_CLASSES=(
  "ptsl2"
  "diag_ssm"
  "pfsa"
  "sl2"
  "sp2"
  "soft_tsl2"
)

BATCH_SIZES=( ${BATCH_SIZES:-4} )
NUM_EPOCHS=( ${NUM_EPOCHS:-10} )
LRS=( ${LRS:-0.01} )

# ── Helpers ────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

open_terminal_with_script() {
  local title="$1" tmp
  tmp="$(mktemp -t "run_${title}_XXXX.command")"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'echo "### %s"\n' "$title"
    cat
    printf 'echo "[%s] ALL DONE ✅"\n' "$title"
    printf 'read -r -p "Press Enter to close... " _ || true\n'
  } >"$tmp"
  chmod +x "$tmp"
  /usr/bin/open -a Terminal "$tmp"
}

q() { printf %q "$1"; }  # shell-escape a single token

mkdir -p "$OUTPUT_DIR"

echo "Scanning datasets under: $DATA_DIRECTORY"
echo

# ── Discover datasets and pick files (first alphabetically, as before) ─────────
# Store records as: BASENAME|TRAIN|TEST_PAIRED|TEST_UNPAIRED
DATASETS=()
for DIR in "$DATA_DIRECTORY"/*/; do
  DIR="${DIR%/}"
  BASENAME="$(basename "$DIR")"
  training="$(find "$DIR" -maxdepth 1 -type f -name '*LearningData*'     | LC_ALL=C sort | sed -n '1p')"
  testing_paired="$(find "$DIR" -maxdepth 1 -type f -name '*TestingPairs*'   | LC_ALL=C sort | sed -n '1p')"
  testing_unpaired="$(find "$DIR" -maxdepth 1 -type f -name '*TestingUnpaired*' | LC_ALL=C sort | sed -n '1p')"

  if [[ -z "${training:-}" || -z "${testing_paired:-}" ]]; then
    echo "   Skipping $BASENAME (missing required files)"
    echo "     training:        ${training:-<none>}"
    echo "     testing_paired:  ${testing_paired:-<none>}"
    echo
    continue
  fi
  DATASETS+=("${BASENAME}|${training}|${testing_paired}|${testing_unpaired:-}")
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No usable datasets found under: $DATA_DIRECTORY" >&2
  exit 1
fi

# ── Launch: one Terminal window per MODEL ──────────────────────────────────────
for model_type in "${MODEL_CLASSES[@]}"; do
  {
    printf 'echo "Model: %s"\n' "$model_type"
    for rec in "${DATASETS[@]}"; do
      IFS='|' read -r BASENAME training testing_paired testing_unpaired <<<"$rec"

      printf 'echo "  >> Dataset: %s"\n' "$BASENAME"
      printf 'outdir=%q\n' "$OUTPUT_DIR/$BASENAME/$model_type"
      printf 'mkdir -p "$outdir"\n'

      for bs in "${BATCH_SIZES[@]}"; do
        for ne in "${NUM_EPOCHS[@]}"; do
          for lr in "${LRS[@]}"; do
            model_string="${model_type}_bs${bs}_ep${ne}_lr${lr}"
            outfile="$OUTPUT_DIR/$BASENAME/$model_type/${model_string}.txt"

            # ----- PRINT command (for visibility) -----
            printf 'echo "     Running: %s %s %s %s %s --batch_size %s --num_epochs %s --lr %s --report_every 10 --save_checkpoints --checkpoint_filename %s --checkpoint_folder %s --test_data_paired"\n' \
              "$(q "$PYBIN")" "$(q "$EVAL_SCRIPT")" "$(q "$model_type")" \
              "$(q "$training")" "$(q "$testing_paired")" \
              "$bs" "$ne" "$lr" "$(q "$model_string")" "$(q "$OUTPUT_DIR/$BASENAME/$model_type")"

            # ----- ACTUAL command (stdout -> file, stderr -> console) -----
            printf '%s %s %s %s %s --batch_size %s --num_epochs %s --lr %s --report_every 10 --save_checkpoints --checkpoint_filename %s --checkpoint_folder %s --test_data_paired >%s &\n' \
              "$(q "$PYBIN")" "$(q "$EVAL_SCRIPT")" "$(q "$model_type")" \
              "$(q "$training")" "$(q "$testing_paired")" \
              "$bs" "$ne" "$lr" \
              "$(q "$model_string")" "$(q "$OUTPUT_DIR/$BASENAME/$model_type")" \
              "$(q "$outfile")"

            printf 'echo "     -> Saved to: %s"\n' "$(q "$outfile")"
          done
        done
      done
      printf 'echo\n'
    done
    printf 'wait\n'
  } | open_terminal_with_script "MODEL_${model_type}"
  sleep 0.1
done

echo "All model windows launched."
