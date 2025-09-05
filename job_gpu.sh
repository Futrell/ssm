#!/usr/bin/env bash
#
# =====================  SLURM HEADER  =====================
# Job name:
#SBATCH --job-name=ssm

# Email notifications:
#SBATCH --mail-user=huteng@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Resources:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=36G
#SBATCH --time=96:00:00
#SBATCH --account=huteng0

# GPU resources:
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:8

# Logs (make sure this directory exists before sbatch):
#SBATCH --output=/home/%u/ssm/output/slurm_logs/%x-%j.log
#SBATCH --error=/home/%u/ssm/output/slurm_logs/%x-%j.err
# ==========================================================

set -euo pipefail

echo "===== SLURM job context ====="
echo "Job: ${SLURM_JOB_NAME:-<local>}, ID: ${SLURM_JOB_ID:-N/A}"
echo "User: ${USER}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "================================"

# --------------------------
# Environment Setup
# --------------------------
module purge
module load python
module load cuda/11.2

# If your conda init is in .bashrc:
source ~/.bashrc || true
# Activate your env
conda activate pytorch_tutorial

# Optional: ensure deps are present (comment out if not needed)
if [[ -f requirements.txt ]]; then
  echo "Installing requirements from requirements.txt ..."
  pip3 install -r requirements.txt
fi

# --------------------------
# Config (from your first script)
# --------------------------
DATA_DIRECTORY="data/truncated_mlregtest"
OUTPUT_DIR="output/model_evaluations"

# Which script to run. Switch to "mlregtest.py" if you prefer.
EVAL_SCRIPT="eval_model.py"

# Models
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
BATCH_SIZES=(4 8 64)
NUM_EPOCHS=(10)
LRS=(0.1 0.01 0.001)

# Reporting/ckpt options
REPORT_EVERY=10
SAVE_CKPT=1

# --------------------------
# Parallelism (use all GPUs by default)
# --------------------------
# Maximum concurrent runs. Defaults to GPUs requested by SLURM; override if desired.
MAX_PARALLEL="${MAX_PARALLEL:-${SLURM_GPUS:-1}}"
if [[ -z "${SLURM_GPUS-}" ]]; then
  # Fallback if SLURM_GPUS isn't set; try to count visible GPUs.
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    MAX_PARALLEL="${MAX_PARALLEL:-${GPU_COUNT:-1}}"
  fi
fi
echo "Max parallel runs: ${MAX_PARALLEL}"

mkdir -p "$OUTPUT_DIR"

# Utility: launch a job, respecting MAX_PARALLEL
launch() {
  # Usage: launch <outfile> <cmd...>
  local outfile="$1"; shift
  # Keep a simple dispatch trace alongside model outputs
  echo "      Running: $*" | tee -a "${outfile%.txt}.dispatch.log"

  if (( MAX_PARALLEL > 1 )); then
    # Allocate 1 GPU per run; --exclusive ensures no sharing of the allocation
    srun --exclusive -N1 -n1 --gres=gpu:1 "$@" >"$outfile" 2>&1 &
    # Throttle to MAX_PARALLEL
    while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]]; do
      wait -n
    done
  else
    "$@" >"$outfile" 2>&1
  fi
}

echo
echo "Scanning datasets under: $DATA_DIRECTORY"
echo

# Only consider immediate subdirectories
shopt -s nullglob
for DIR in "$DATA_DIRECTORY"/*/; do
  DIR="${DIR%/}"
  BASENAME="$(basename "$DIR")"
  echo ">> Dataset: $BASENAME"

  # First file (alphabetically) matching each pattern
  training="$(find "$DIR" -maxdepth 1 -type f -name '*LearningData*'  | LC_ALL=C sort | sed -n '1p')"
  testing_paired="$(find "$DIR" -maxdepth 1 -type f -name '*TestingPairs*' | LC_ALL=C sort | sed -n '1p')"
  testing_unpaired="$(find "$DIR" -maxdepth 1 -type f -name '*TestingUnpaired*' | LC_ALL=C sort | sed -n '1p' || true)"

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

          # Build the command as an array to keep quoting robust
          cmd=(python "$EVAL_SCRIPT"
               "$model_type"
               "$training"
               "$testing_paired"
               --batch_size "$bs"
               --num_epochs "$ne"
               --lr "$lr"
               --report_every "$REPORT_EVERY"
          )

          if (( SAVE_CKPT )); then
            cmd+=( --save_checkpoints
                   --checkpoint_filename "$model_string"
                   --checkpoint_folder "$outdir" )
          fi

          # Launch (possibly in parallel, respecting MAX_PARALLEL)
          launch "$outfile" "${cmd[@]}"
          echo "      -> Output will be saved to: $outfile"
        done
      done
    done
    echo
  done

  echo ">> Completed dataset: $BASENAME"
  echo
done
shopt -u nullglob

# Wait for any remaining backgrounded srun jobs
wait || true

echo "All model evaluations completed."
echo "Finished at: $(date)"
