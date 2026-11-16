#!/usr/bin/env bash
set -euo pipefail

# --- EDIT THESE IF YOUR CLUSTER REQUIRES THEM ---
PARTITION=""         # e.g., "short"
ACCOUNT="rfutrell"           # e.g., "myacct"
TIME="24:00:00"     # ~5-30 min of training per run at bs16 on a laptop CPU; generous margin
MEM="8G"
CPUS="2"
GPUS=0               # set to 1 if you want a GPU
# -------------------------------------------------

# --- HOW TO GET PYTHON + DEPENDENCIES IN THE JOB ---
# Option A: conda env (recommended) — name it to your env that has:
#   torch numpy scipy tqdm torch_semiring_einsum
ENV_CMD='source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch2'

# Option B: module load (uncomment & adjust, then comment the conda line)
# ENV_CMD='module load python/3.10 pytorch/2.3 && python -c "import torch"'

# (Avoid pip-installing in jobs unless your cluster allows it.)
# -------------------------------------------------

# --- TURKISH DATA ---
# Training/dev files: one space-separated form per line, no header, no labels.
# Test file: comma-separated, header "word,z_mean_response,mean_response,violates_harmony";
# every label column is evaluated (correlations with z_mean_response, under the
# plain names spearman/pearson, and with mean_response; harmonic vs. disharmonic
# score diff as violates_harmony_FALSE_TRUE_diff).
LANG_NAME="turkish"
TRAIN_FILE="data/turkish/turkish_train.txt"
DEV_FILE="data/turkish/turkish_dev.txt"
TEST_FILE="data/turkish/turkish_test_data.csv"

# One job per (model, batch size, learning rate) combination
BATCH_SIZES=(1 4 16)
LRS=(.001 .01)
NUM_EPOCHS=1
REPORT_EVERY=10   
MODEL_CLASSES=(
  "sl2"
  "sp2"
  "pfsa"
  "diag_ssm"  
  "sl2_times_sp2"
  "sl2_times_pfsa"
  "sl2_times_ptsl2"
  "sl2_times_sp2_times_ptsl2"
)

mkdir -p "output/model_evaluations/${LANG_NAME}" logs

for model_type in "${MODEL_CLASSES[@]}"; do
  # output/model_evaluations/<lang>/<model>/<model>_bs<N>_lr<lr>.csv is the
  # layout aggregate_output.py expects
  checkpoint_dir="output/model_evaluations/${LANG_NAME}/${model_type}"
  mkdir -p "$checkpoint_dir/checkpoints"

  for batch_size in "${BATCH_SIZES[@]}"; do
    for lr in "${LRS[@]}"; do
      model_string="${model_type}_bs${batch_size}_lr${lr}"
      outfile="${checkpoint_dir}/${model_string}.csv"
      jobname="${model_string}_${LANG_NAME}"

      echo "Submitting ${jobname}"

      sbatch \
        ${PARTITION:+-p "$PARTITION"} \
        ${ACCOUNT:+--account="$ACCOUNT"} \
        -J "$jobname" \
        -t "$TIME" \
        --mem="$MEM" \
        --cpus-per-task="$CPUS" \
        ${GPUS:+$( ((GPUS>0)) && echo --gpus="$GPUS" )} \
        -o "$outfile" \
        -e "logs/${jobname}.err" \
        --wrap "bash -lc '
          set -euo pipefail
          $ENV_CMD
          python eval_model.py \"$model_type\" \"$TRAIN_FILE\" \"$TEST_FILE\" \
            --batch_size $batch_size --num_epochs $NUM_EPOCHS --lr $lr \
            --report_every $REPORT_EVERY \
            --no_header --no_filter_training_grammatical --col_separator , \
            --dev_file \"$DEV_FILE\" --save_checkpoints \
            --checkpoint_folder \"$checkpoint_dir\" \
            --checkpoint_filename \"$model_string\"
        '"
    done
  done
done
