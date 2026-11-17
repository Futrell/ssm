#!/usr/bin/env bash
set -euo pipefail

# --- EDIT THESE IF YOUR CLUSTER REQUIRES THEM ---
PARTITION=""         # e.g., "short"
ACCOUNT="rfutrell"           # e.g., "myacct"
TIME="02:00:00"      # Quechua is small: a few minutes per run on a laptop CPU
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

# --- QUECHUA DATA ---
# Training/dev files: one space-separated form per line, no header, no labels
# (train + dev = quechua_training_shuffled.txt).
# Test file: tab-separated "form<TAB>TRUE|FALSE", no header. Reported as
# TRUE_scores, FALSE_scores, FALSE_TRUE_diff (ungrammatical minus grammatical,
# so negative when the model has learned the pattern) and its Welch t.
LANG_NAME="quechua"
TRAIN_FILE="data/quechua/quechua_train.txt"
DEV_FILE="data/quechua/quechua_dev.txt"
TEST_FILE="data/quechua/quechua_test.txt"

# One job per (model, batch size, learning rate) combination
BATCH_SIZES=(1 2 4 8)
LRS=(.001 .01)
NUM_EPOCHS=10       # 1,975 training forms: ~4,940 steps at bs4, ~620 at bs32
REPORT_EVERY=50     # eval + checkpoint every N steps: ~99 per run at bs4, ~12 at bs32
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
            --no_header --test_no_header --no_filter_training_grammatical \
            --dev_file \"$DEV_FILE\" --save_checkpoints \
            --checkpoint_folder \"$checkpoint_dir\" \
            --checkpoint_filename \"$model_string\"
        '"
    done
  done
done
