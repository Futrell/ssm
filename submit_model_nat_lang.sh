#!/usr/bin/env bash
set -euo pipefail

# Check if the correct number of arguments is passed
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <model_type> <dataset>"
  exit 1
fi

# Assign arguments to variables
MODEL_TYPE=$1
DATASET=$2

# Build the JOB_NAME from the arguments
JOB_NAME="run_${MODEL_TYPE}_${DATASET}"

# Build the ENV_CMD based on the arguments
ENV_CMD="
module load python/3.10
source .venvs/ssm-akondur/bin/activate
python run_model_nat_lang.py $MODEL_TYPE $DATASET
"

PARTITION=standard
TIME=7-0:00:00
MEM=4G

sbatch \
  -p "$PARTITION" \
  -J "$JOB_NAME" \
  -t "$TIME" \
  --mem="$MEM" \
  --cpus-per-task=1 \
  --ntasks=1 \
  -o logs/$JOB_NAME.%j.out \
  --account="CJMAYER_LAB" \
  --wrap "bash -lc '
    set -euo pipefail
    $ENV_CMD
  '"
