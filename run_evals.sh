#!/usr/bin/env bash

DATASETS="data/converted_mlregtest/*/*LearningData.txt"

BATCH_SIZE=4
NUM_EPOCHS=10
LR=.01
MODEL_CLASSES=(
    "ptsl2"
    "ptsl2_times_pfsa"
    "diag_ssm"
    "pfsa"
    "sl2"
    "sp2"
    "sl2_times_sp2"
    "sl2_times_pfsa"
    "soft_tsl2"
)

mkdir -p "output/model_evaluations"


# python eval_model.py ptsl2 data/converted_mlregtest/SL.2.1.0/SL.2.1.0LearningData.txt data/converted_mlregtest/SL.2.1.0/SL.2.1.0TestingPairs.tsv --batch_size 512 --num_epochs 10 --lr 0.001 --test_data_paired --no_header --dev_split --save_checkpoints --checkpoint_folder ... --checkpoint_filename ... > ...


# ── Launch: one Terminal window per MODEL ──────────────────────────────────────
for model_type in $MODEL_CLASSES; do
    model_string="${model_type}_bs${BATCH_SIZE}_lr${LR}"
    for training_file in $DATASETS; do
	full_dir=$(dirname $training_file)
	the_basename=$(basename $full_dir)
	test_file="${full_dir}/${the_basename}TestingData.txt"
	checkpoint_dir="output/model_evaluations/${the_basename}/${model_type}/checkpoints"
	mkdir -p $checkpoint_dir
	outfile="output/model_evaluations/${the_basename}/${model_type}/${model_string}.csv"
	echo "Running ${model_type} on ${the_basename}"
	python eval_model.py $model_type $training_file $test_file --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --lr $LR --no_header --dev_split --save_checkpoints --checkpoint_folder $checkpoint_dir --checkpoint_filename $model_string > $outfile &
    done
done

