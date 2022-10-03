#!/bin/bash

# usage: ./kfold_run.sh run_name

# assumes config is properly configured
# i.e., TEST.FINAL_EVAL_METRICS is not empty
# and thresholds are not set for evaluation

NAME=$1;

echo "Running ${NAME} for fold 0..."
python -m seg_3d.train_loop -n "${NAME}" with "DATASET.FOLD=0"

echo "Running ${NAME} for fold 1..."
python -m seg_3d.train_loop -n "${NAME}" with "DATASET.FOLD=1"

echo "Running ${NAME} for fold 2..."
python -m seg_3d.train_loop -n "${NAME}" with "DATASET.FOLD=2"

echo "Final evaluation for ${NAME}..."
python -m seg_3d.evaluation.aggregate_preds -n ${NAME}