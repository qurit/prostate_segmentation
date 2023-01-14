#!/bin/bash

# usage: ./run_kfold.sh config_path num_folds
#
# e.g. ./run_kfold.sh path/config.yaml 3

# assumes config is properly configured
# i.e., TEST.FINAL_EVAL_METRICS is not empty
# and INFERENCE_FILE_NAME is set to a filename

# append '-u' to train_loop to disable sacred logging

CFG=$1;
NUM_FOLDS=$2;
NAME="$(basename -- ${CFG%.yaml})"

# remove params from config file since they
# will be set by the train_loop command line
sed -i "/FOLD/d" $CFG  # very hacky...
sed -i "/EVAL_ONLY/d" $CFG

COUNT=0;
for i in $(seq 1 $NUM_FOLDS)
do 
  echo "Running $CFG for fold $COUNT...";
  # train
  python -m seg_3d.train_loop -n $NAME with "CONFIG_FILE=$CFG" "LOAD_ONLY_CFG_FILE=True" "DATASET.FOLD=$COUNT"  # -u
  # eval
  echo "Evaluating on test set for fold $COUNT...";
  python -m seg_3d.train_loop -n $NAME with "CONFIG_FILE=$CFG" "LOAD_ONLY_CFG_FILE=True" "DATASET.FOLD=$COUNT" "EVAL_ONLY=True"  # -u

  let COUNT++;
done

echo "Final evaluation for $CFG..."
python -m seg_3d.evaluation.aggregate_preds -n $NAME with "load_inference_fp='eval_0/inference.pk'"  # -u