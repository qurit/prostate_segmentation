#!/bin/bash

# usage: ./run_configs.sh dir_path

# where dir_path is a directory storing
# all the config yamls for the runs

# assumes the configs are properly configured
# along with train_loop.py

DIR=$1;
COUNT=0;
for CFG in `ls $DIR/*.yaml`
do 
  echo "$COUNT: Running config $CFG...";
  python -m seg_3d.train_loop -n "$(basename -- $CFG)" with "CONFIG_FILE=$CFG"
  let COUNT++;
done