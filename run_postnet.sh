#!/bin/bash

RUNS_ROOT="./runs"
mkdir -p $RUNS_ROOT

dumps_dir=$RUNS_ROOT"/dumps_libritts_postnet_$(date +"%s")" 
plots_dir=$RUNS_ROOT"/plots_libritts_postnet_$(date +"%s")"
logs_dir=$RUNS_ROOT"/logs_libritts_postnet_$(date +"%s")"
mkdir $logs_dir

#cp -r dumps $dumps_dir
#cp -r plots $plots_dir
#cp train_loss.png $logs_dir
#cp test_loss.png $logs_dir
#cp libritts_postnet.log $logs_dir
#cp model.py $logs_dir
#cp train.py $logs_dir
#cp test.py $logs_dir
#cp libritts_postnet.py $logs_dir
#cp dataset.py $logs_dir

rm -rf plots
python libritts_postnet.py 2>&1|tee libritts_postnet.log
