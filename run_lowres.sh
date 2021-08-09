#!/bin/bash
#for fold in {0..124}; do
#set -x
# mem in MB per core
#epoch=36
for g in $(seq 0 100); do
    bsub -R "rusage[mem=300]" -eo logfiles/log_g${g}.out -oo logfiles/log_g${g}.out -n 100 -W 00:10 python benchmark_rf_cmip_euler.py -g ${g}
    echo "${g} subitted to queue ..."
done
