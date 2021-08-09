#!/bin/bash
#for fold in {0..124}; do
#set -x
# mem in MB per core
#epoch=36
for g in $(seq 0 10); do
    bsub -R "rusage[mem=300]" -eo log_g${g}.out -oo log_g${g}.out -n 100 -W 00:10 python benchmark_rf_cmip_euler.py -g ${g}
    echo "${g} subitted to queue ..."
done
