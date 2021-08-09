#!/bin/bash
#for fold in {0..124}; do
#set -x
# mem in MB per core
#epoch=36
missingness='real'
frac_missing='None'
for epoch in 36 50 53; do
    COUNT=$(expr $epoch - 1)
    for fold in $(seq 0 ${COUNT}); do
        if [[ ! -f /cluster/work/climate/bverena/features_iter_label_e${epoch}f${fold}_${missingness}_${frac_missing}_smart_rfkmeans_idebug_True.nc ]]; then
            bsub -R "rusage[mem=40]" -eo log_e${epoch}f${fold}_lowres.out -oo log_e${epoch}f${fold}_lowres.out -n 100 -W 00:15 python rfkmeans_cluster.py -m ${missingness} -p ${frac_missing} -e ${epoch} -f ${fold} 
            echo "file ${missingness} ${frac_missing} e${epoch}f${fold} subitted to queue ..."
        else
            echo "file ${missingness} ${frac_missing} e${epoch}f${fold} already exists. skipping ..."
        fi
    done
done
