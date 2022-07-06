#!/bin/bash
#modelnames='IPSL-CM6A-LR HadGEM3-GC31-MM MIROC6 MPI-ESM1-2-HR'
modelnames='HadGEM3-GC31-MM MIROC6 MPI-ESM1-2-HR IPSL-CM6A-LR ACCESS-ESM1-5 BCC-CSM2-MR CESM2 CMCC-ESM2 CNRM-ESM2-1 CanESM5 E3SM-1-1 FGOALS-g3 GFDL-ESM4 GISS-E2-1-H INM-CM4-8 UKESM1-0-LL'
#modelnames='HadGEM3-GC31-MM MPI-ESM1-2-HR IPSL-CM6A-LR ACCESS-ESM1-5 BCC-CSM2-MR CESM2 CMCC-ESM2 CNRM-ESM2-1 CanESM5 E3SM-1-1 FGOALS-g3 GFDL-ESM4 GISS-E2-1-H INM-CM4-8 UKESM1-0-LL'
#metrics='corr seasonality trend r2'
metrics='r2'
strategies='random interp systematic'
for modelname in $modelnames; do
    for strategy in $strategies; do
        for metric in $metrics; do
            echo "$modelname $strategy $metric"
            python upscale_cmip.py --model $modelname --metric $metric --method $strategy
        done
    done
done
