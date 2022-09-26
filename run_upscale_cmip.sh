#!/bin/bash
modelnames='ACCESS-CM2 ACCESS-ESM1-5 BCC-CSM2-MR CESM2-WACCM CESM2 CMCC-CM2-SR5 CMCC-ESM2 CNRM-CM6-1-HR CNRM-CM6-1 CNRM-ESM2-1 CanESM5-CanOE CanESM5 E3SM-1-1 EC-Earth3-AerChem EC-Earth3-Veg-LR EC-Earth3-Veg FGOALS-f3-L FGOALS-g3 GFDL-ESM4 GISS-E2-1-G GISS-E2-1-H GISS-E2-2-G HadGEM3-GC31-MM INM-CM4-8 INM-CM5-0 IPSL-CM6A-LR MIROC-ES2L MIROC6 MPI-ESM1-2-HR MPI-ESM1-2-LR MRI-ESM2-0 UKESM1-0-LL'
#modelnames='HadGEM3-GC31-MM MIROC6 MPI-ESM1-2-HR IPSL-CM6A-LR ACCESS-ESM1-5 BCC-CSM2-MR CESM2 CMCC-ESM2 CNRM-ESM2-1 CanESM5 E3SM-1-1 FGOALS-g3 GFDL-ESM4 GISS-E2-1-H INM-CM4-8 UKESM1-0-LL'
metrics='corr seasonality trend r2'
metrics='trend corr'
strategies='systematic random interp'
#strategies='systematic'
#strategies='random'
testcase='smmask2'
for strategy in $strategies; do
    for metric in $metrics; do
        for modelname in $modelnames; do
            if [[ ! -f /home/bverena/optimal_station_network/corrmap_${strategy}_${modelname}_${metric}_${testcase}.nc ]]; then
                echo "$modelname $strategy $metric running ..."
                python upscale_cmip.py --model $modelname --metric $metric --method $strategy
            else
                echo "$modelname $strategy $metric already exists. skipping ..."
            fi
        done
    done
done

#modelnames = ['ACCESS-CM2 ','ACCESS-ESM1-5 ','BCC-CSM2-MR ','CESM2-WACCM ',
#              'CESM2 ','CMCC-CM2-SR5 ','CMCC-ESM2 ','CNRM-CM6-1-HR ',
#              'CNRM-CM6-1 ','CNRM-ESM2-1 ','CanESM5-CanOE ','CanESM5 E3SM-1-1 ',
#              'EC-Earth3-AerChem ','EC-Earth3-Veg-LR ','EC-EARTH3-Veg ',
#              'FGOALS-f3-L ','FGOALS-g3 ','GFDL-ESM4 ','GISS-E2-1-G ',
#              'GISS-E2-1-H ','GISS-E2-2-G ','HadGEM3-GC31-MM ','INM-CM4-8 ',
#              'INM-CM5-0 ','IPSL-CM6A-LR ','MIROC-ES2L ','MIROC6 ',
#              'MPI-ESM1-2-HR ','MPI-ESM1-2-LR ','MRI-ESM2-0 ','UKESM1-0-LL']
