"""
TEST
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# load data
largefilepath = '/net/so4/landclim/bverena/large_files/'
modelname = 'CanESM5'
experimentname = 'historical'
ensemblename = 'r1i1p1f1'
orig = xr.open_dataset(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc')['mrso']
pred = xr.open_dataset(f'{largefilepath}mrso_pred_{modelname}_{experimentname}_{ensemblename}.nc')['mrso']

# drought indices: duration, severity, frequency, extent 
perc90 = orig.quantile(0.1, dim='time')
orig_drought = (orig - perc90)
orig_drought = orig_drought.where(orig_drought > 0)
pred_drought = (pred - perc90)
pred_drought = pred_drought.where(pred_drought > 0)
