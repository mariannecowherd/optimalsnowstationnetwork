import glob
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def _calc_slope(x, y):
    '''wrapper that returns the slop from a linear regression fit of x and y'''
    slope = stats.linregress(x, y)[0]  # extract slope only
    return slope

def linear_trend(obj):
    time_nums = xr.DataArray(obj['time'].values.astype(float),
                             dims='time',
                             coords={'time': obj['time']},
                           name='time_nums')
    trend = xr.apply_ufunc(_calc_slope, time_nums, obj,
                           vectorize=True,
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[[]],
                           output_dtypes=[float],
                           dask='parallelized')
    return trend

def rename_vars(data):
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({'mrso': f'{modelname}_{ensemblename}'})
    data = data.drop_dims('bnds')
    return data

cmip6path = '/net/atmos/data/cmip6-ng/mrso/ann/g025/'
ssp = 5
rcp = '85'
cmip = xr.open_mfdataset(f'{cmip6path}mrso*_ssp{ssp}{rcp}_*.nc', # shoyer says better use instead of glob
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars,  # rename vars to model name
                         join='left') # timestamp problem
cmip = cmip.to_array()

# calculate MM trend
mmm = cmip.mean(dim='variable').load()
trend = linear_trend(mmm)
mmm.mean(dim=('lat','lon')).plot()
trend.plot(cmap='coolwarm_r')
