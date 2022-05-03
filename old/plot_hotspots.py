import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import xesmf as xe
import cftime

largefilepath = '/net/so4/landclim/bverena/large_files/'

def rename_vars(data):
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({varname: f'{modelname}_{ensemblename}'})
    try:
        data = data.drop_vars('file_qf')
    except ValueError:
        pass
    #data = data.drop_dims('height')
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap):
        data['time'] = data.indexes['time'].to_datetimeindex()
    return data

tasmax_path = '/net/atmos/data/cmip6-ng/tasmax/mon/g025/'
sm_path = '/net/atmos/data/cmip6-ng/mrso/mon/g025/'
ssp = 5
rcp = '85'
varname = 'tasmax'
tasmax = xr.open_mfdataset(f'{tasmax_path}{varname}*_ssp{ssp}{rcp}_r1i1p1f1_*.nc', # shoyer says better use instead of glob
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars)  # rename vars to model name
                         #join='left', # timestamp problem
                         #compat='override')
varname = 'mrso'
mrso = xr.open_mfdataset(f'{sm_path}{varname}*_ssp{ssp}{rcp}_r1i1p1f1_*.nc', # shoyer says better use instead of glob
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars,  # rename vars to model name
                         join='left', # timestamp problem
                         compat='override')

# calculate sm anomaly cumulated in last three months
seasonal_mean = mrso.groupby('time.month').mean()
seasonal_std = mrso.groupby('time.month').std()
mrso = (mrso.groupby('time.month') - seasonal_mean) 
mrso = mrso.groupby('time.month') / seasonal_std
mrso = mrso.rolling(time=3).mean()
mrso = mrso.to_array()

# get reference period 90th percentile from ERA5
icalc = False
if icalc:
    era5path = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid_tmax1d/'
    years = list(np.arange(1979,2010))
    filenames = [f'{era5path}era5_deterministic_recent.t2m.025deg.1h.{year}.tmax1d.nc' for year in years]
    tmax_hist = xr.open_mfdataset(filenames, combine='by_coords').drop('time_bnds')['t2m']
    np_q90 = np.percentile(tmax_hist, q=90, axis=0)
    q90 = xr.full_like(tmax_hist[0,:,:], np.nan)
    q90[:] = np_q90
    regridder = xe.Regridder(q90, mrso, 'bilinear', reuse_weights=False)
    q90_regrid = regridder(q90)
    q90_regrid.to_netcdf(f'{largefilepath}tmax_q90_era5_1979_2010.nc')
else:
    q90_regrid = xr.open_dataset(f'{largefilepath}tmax_q90_era5_1979_2010.nc')['t2m']

# calculate NHD (alternative: heatwave intensity in K over 90th percentile)
tasmax = tasmax.to_array().drop('height')
heatwave_intensity = tasmax - q90_regrid
heatwave_intensity = heatwave_intensity.where(heatwave_intensity > 0)
heatwave_intensity = heatwave_intensity.load()

# calculate correlation of the two in decade instalments
current = slice('2015','2025')
future = slice('2085','2095')

corr_current = xr.corr(heatwave_intensity.sel(time=current), mrso.sel(time=current), dim='time')
import IPython; IPython.embed()
corr_current.plot()

