"""
mask of land-atmosphere coupling (as defined p(P,E) positive) and agpop region

"""

import cftime
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def rename_vars(data):
    varname = list(data.keys())[0]
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({varname: f'{modelname}'})
    try:
        data = data.drop_vars('file_qf')
    except ValueError:
        pass
    try:
        data = data.drop_dims('height')
    except ValueError:
        pass
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap):
        data['time'] = data.indexes['time'].to_datetimeindex()
    if isinstance(data.time[0].item(), cftime._cftime.Datetime360Day):
        data['time'] = data.indexes['time'].to_datetimeindex()
    return data

def open_from_list(filelist):
    data = xr.open_mfdataset(filenames, 
                             combine='nested', # timestamp problem
                             concat_dim=None,  # timestamp problem
                             compat='override', # DANGER
                             preprocess=rename_vars)  # rename vars to model name
    data = data.to_array(dim='model').rename(varname)
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')
    return data

cmip6_path = f'/net/atmos/data/cmip6-ng/mrso/mon/g025/'
filepaths = glob.glob(f'{cmip6_path}mrso_mon_*_ssp370_*r1i1*_g025.nc')
modelnames = [filepath.split('_')[2] for filepath in filepaths]
modelnames.remove('MCM-UA-1-0') # does not have pr
modelnames.remove('GISS-E2-2-G') # does not have et
modelnames = np.unique(modelnames) # double entries

varname = 'tas'
filenames = []
for modelname in modelnames:
    cmip6_path = f'/net/atmos/data/cmip6-ng/{varname}/mon/g025/'
    filenames.append(glob.glob(cmip6_path + f'{varname}_mon_{modelname}_ssp370_r1i1*_g025.nc')[0])
tas = open_from_list(filenames).load()

varname = 'evspsbl'
filenames = []
for modelname in modelnames:
    cmip6_path = f'/net/atmos/data/cmip6-ng/{varname}/mon/g025/'
    filenames.append(glob.glob(cmip6_path + f'{varname}_mon_{modelname}_ssp370_r1i1*_g025.nc')[0])
et = open_from_list(filenames).load()

varname = 'pr'
filenames = []
for modelname in modelnames:
    cmip6_path = f'/net/atmos/data/cmip6-ng/{varname}/mon/g025/'
    filenames.append(glob.glob(cmip6_path + f'{varname}_mon_{modelname}_ssp370_r1i1*_g025.nc')[0])
pr = open_from_list(filenames).load()

# resample to common month def
# models have different days for monthly values (15th, 16th, diff hours)
tas = tas.resample(time='M').mean()
et = et.resample(time='M').mean()
pr = pr.resample(time='M').mean()

# get landmask
upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'
landmask = xr.open_dataarray(f'{upscalepath}landmask.nc')
import regionmask
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(tas.lon, tas.lat) # DEBUG

# cut out 2014 to 2050
et = et.sel(time=slice('2015','2050'))
tas = tas.sel(time=slice('2015','2050'))
pr = pr.sel(time=slice('2015','2050'))

# take driest 3 months and mean yearly # consecutive, solution from MH
n_months = 3
monthly = pr.groupby('time.month').mean()
padded = monthly.pad(month=n_months, mode="wrap")
rolled = padded.rolling(center=True, month=n_months).mean()
sliced = rolled.isel(month=slice(n_months, -n_months))
model_mean = sliced.mean(dim='model')
#central_month = model_mean.argmax(dim='month')
month_mask = xr.zeros_like(model_mean, bool)
central_month = getattr(model_mean, "idxmin")("month")
all_nan = central_month.isnull()
central_month_arg = (central_month.fillna(0) - 1).astype(int)
for i in range(-1, 2):
    # the "% 12" normalizes the index 12 -> 0; 13 -> 1; -1 -> 11
    idx = (central_month_arg + i) % 12
    month_mask[{"month": idx}] = True
#.rolling(month=3).sum().pad(month=1, mode='wrap').mean(dim='model').argmax(dim='month')
#middle_month.where(landmask).plot()
#month_mask = month_mask.rename({'month':'time'})

for year in np.unique(pr.coords['time.year']):
    tas.loc[dict(time=slice(f'{year}-01-01', f'{year+1}-01-01'))] = tas.loc[dict(time=slice(f'{year}-01-01', f'{year+1}-01-01'))].where(month_mask.values)
    et.loc[dict(time=slice(f'{year}-01-01', f'{year+1}-01-01'))] = et.loc[dict(time=slice(f'{year}-01-01', f'{year+1}-01-01'))].where(month_mask.values)

# interannual variability
et = et.resample(time='Y').mean()
tas = tas.resample(time='Y').mean()

# calc corr and plot
corr = xr.corr(et, tas, dim='time')
corr = corr.mean(dim='model')
corr = corr.where(~np.isnan(landmask)) # DEBUG
#for modelname in modelnames:
#    corr.sel(model=modelname).plot()
#    plt.show()
corr.plot()
plt.show()

# popcrop
pop = xr.open_dataarray(f'{upscalepath}population_density_regridded.nc')
crop = xr.open_dataarray(f'{upscalepath}cropland_regridded.nc')

# combine args
mask = ((corr < 0.2) | (crop > 10) | (pop > 100)).drop(['variable','height','band','spatial_ref','raster']).squeeze()
mask.where(~np.isnan(landmask)).plot()
plt.show()

# smooth filter
from scipy.ndimage import median_filter
tmp = median_filter(mask, size=(5,5))
mask = mask | tmp
mask.where(~np.isnan(landmask)).plot()
plt.show()

# indiv points
mask = mask.where((mask.lat >= 40) | (mask.lat <=-40), True)
mask.where(~np.isnan(landmask)).plot()
plt.show()

# save
mask = mask.where(~np.isnan(landmask), False)
mask.to_netcdf(f'{upscalepath}smcoup_agpop_mask.nc')
