import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# no overall pattern visible in trends, no overall trend visible
# could be due to errors in preprocessing (values different units?)
# could be due to short timespan of data
# could be because no trend is emerging yet # check this option by comparing with era5


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

largefilepath = '/net/so4/landclim/bverena/large_files/'

# get era5 soil moisture data
years = list(np.arange(1979,2020))
varnames = ['swvl1','swvl2','swvl3','swvl4']
era5path_variant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1m/processed/regrid/'
filenames = [f'{era5path_variant}era5_deterministic_recent.{varname}.025deg.1m.{year}.nc' for year in years for varname in varnames]
era5 = xr.open_mfdataset(filenames)
era5 = era5.resample(time='1y').mean().to_array().mean(dim='variable')
era5 = era5.load()

# get station soil moisture data
data = xr.open_dataset(f'{largefilepath}df_gaps.nc')
data = data['__xarray_dataarray_variable__']
#data = (data - data.mean(dim='time')) / data.std(dim='time')
data = (data - data.mean()) / data.std()

#data.resample(time='1y').mean().mean(dim='stations').plot() # no global trend 

data = data.resample(time='1y').mean()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#trends = np.full(data.shape[1], np.nan)
#for s in range(data.stations.shape[0]):
#    #data.loc[:,station].plot(ax=ax, c='grey', alpha=0.1) # too busy
#    station = data.loc[:,s]
#    station = station.where(~np.isnan(station), drop=True)
#    time = np.arange(station.shape[0]) 
#    time = time[~np.isnan(station)]
#    try:
#        trends[s] = _calc_slope(time, station)
#    except ValueError:
#        continue
#ax.boxplot(trends[~np.isnan(trends)]) # no overall trend visible
#plt.show()

#trend = xr.full_like(era5[0,:,:].squeeze(), np.nan)
#trend = linear_trend(era5)
#proj = ccrs.PlateCarree()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection=proj)
#trend.plot(ax=ax, cmap='coolwarm_r', vmin=-1, vmax=1)
#plt.scatter(data.lon, data.lat, c=trends, cmap='coolwarm_r', vmin=-1, vmax=1, s=1)
#plt.show()

# select era5 pixel where stations
coords_unique = np.unique(np.array([data.lat_grid, data.lon_grid]), axis=1)

stationdata = xr.DataArray(np.full((era5.shape[0], coords_unique.shape[1]), np.nan),
                   coords=[era5.coords['time'], range(coords_unique.shape[1])], 
                   dims=['time','stations'])
for s in range(coords_unique.shape[1]):
    lat, lon = coords_unique[:,s]
    era5.sel(lat=lat, lon=lon)
    stationdata[:,s] = era5.sel(lat=lat, lon=lon)
era5 = era5[:,100:-100,:]
era5 = (era5 - era5.mean()) / era5.std()

# plot global trend
fig = plt.figure()
ax = fig.add_subplot(111)
era5.mean(dim=('lat','lon')).plot(ax=ax)
stationdata.mean(dim='stations').plot(ax=ax)
data.mean(dim='stations').plot(ax=ax)
plt.show()
