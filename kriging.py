"""
create dataset for investigating optimal station networks

    @author: verena bessenbacher
    @date: 26 05 2020
"""

# decide which variables to use
# temperature, precipitation, evapotranspiration, runoff, soil moisture, sensible heat, carbon fluxes?
# mean, extremes and trends
# constant maps: altitude, topographic complexity, vegetation cover

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor

# define time range
years = list(np.arange(1979,2020))
#years = [2000]

# define paths
era5path_variant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1m/processed/regrid/'
era5path_invariant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'
invarnames = ['lsm','z','slor','cvl','cvh', 'tvl', 'tvh']
varnames = ['skt','tp','swvl1','swvl2','swvl3','swvl4','e','ro','sshf','slhf','ssr','str']

# define files
filenames_var = [f'{era5path_variant}era5_deterministic_recent.{varname}.025deg.1m.{year}.nc' for year in years for varname in varnames]
filenames_invar = [f'{era5path_invariant}era5_deterministic_recent.{varname}.025deg.time-invariant.nc' for varname in invarnames]

# open files
data = xr.open_mfdataset(filenames_var, combine='by_coords')
constant_maps = xr.open_mfdataset(filenames_invar, combine='by_coords')

# create statistics: mean, extreme, trends
datamean = data.to_array().mean(dim='time').to_dataset(dim='variable')

# merge constant maps and variables
landmask = (constant_maps['lsm'].squeeze() > 0.8).load() # land is 1, ocean is 0
landlat, landlon = np.where(landmask)

data = datamean.merge(constant_maps).to_array()
data = data.isel(lon=xr.DataArray(landlon, dims='landpoints'), 
                 lat=xr.DataArray(landlat, dims='landpoints')).squeeze()

# normalise data
datamean = data.mean(dim=('landpoints'))
datastd = data.std(dim=('landpoints'))
data = (data - datamean) / datastd

# calculate grid point that contains station
largefilepath = '/net/so4/landclim/bverena/large_files/'
station_coords = pd.read_csv(largefilepath + 'fluxnet_station_coords.csv')
stations_lat = station_coords.LOCATION_LAT.values
stations_lon = station_coords.LOCATION_LONG.values

# find gridpoint where fluxnet station is
def find_closest(list_of_values, number):
    return min(list_of_values, key=lambda x:abs(x-number))
data_lat = np.unique(data['lat'])
data_lon = np.unique(data['lon'])
station_grid_lat = []
station_grid_lon = []
for lat, lon in zip(stations_lat,stations_lon):
    station_grid_lat.append(find_closest(data_lat, lat))
    station_grid_lon.append(find_closest(data_lon, lon))

lat_landpoints = data.lat.values
lon_landpoints = data.lon.values
selected_landpoints = []
for l, (lat, lon) in enumerate(zip(station_grid_lat,station_grid_lon)):
    # get index of landpoint with this lat and lon
    try:
        selected_landpoints.append(np.intersect1d(np.where(lat_landpoints == lat)[0], np.where(lon_landpoints == lon)[0])[0])
    except:
        pass
selected_landpoints = np.unique(selected_landpoints) # some grid points are chosen more than one because contain more than one FLUXNET station
other_landpoints = np.arange(data.shape[1]).tolist()
for pt in selected_landpoints:
    other_landpoints.remove(pt)

# apply gaussian processes
gp = GaussianProcessRegressor()
gp.fit(X=data.sel(landpoints=selected_landpoints).to_dataset('variable').drop('skt').to_array().T,
       y=data.sel(landpoints=selected_landpoints, variable='skt'))
y_mean, y_std = gp.predict(X=data.sel(landpoints=other_landpoints).to_dataset('variable').drop('skt').to_array().T, return_std=True)

# renormalise
data = data * datastd + datamean

# wrap back to worldmap
prediction = xr.full_like(data.sel(variable='skt'), np.nan)
uncertainty = xr.full_like(data.sel(variable='skt'), np.nan)

prediction[other_landpoints] = y_mean
uncertainty[other_landpoints] = y_std

pred_map = xr.full_like(landmask.astype(float), np.nan)
pred_map.values[landlat,landlon] = prediction

unc_map = xr.full_like(landmask.astype(float), np.nan)
unc_map.values[landlat,landlon] = uncertainty

varmap = xr.full_like(landmask.astype(float), np.nan)
varmap.values[landlat,landlon] = data.sel(variable='skt')

# renormalise
pred_map = pred_map * datastd.sel(variable='skt') + datamean.sel(variable='skt')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
(varmap - pred_map).plot(ax=axes[0])
unc_map.plot(ax=axes[1])
axes[0].scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
axes[1].scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
plt.show()
