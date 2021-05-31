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
from sklearn.ensemble import RandomForestRegressor

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

# TODO clean up below
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

# TODO clean up below
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

n_trees = 100
kwargs = {'n_estimators': n_trees,
          'min_samples_leaf': 2,
          'max_features': 0.5, 
          'max_samples': 0.5, 
          'bootstrap': True,
          'warm_start': True,
          'n_jobs': 100, # set to number of trees
          'verbose': 0}
rf = RandomForestRegressor(**kwargs)
rf.fit(X=data.sel(landpoints=selected_landpoints).to_dataset('variable').drop('skt').to_array().T,
       y=data.sel(landpoints=selected_landpoints, variable='skt'))

res = np.zeros((n_trees, data.sel(landpoints=other_landpoints).shape[1]))
for t, tree in enumerate(rf.estimators_):
    print(t)
    res[t,:] = tree.predict(X=data.sel(landpoints=other_landpoints).to_dataset('variable').drop('skt').to_array().T)
import IPython; IPython.embed()
upper, lower = np.percentile(res, [95 ,5], axis=0)
