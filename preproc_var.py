"""
TEST
"""

import numpy as np
import pandas as pd
import xarray as xr

# save means in file
#years = list(np.arange(1979,2020))
#largefilepath = '/net/so4/landclim/bverena/large_files/'
#era5path_variant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1m/processed/regrid/'
#varnames = ['t2m','skt','tp','swvl1','swvl2','swvl3','swvl4','e','ro','sshf','slhf','ssr','str']
#filenames_var = [f'{era5path_variant}era5_deterministic_recent.{varname}.025deg.1m.{year}.nc' for year in years for varname in varnames]
#data = xr.open_mfdataset(filenames_var, combine='by_coords')
#data.mean(dim='time').to_netcdf(largefilepath + 'era5_deterministic_recent.var.025deg.1m.mean.nc')

# load feature space X
largefilepath = '/net/so4/landclim/bverena/large_files/'
era5path_invariant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'
invarnames = ['lsm','z','slor','cvl','cvh', 'tvl', 'tvh']
filenames_constant = [f'{era5path_invariant}era5_deterministic_recent.{varname}.025deg.time-invariant.nc' for varname in invarnames]
filenames_variable = [f'{largefilepath}era5_deterministic_recent.temp.025deg.1y.max.nc', 
                     f'{largefilepath}era5_deterministic_recent.temp.025deg.1y.min.nc',
                     f'{largefilepath}era5_deterministic_recent.var.025deg.1y.mean.nc',
                     f'{largefilepath}era5_deterministic_recent.precip.025deg.1y.sum.nc']
constant = xr.open_mfdataset(filenames_constant, combine='by_coords').load()
variable = xr.open_mfdataset(filenames_variable, combine='by_coords').load()
landlat, landlon = np.where((constant['lsm'].squeeze() > 0.8).load()) # land is 1, ocean is 0
datalat, datalon = constant.lat.values, constant.lon.values
constant = constant.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                         lat=xr.DataArray(landlat, dims='landpoints')).squeeze()
variable = variable.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                         lat=xr.DataArray(landlat, dims='landpoints')).squeeze()
constant['latdat'] = ('landpoints', constant.lat.values)
constant['londat'] = ('landpoints', constant.lon.values)

# stack constant maps and merge with variables
ntimesteps = variable.coords['time'].size # TODO use climfill package
constant = constant.expand_dims({'time': ntimesteps}, axis=1)
constant['time'] = variable['time']
variable = variable.merge(constant)

# load station locations
largefilepath = '/net/so4/landclim/bverena/large_files/'
station_coords = pd.read_csv(largefilepath + 'fluxnet_station_coords.csv')
stations_lat = station_coords.LOCATION_LAT.values
stations_lon = station_coords.LOCATION_LONG.values

# interpolate station locations on era5 grid
def find_closest(list_of_values, number):
    return min(list_of_values, key=lambda x:abs(x-number))
station_grid_lat = []
station_grid_lon = []
for lat, lon in zip(stations_lat,stations_lon):
    station_grid_lat.append(find_closest(datalat, lat))
    station_grid_lon.append(find_closest(datalon, lon))

# load era5 data
years = list(np.arange(1979,2015))
varnames = ['swvl1','swvl2','swvl3','swvl4']
era5path_variant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1m/processed/regrid/'
filenames = [f'{era5path_variant}era5_deterministic_recent.{varname}.025deg.1m.{year}.nc' for year in years for varname in varnames]
data = xr.open_mfdataset(filenames)
data = data.resample(time='1y').mean().to_array().mean(dim='variable')
data = data.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                 lat=xr.DataArray(landlat, dims='landpoints')).squeeze()

# divide era5 gridpoints into those w station and those without
lat_landpoints = data.lat.values
lon_landpoints = data.lon.values
selected_landpoints = []
for lat, lon in zip(station_grid_lat,station_grid_lon):
    try:
        selected_landpoints.append(np.intersect1d(np.where(lat_landpoints == lat), np.where(lon_landpoints == lon))[0])
    except IndexError as e: # station in the ocean acc to era5 landmask
        pass
selected_landpoints = np.unique(selected_landpoints) # some era5 gridpoints contain more than two stations
other_landpoints = np.arange(data.landpoints.shape[0]).tolist()
for pt in selected_landpoints:
    other_landpoints.remove(pt)

# define data for learning and save
y_train = data.sel(landpoints=selected_landpoints)
X_train = variable.sel(landpoints=selected_landpoints).to_array()
y_test = data.sel(landpoints=other_landpoints)
X_test = variable.sel(landpoints=other_landpoints).to_array()

# stack along time axis
y_train = y_train.stack(datapoints=("time", "landpoints")).reset_index("datapoints").T
y_test = y_test.stack(datapoints=("time", "landpoints")).reset_index("datapoints").T
X_train = X_train.stack(datapoints=("time", "landpoints")).reset_index("datapoints").T
X_test = X_test.stack(datapoints=("time", "landpoints")).reset_index("datapoints").T

# save to file
case = 'yearly'
X_train.to_netcdf(f'{largefilepath}X_train_{case}.nc')
y_train.to_netcdf(f'{largefilepath}y_train_{case}.nc')
X_test.to_netcdf(f'{largefilepath}X_test_{case}.nc')
y_test.to_netcdf(f'{largefilepath}y_test_{case}.nc')
np.array(station_grid_lat).dump(f'{largefilepath}station_grid_lat.npy')
np.array(station_grid_lon).dump(f'{largefilepath}station_grid_lon.npy')
