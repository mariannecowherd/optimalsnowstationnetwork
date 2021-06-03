"""
TEST
"""

import numpy as np
import pandas as pd
import xarray as xr

# load lat lon grid
lsmfile = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/era5_deterministic_recent.lsm.025deg.time-invariant.nc'
lsm = xr.open_mfdataset(lsmfile, combine='by_coords')
landlat, landlon = np.where((lsm['lsm'].squeeze() > 0.8).load()) # land is 1, ocean is 0
datalat, datalon = lsm.lat.values, lsm.lon.values

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
sktfile = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1m/processed/regrid/era5_deterministic_recent.skt.025deg.1m.2020.nc'
data = xr.open_dataset(sktfile)
data = data.mean(dim='time')['skt']
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
other_landpoints = np.arange(data.shape[0]).tolist()
for pt in selected_landpoints:
    other_landpoints.remove(pt)

# define data for learning and save
y_train = data.sel(landpoints=selected_landpoints)
X_train = xr.concat([data.sel(landpoints=selected_landpoints).lon, 
                     data.sel(landpoints=selected_landpoints).lat],
                     dim='variable')
y_test = data.sel(landpoints=other_landpoints)
X_test = xr.concat([data.sel(landpoints=other_landpoints).lon, 
                    data.sel(landpoints=other_landpoints).lat],
                    dim='variable')

# save to file
X_train.to_netcdf(f'{largefilepath}X_train.nc')
y_train.to_netcdf(f'{largefilepath}y_train.nc')
X_test.to_netcdf(f'{largefilepath}X_test.nc')
y_test.to_netcdf(f'{largefilepath}y_test.nc')
