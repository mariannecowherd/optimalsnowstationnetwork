"""
TEST
"""

from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr

# TODO add extremes, trends

# load feature space X
print('load and stack data')
largefilepath = '/net/so4/landclim/bverena/large_files/'
era5path_invariant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'
invarnames = ['lsm','z','slor','cvl','cvh', 'tvl', 'tvh']
filenames_constant = [f'{era5path_invariant}era5_deterministic_recent.{varname}.025deg.time-invariant.nc' for varname in invarnames]
filenames_variable = [f'{largefilepath}era5_deterministic_recent.temp.025deg.1y.max.nc', 
                     f'{largefilepath}era5_deterministic_recent.temp.025deg.1y.min.nc',
                     f'{largefilepath}era5_deterministic_recent.var.025deg.1y.mean.nc',
                     f'{largefilepath}era5_deterministic_recent.var.025deg.1y.roll.nc',
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
print('load station data')
largefilepath = '/net/so4/landclim/bverena/large_files/'
station_coords = pd.read_csv(largefilepath + 'station_info_grid.csv')
stations_grid_lat = station_coords.lat_grid.values
stations_grid_lon= station_coords.lon_grid.values
stations_start = [datetime.strptime(date, '%Y-%m-%d %M:%S:%f') for date in station_coords.start.values]
stations_end = [datetime.strptime(date, '%Y-%m-%d %M:%S:%f') for date in station_coords.end.values]

# using sel would be easier for slicing, but we cannot split data into test and train before
# creating dimension datapoints because reindexing to worldmap would fail if test and train
# are not merged again after learning

# load era5 data
print('load y data')
years = list(np.arange(1979,2015))
varnames = ['swvl1','swvl2','swvl3','swvl4']
era5path_variant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1m/processed/regrid/'
filenames = [f'{era5path_variant}era5_deterministic_recent.{varname}.025deg.1m.{year}.nc' for year in years for varname in varnames]
data = xr.open_mfdataset(filenames)
data = data.resample(time='1y').mean().to_array().mean(dim='variable')
data = data.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                 lat=xr.DataArray(landlat, dims='landpoints')).squeeze()

# stack along time axis
print('stack and normalise')
data = data.stack(datapoints=("time", "landpoints")).reset_index("datapoints").T
variable = variable.stack(datapoints=("time", "landpoints")).reset_index("datapoints").to_array().T
data['datapoints'] = np.arange(data.shape[0]) # give some coords for datapoint dimension
variable['datapoints'] = np.arange(variable.shape[0]) # give some coords for datapoint dimension

# normalise values
case = 'latlontime'
datamean = data.mean()
datastd = data.std()
data = (data - datamean) / datastd
variablemean = variable.mean(dim='datapoints')
variablestd = variable.std(dim='datapoints')
variable = (variable - datamean) / datastd
datamean.to_netcdf(f'{largefilepath}datamean_{case}.nc')
variablemean.to_netcdf(f'{largefilepath}variablemean_{case}.nc')
datastd.to_netcdf(f'{largefilepath}datastd_{case}.nc')
variablestd.to_netcdf(f'{largefilepath}variablestd_{case}.nc')

# define data for testing and training
# caution: takes 45 minutes !!!
print('calculate landpoints of stations')
y_train_datacoords = []
for i, (lat, lon, start, end) in enumerate(zip(stations_grid_lat, stations_grid_lon, stations_start, stations_end)):
    one_station = data.where((data.lat == lat) & (data.lon == lon) & (data.time.isin(pd.date_range(start,end,freq='y'))), drop=True)
    datapoints_one_station = one_station.datapoints.values.tolist()
    y_train_datacoords.append(datapoints_one_station)
    print(len(stations_grid_lat),i,datapoints_one_station)
y_train_datacoords = [item for items in y_train_datacoords for item in items] # flatten list
import IPython; IPython.embed()

y_train = data.where(data.datapoints.isin(y_train_datacoords), drop=True)
y_test = data.where(~data.datapoints.isin(y_train_datacoords), drop=True)
X_train = variable.where(data.datapoints.isin(y_train_datacoords), drop=True)
X_test = variable.where(~data.datapoints.isin(y_train_datacoords), drop=True)

# save to file
X_train.to_netcdf(f'{largefilepath}X_train_{case}.nc')
y_train.to_netcdf(f'{largefilepath}y_train_{case}.nc')
X_test.to_netcdf(f'{largefilepath}X_test_{case}.nc')
y_test.to_netcdf(f'{largefilepath}y_test_{case}.nc')
