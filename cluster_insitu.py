"""
cluster landpoints along environmental similiarity, then count how many in-situ observations are in each cluster

    @author: verena bessenbacher
    @date: 08 03 2021
"""

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING) # no matplotlib logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

# read feature table
logging.info(f'read feature table...')
largefilepath = '/net/so4/landclim/bverena/large_files/'
plotpath = "/home/bverena/python/plots/"
idebugspace = False
data = xr.open_dataset(largefilepath + f'features_init_None_None_idebug_{idebugspace}.nc').load() # DEBUG None None
data = data['data']

# read fluxnet station coordinates
logging.info(f'read fluxnet stations data...')
def find_closest(list_of_values, number):
    return min(list_of_values, key=lambda x:abs(x-number))

station_coords = pd.read_csv(largefilepath + 'fluxnet_station_coords.csv')
stations_lat = station_coords.LOCATION_LAT.values
stations_lon = station_coords.LOCATION_LONG.values
data_lat = np.unique(data['latitude'])
data_lon = np.unique(data['longitude'])

station_grid_lat = []
station_grid_lon = []
for lat, lon in zip(stations_lat,stations_lon):
    station_grid_lat.append(find_closest(data_lat, lat))
    station_grid_lon.append(find_closest(data_lon, lon))

# cluster
logging.info(f'cluster...')
n_clusters = 100
labels = MiniBatchKMeans(n_clusters=n_clusters, verbose=0, batch_size=1000, random_state=0).fit_predict(data)
#labels = xr.DataArray(labels, dims=['datapoints','variable'], coords=[data['datapoints'], 'labels'])

# reshape data back to latlon format
logging.info(f'transform to worldmap...')
data = data.to_dataset(dim='variable')
data['labels'] = (('datapoints'), labels)
data = data.to_array()
data = data.set_index(datapoints=('time','landpoints')).unstack('datapoints')

lsm = xr.open_dataset(largefilepath + f'landmask_idebug_{idebugspace}_None.nc')
lsm = lsm.to_array().squeeze()
shape = lsm.shape
landlat, landlon = np.where(lsm)
def to_latlon(data):
    tmp = xr.DataArray(np.full((data.coords['variable'].size,data.coords['time'].size,shape[0],shape[1]),np.nan), coords=[data.coords['variable'], data.coords['time'], lsm.coords['latitude'], lsm.coords['longitude']], dims=['variable','time','latitude','longitude'])
    tmp.values[:,:,landlat,landlon] = data
    return tmp
data = to_latlon(data)

# find cluster for each station gridpoint
logging.info(f'...')
labels_stations = np.zeros((len(station_grid_lat), *data.coords['time'].shape))
for l, (lat, lon) in enumerate(zip(station_grid_lat,station_grid_lon)):
    labels_stations[l,:] = data.sel(latitude=lat, longitude=lon, variable='labels').values

insitu_observations_per_cluster = np.zeros(n_clusters)
for c in range(n_clusters):
    insitu_observations_per_cluster[c] = len(np.where(labels_stations == c)[0])
    data.sel(variable='labels').where(data.sel(variable='labels') == c).sum(axis=0).plot(cmap='Blues')
    plt.savefig(plotpath + f'cluster_{c}.png')
    plt.close()
plt.plot(insitu_observations_per_cluster)
plt.savefig(plotpath + f'cluster_allpng')
