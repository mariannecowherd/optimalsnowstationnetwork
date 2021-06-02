import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.ensemble import RandomForestRegressor

# lat lon grid
lsmfile = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/era5_deterministic_recent.lsm.025deg.time-invariant.nc'
lsm = xr.open_mfdataset(lsmfile, combine='by_coords')
shape = lsm['lsm'].squeeze().shape
landlat, landlon = np.where((lsm['lsm'].squeeze() > 0.8).load()) # land is 1, ocean is 0
datalat, datalon = lsm.lat.values, lsm.lon.values
X, Y = np.meshgrid(lsm.lon, lsm.lat)
X = X[landlat, landlon]
Y = Y[landlat, landlon]

def to_latlon(data):
    tmp = xr.DataArray(np.full((shape[0],shape[1]),np.nan), coords=[lsm.coords['lat'], lsm.coords['lon']], dims=['lat','lon'])
    tmp.values[landlat,landlon] = data
    return tmp

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

# select pseudo observations at station locations from era5
sktfile = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1m/processed/regrid/era5_deterministic_recent.skt.025deg.1m.2020.nc'
data = xr.open_dataset(sktfile)
data = data.mean(dim='time')['skt']
data = data.isel(lon=xr.DataArray(landlon, dims='landpoints'), 
                 lat=xr.DataArray(landlat, dims='landpoints')).squeeze()
lat_landpoints = data.lat.values
lon_landpoints = data.lon.values
selected_landpoints = []
for lat, lon in zip(station_grid_lat,station_grid_lon):
    try:
        selected_landpoints.append(np.intersect1d(np.where(lat_landpoints == lat), np.where(lon_landpoints == lon))[0])
    except IndexError as e: # station in the ocean acc to era5 landmask
        pass
selected_landpoints = np.unique(selected_landpoints) # some era5 gridpoints contain more than two stations
values = data.sel(landpoints=selected_landpoints)
datamean = values.mean().values.copy()
datastd = values.std().values.copy()
values = (values - datamean) / datastd

# select unique interpolated station locations
points = np.array([data.sel(landpoints=selected_landpoints).lon.values, data.sel(landpoints=selected_landpoints).lat.values]).T
#points = (points - points.mean(axis=0)) / points.std(axis=0)

# train GP on observed points 
#std_prior = 1**2 # std deviation of values
##length_scale = # std deviation in space
#kernel = std_prior * RBF(1.0, length_scale_bounds='fixed')
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
rf.fit(points, values)

XY = np.column_stack([X.flatten(), Y.flatten()])
res = np.zeros((n_trees, XY.shape[0]))
for t, tree in enumerate(rf.estimators_):
    print(t)
    res[t,:] = tree.predict(XY)
mean = np.mean(res, axis=0)
upper, lower = np.percentile(res, [95 ,5], axis=0)

# predict GP on all other points of grid
#XY = (XY - XY.mean(axis=0)) / XY.std(axis=0)
datamap = to_latlon(data)
predmap = to_latlon(mean)
uncmap = to_latlon((upper - lower))
predmap = predmap * datastd + datamean
uncmap = uncmap * datastd 


# plot prediction and uncertainty
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(131, projection=proj)
ax2 = fig.add_subplot(132, projection=proj)
ax3 = fig.add_subplot(133, projection=proj)
datamap.plot(ax=ax1, transform=proj, cmap='coolwarm', vmin=230, vmax=310)
predmap.plot(ax=ax2, transform=proj, cmap='coolwarm', vmin=230, vmax=310)
uncmap.plot(ax=ax3, transform=proj, cmap='pink_r')
ax1.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
ax2.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
ax3.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
plt.show()
