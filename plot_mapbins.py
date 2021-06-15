"""
plot data climate classification and classification per station
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

# open data map
case = 'latlontime'
largefilepath = '/net/so4/landclim/bverena/large_files/'
pred = f'{largefilepath}RFpred_{case}.nc'
orig = f'{largefilepath}ERA5_{case}.nc'
pred = xr.open_dataarray(pred)
orig = xr.open_dataarray(orig)
pred = (pred - orig).mean(dim='time')
unc = f'{largefilepath}UncPred_{case}.nc'
unc = xr.open_dataarray(unc)
filename = f'{largefilepath}Beck_KG_V1_present_0p5.tif'
koeppen = xr.open_rasterio(filename)
koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()

regridder = xe.Regridder(unc, koeppen, 'bilinear', reuse_weights=False)
unc = regridder(unc)
pred = regridder(pred)

# open station locations
station_coords = np.load(largefilepath + 'ISMN_station_locations.npy', allow_pickle=True)
stations_lat = station_coords[:,0]
stations_lon = station_coords[:,1]

# interpolate station locations on koeppen grid
datalat, datalon = koeppen.lat.values, koeppen.lon.values
def find_closest(list_of_values, number):
    return min(list_of_values, key=lambda x:abs(x-number))
station_grid_lat = []
station_grid_lon = []
for lat, lon in zip(stations_lat,stations_lon):
    station_grid_lat.append(find_closest(datalat, lat))
    station_grid_lon.append(find_closest(datalon, lon))

# extract koeppen class of station location
koeppen_class = []
for lat, lon in zip(station_grid_lat,station_grid_lon):
    koeppen_class.append(koeppen.sel(lon=lon, lat=lat).values.item())
koeppen_data = np.unique(koeppen_class, return_counts=True)[1]

# data for boxplot per koeppen
mean_unc = []
mean_pred = []
no_stations = []
for i in range(30):

    tmp = unc.where(koeppen == i).values.flatten()
    tmp = tmp[~np.isnan(tmp)]
    #boxplot_unc.append(tmp)
    mean_unc.append(np.abs(tmp.mean()))

    tmp = pred.where(koeppen == i).values.flatten()
    tmp = tmp[~np.isnan(tmp)]
    #boxplot_unc.append(tmp)
    mean_pred.append(np.abs(tmp.mean()))

    no_stations.append((np.array(koeppen_class) == i).sum())

legend = pd.read_csv('koeppen_legend.txt', delimiter=';')
labels = legend.Short.values
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
fig.suptitle('koeppen climate classes')
ax[0].scatter(mean_pred, mean_unc, c='blue')
#plt.scatter(no_stations, mean_pred, c='red')
for i,(x,y) in enumerate(zip(mean_pred,mean_unc)):
    ax[0].annotate(labels[i], xy=(x,y))
ax[1].scatter(no_stations, mean_unc, c='blue')
for i,(x,y) in enumerate(zip(no_stations,mean_unc)):
    ax[1].annotate(labels[i], xy=(x,y))
ax[2].scatter(no_stations, mean_pred, c='blue')
for i,(x,y) in enumerate(zip(no_stations,mean_pred)):
    ax[2].annotate(labels[i], xy=(x,y))
ax[0].set_ylabel('mean prediction uncertainty')
ax[0].set_xlabel('mean prediction error')
ax[1].set_ylabel('mean prediction uncertainty')
ax[1].set_xlabel('number of stations')
ax[2].set_ylabel('mean prediction error')
ax[2].set_xlabel('number of stations')
plt.savefig(f'scatter_{case}.png')

quit()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
ax.boxplot(boxplot_data)
ax.set_xticks(np.arange(30))
ax.set_xticklabels(legend.Short.values, rotation=90)
plt.show()

# plot
import IPython; IPython.embed()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
data.plot(ax=ax[0], add_colorbar=False, cmap='terrain')
ax[0].set_title('prediction error')
ax[0].set_xticklabels('')
ax[0].set_yticklabels('')
ax[0].scatter(stations_lon, stations_lat, marker='x', s=5, c='indianred')

ax[1].hist(data_class, bins=np.arange(30), align='left')
ax[1].set_xticks(np.arange(30))
ax[1].set_xticklabels(legend.Short.values, rotation=90)
ax[1].set_xlabel('koeppen climate classes')
ax[1].set_ylabel('number of stations')
plt.savefig('data_ismn.png')
