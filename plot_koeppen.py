"""
plot koeppen climate classification and classification per station
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# open koeppen map
largefilepath = '/net/so4/landclim/bverena/large_files/'
filename = f'{largefilepath}Beck_KG_V1_present_0p0083.tif'
koeppen = xr.open_rasterio(filename)

# open station locations
station_coords = np.load(largefilepath + 'ISMN_station_locations.npy', allow_pickle=True)
stations_lat = station_coords[:,0]
stations_lon = station_coords[:,1]

# interpolate station locations on koeppen grid
datalat, datalon = koeppen.y.values, koeppen.x.values
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
    koeppen_class.append(koeppen.sel(x=lon, y=lat).values[0])


# plot
legend = pd.read_csv('koeppen_legend.txt', delimiter=';')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
koeppen.plot(ax=ax[0], add_colorbar=False, cmap='terrain')
ax[0].set_title('koeppen climate classes and ISMN stations')
ax[0].set_xticklabels('')
ax[0].set_yticklabels('')
ax[0].scatter(stations_lon, stations_lat, marker='x', s=5, c='indianred')
ax[1].hist(koeppen_class, bins=np.arange(30), align='left')
ax[1].set_xticks(np.arange(30))
ax[1].set_xticklabels(legend.Short.values, rotation=90)
ax[1].set_xlabel('koeppen climate classes')
ax[1].set_ylabel('number of stations')
plt.savefig('koeppen_ismn.png')
