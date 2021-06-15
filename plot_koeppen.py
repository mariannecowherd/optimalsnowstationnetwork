"""
plot koeppen climate classification and classification per station
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# open koeppen map
largefilepath = '/net/so4/landclim/bverena/large_files/'
filename = f'{largefilepath}Beck_KG_V1_present_0p083.tif'
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

# count grid points per koeppen classe
station_density = []
for i in range(30):
    #weights # TODO area per gridpoints
    n_gridpoints = (koeppen == i).sum().item()
    n_stations = (np.array(koeppen_class) == i).sum()
    station_density.append(n_stations / n_gridpoints)

# plot station density per koeppen climate map
density = xr.full_like(koeppen, np.nan)
for i in range(30):
    density = density.where(koeppen != i, station_density[i])

# plot
legend = pd.read_csv('koeppen_legend.txt', delimiter=';')
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(131, projection=proj)
ax2 = fig.add_subplot(132, projection=proj)
ax3 = fig.add_subplot(133)
koeppen.plot(ax=ax1, add_colorbar=False, cmap='terrain', transform=proj)
ax1.scatter(stations_lon, stations_lat, marker='x', s=5, c='indianred', transform=proj)
density.plot(ax=ax2, add_colorbar=False, cmap='hot_r', transform=proj)
ax1.set_title('koeppen climate classes and ISMN stations')
ax2.set_title('station density per koeppen class')
ax1.coastlines()
ax2.coastlines()
ax3.hist(koeppen_class, bins=np.arange(30), align='left')
ax3.set_xticks(np.arange(30))
ax3.set_xticklabels(legend.Short.values, rotation=90)
ax3.set_xlabel('koeppen climate classes')
ax3.set_ylabel('number of stations')
plt.savefig('koeppen_worldmap.png')
#plt.show()


# plot hist
#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
#koeppen.plot(ax=ax[0], add_colorbar=False, cmap='terrain')
#ax[0].set_title('koeppen climate classes and ISMN stations')
#ax[0].set_xticklabels('')
#ax[0].set_yticklabels('')
#ax[0].scatter(stations_lon, stations_lat, marker='x', s=5, c='indianred')
#density.plot(ax=ax[1], add_colorbar=False, cmap='hot_r')
###ax[1].hist(koeppen_class, bins=np.arange(30), align='left')
##ax[1].bar(range(30), station_density)
##ax[1].set_ylabel('stations per gridpoint')
##plt.savefig('koeppen_ismn_pergridpoint.png')
#plt.show()
