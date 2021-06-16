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
gridarea = xr.open_dataset(f'{largefilepath}gridarea_koeppen0p083.nc')
koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()
gridarea = gridarea['cell_area']

# interlude: calculate grid area for koeppen
#import xesmf as xe
#lsmfile = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/era5_deterministic_recent.lsm.025deg.time-invariant.nc'
#lsm = xr.open_dataset(lsmfile)
#regridder = xe.Regridder(lsm, koeppen, 'bilinear', reuse_weights=False)
#lsm = regridder(lsm)
# then: add all lat lon arguments in console via ncatted -a long_name,lat,a,c,"latitude" lsmkoeppen.nc etc
# then use cdo gridarea lsmkoeppen.nc gridarea_koeppen0p083.nc

# optional: aggregate koeppen classes to first two letters
koeppen_reduced = xr.full_like(koeppen, np.nan)
k_reduced = [0,1,2,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,13]
kdict = dict(zip(range(31),k_reduced))
for i in range(31):
   koeppen_reduced = koeppen_reduced.where(koeppen != i, kdict[i]) 
koeppen_reduced.plot()
koeppen = koeppen_reduced
klist = np.unique(koeppen.values).tolist()

# open station locations
stations = pd.read_csv(largefilepath + 'station_info_grid.csv')

# count grid points per koeppen classe
koeppen_station_density = []
for i in klist:
    n_stations = (stations.koeppen_class == i).sum()
    area = gridarea.where(koeppen == i).sum().values.item() / (1000 * 1e9) # unit bio square km
    koeppen_station_density.append(n_stations / area) # unit stations per bio square km

# plot station density per koeppen climate map
density = xr.full_like(koeppen, np.nan)
for i in klist:
    density = density.where(koeppen != i, koeppen_station_density[i])

# plot
legend = pd.read_csv('koeppen_legend.txt', delimiter=';')
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(131, projection=proj)
ax2 = fig.add_subplot(132, projection=proj)
ax3 = fig.add_subplot(133)
koeppen.plot(ax=ax1, add_colorbar=False, cmap='terrain', transform=proj)
ax1.scatter(stations.lon, stations.lat, marker='x', s=5, c='indianred', transform=proj)
density.plot(ax=ax2, add_colorbar=False, cmap='hot_r', transform=proj)
ax1.set_title('koeppen climate classes and ISMN stations')
ax2.set_title('station density per koeppen class \n[stations per billion km^2]')
ax1.coastlines()
ax2.coastlines()
ax3.bar(klist, koeppen_station_density)
#ax3.hist(koeppen_class, bins=np.arange(30), align='left')
ax3.set_xticks(np.arange(30))
ax3.set_xticklabels(legend.Short.values, rotation=90)
ax3.set_xlabel('koeppen climate classes')
ax3.set_ylabel('station density \n[stations per billion km^2]')
#plt.savefig('koeppen_worldmap.png')
plt.show()


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
