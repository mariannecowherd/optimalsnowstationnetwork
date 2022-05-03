"""
plot koeppen climate classification and classification per station
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# TODO
# save koeppen in all resolutiosn to nc and also simplified classes (6 nc files in total)

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

# open station locations
stations = pd.read_csv(largefilepath + 'station_info_grid.csv')
#stations = stations[stations.end > '2016-01-01'] # only stations still running

# optional: aggregate koeppen classes to first two letters
koeppen_reduced = xr.full_like(koeppen, np.nan)
k_reduced = [0,1,2,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,13]
reduced_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','ET','EF']
kdict = dict(zip(range(31),k_reduced))
for i in range(31):
   koeppen_reduced = koeppen_reduced.where(koeppen != i, kdict[i]) 
koeppen = koeppen_reduced
klist = np.unique(koeppen.values).tolist()
#stations_koeppen_class = stations.koeppen_class # all classes
stations_koeppen_class = []
for s, station in stations.iterrows():
    stations_koeppen_class.append(kdict[station.koeppen_class])
stations_koeppen_class = np.array(stations_koeppen_class)

# count grid points per koeppen classe
koeppen_station_density = []
koeppen_station_number = []
for i in klist:
    n_stations = (stations_koeppen_class == i).sum()
    area = gridarea.where(koeppen == i).sum().values.item() / (1000 * 1e9) # unit bio square km
    koeppen_station_density.append(float(n_stations) / area) # unit stations per bio square km
    koeppen_station_number.append(float(n_stations)) # unitless 
    #print(reduced_names[i], n_stations, area)
    print(reduced_names[i], float(n_stations) / area)

# plot station density per koeppen climate map
density = xr.full_like(koeppen, np.nan)
for i in klist:
    density = density.where(koeppen != i, koeppen_station_density[i])
    #proj = ccrs.PlateCarree()
    #fig = plt.figure(figsize=(10,10))
    #ax1 = fig.add_subplot(111, projection=proj)
    #density.where(koeppen != i, koeppen_station_density[i]).plot(ax=ax1)
    #ax1.scatter(stations.lon, stations.lat, marker='x', s=5, c='indianred', transform=proj)
    #ax1.set_title(f'{reduced_names[i]} {koeppen_station_number[i]}')
    #plt.show()
density = density.to_dataset(name='data')
density.to_netcdf(f'{largefilepath}koeppen_station_density.nc')
density = density.to_array()


# plot
legend = pd.read_csv('koeppen_legend.txt', delimiter=';')
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(131, projection=proj)
ax2 = fig.add_subplot(132, projection=proj)
ax3 = fig.add_subplot(133)
koeppen.plot(ax=ax1, add_colorbar=False, cmap='terrain', transform=transf)
ax1.scatter(stations.lon, stations.lat, marker='x', s=5, c='indianred', transform=transf)
density.plot(ax=ax2, add_colorbar=True, cmap='Greens', transform=transf, vmin=0, vmax=50)
#ax2.scatter(stations.lon, stations.lat, marker='x', s=5, c='indianred', transform=proj)
ax1.set_title('koeppen climate classes and ISMN stations')
ax2.set_title('station density per koeppen class \n[stations per billion km^2]')
ax1.coastlines()
ax2.coastlines()
ax3.bar(klist, koeppen_station_density)
#ax3.hist(koeppen_class, bins=np.arange(30), align='left')
ax3.set_xticks(np.arange(len(klist)))
ax3.set_xticklabels(reduced_names, rotation=90)
ax3.set_xlabel('koeppen climate classes')
ax3.set_ylabel('station density \n[stations per billion km^2]')
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
