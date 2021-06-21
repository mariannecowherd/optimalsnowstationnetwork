"""
plot data climate classification and classification per station
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

# TODO remove ET and EF because of definition of soil moisture in frozen ground?

# open data map
case = 'latlontime'
largefilepath = '/net/so4/landclim/bverena/large_files/'
pred = f'{largefilepath}RFpred_{case}.nc'
orig = f'{largefilepath}ERA5_{case}.nc'
pred = xr.open_dataarray(pred)
orig = xr.open_dataarray(orig)
unc = f'{largefilepath}UncPred_{case}.nc'
unc = xr.open_dataarray(unc)
filename = f'{largefilepath}Beck_KG_V1_present_0p083.tif'
koeppen = xr.open_rasterio(filename)
koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()

# calculate rmse prediction error
rmse = np.sqrt(((pred - orig)**2).mean(dim='time'))

# regrid to koeppen grid
regridder = xe.Regridder(unc, koeppen, 'bilinear', reuse_weights=False)
unc = regridder(unc)
rmse = regridder(rmse)

# open station locations
stations = pd.read_csv(largefilepath + 'station_info_grid.csv')

# calculate reduced koeppen classes
k_reduced = [0,1,2,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,13]
reduced_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','ET','EF']
kdict = dict(zip(range(31),k_reduced))
stations_koeppen_class = []
for s, station in stations.iterrows():
    stations_koeppen_class.append(kdict[station.koeppen_class])
koeppen_class = np.array(stations_koeppen_class)

koeppen_reduced = xr.full_like(koeppen, np.nan)
for i in range(31):
   koeppen_reduced = koeppen_reduced.where(koeppen != i, kdict[i]) 
koeppen = koeppen_reduced
klist = np.unique(koeppen.values).tolist()

# data for boxplot per koeppen
mean_unc = []
mean_rmse = []
no_stations = []
for i in klist:

    tmp = unc.where(koeppen == i).values.flatten()
    tmp = tmp[~np.isnan(tmp)]
    #boxplot_unc.append(tmp)
    mean_unc.append(np.median(tmp))

    tmp = rmse.where(koeppen == i).values.flatten()
    tmp = tmp[~np.isnan(tmp)]
    #boxplot_unc.append(tmp)
    mean_rmse.append(np.median(tmp))

    no_stations.append((np.array(koeppen_class) == i).sum())

# remove ocean
labels = reduced_names
mean_rmse = mean_rmse[1:]
mean_unc = mean_unc[1:]
no_stations = no_stations[1:]
labels = labels[1:]

# remove frozen soil
labels = labels[:-2]
mean_rmse = mean_rmse[:-2]
mean_unc = mean_unc[:-2]
no_stations = no_stations[:-2]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
fig.suptitle('koeppen climate classes')
ax[0].scatter(mean_rmse, mean_unc, c='blue')
ax[1].scatter(no_stations, mean_unc, c='blue')
ax[2].scatter(no_stations, mean_rmse, c='blue')
for i,(x,y) in enumerate(zip(mean_rmse,mean_unc)):
    ax[0].annotate(labels[i], xy=(x,y))
for i,(x,y) in enumerate(zip(no_stations,mean_unc)):
    ax[1].annotate(labels[i], xy=(x,y))
for i,(x,y) in enumerate(zip(no_stations,mean_rmse)):
    ax[2].annotate(labels[i], xy=(x,y))
ax[0].set_ylabel('mean prediction uncertainty')
ax[0].set_xlabel('mean prediction RMSE')
ax[1].set_ylabel('mean prediction uncertainty')
ax[1].set_xlabel('number of stations')
ax[2].set_ylabel('mean prediction RMSE')
ax[2].set_xlabel('number of stations')
#plt.show()
plt.savefig(f'scatter_{case}.png')
