"""
number of stations per year vs prediction uncertainty and error
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

largefilepath = '/net/so4/landclim/bverena/large_files/'
stations = pd.read_csv(f'{largefilepath}station_info_grid.csv')
case = 'latlontime'

# get number of stations per year
start_year = []
end_year = []
for s in range(stations.shape[0]):
    start_year.append(int(stations.start[s][:4]))
    end_year.append(int(stations.end[s][:4]))

start_year = np.array(start_year)
end_year = np.array(end_year)

no_stations = []
for year in np.arange(1950,2021):
    no_stations.append(((year >= start_year) & (year <= end_year)).sum())

import cartopy.crs as ccrs
proj = ccrs.PlateCarree()
no_years = []
for start, end in zip(stations.start, stations.end):
    no_years.append(pd.date_range(start,end,freq='y').shape[0])

# plot statistcs on stations
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(131, projection=proj)
ax2 = fig.add_subplot(132, projection=proj)
ax3 = fig.add_subplot(133, projection=proj)
ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax1.set_title('length of observation period [years]')
ax2.set_title('start of observation period [year]')
ax3.set_title('end of observation period [year]')
ax1.scatter(stations.lon,stations.lat, c=no_years, cmap='Greens', s=0.5, vmin=1, vmax=40)
sc = ax2.scatter(stations.lon,stations.lat, c=start_year, cmap='Greens', s=0.5, vmin=1950, vmax=2021)
plt.colorbar(sc)
ax3.scatter(stations.lon,stations.lat, c=end_year, cmap='Greens', s=0.5, vmin=1950, vmax=2021)
plt.show()

# get values per year
pred = f'{largefilepath}RFpred_{case}.nc'
orig = f'{largefilepath}ERA5_{case}.nc'
unc = f'{largefilepath}UncPred_{case}.nc'
pred = xr.open_dataarray(pred)
orig = xr.open_dataarray(orig)
pred = (pred - orig)
unc = xr.open_dataarray(unc)
unc = unc.mean(dim=('lat','lon'))
pred = pred.mean(dim=('lat','lon'))

# add missing years
unc_t = np.zeros(len(no_stations))
pre_t = np.zeros(len(no_stations))
unc_t[29:65] = unc.values
pre_t[29:65] = pred.values

# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
# TODO plot this per area or koeppen 
ax.plot(no_stations)
ax.plot(unc_t*1000)
ax.plot(pre_t*1000)
ax.set_xticks(np.arange(0,80,10))
ax.set_xticklabels(np.arange(1950,2030,10))
ax.set_xlabel('year')
ax.set_ylabel('number of stations')
plt.show()
