"""
number of stations per year vs prediction uncertainty and error
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

largefilepath = '/net/so4/landclim/bverena/large_files/'
stations = pd.read_csv(f'{largefilepath}station_info.csv')

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

# get values per year
pred = f'{largefilepath}RFpred_yearly.nc'
orig = f'{largefilepath}ERA5_yearly.nc'
unc = f'{largefilepath}UncPred_yearly.nc'
pred = xr.open_dataarray(pred)
orig = xr.open_dataarray(orig)
pred = (pred - orig)
unc = xr.open_dataarray(unc)
unc = unc.mean(dim=('lat','lon'))
pred = pred.mean(dim=('lat','lon'))

# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
ax.plot(no_stations)
ax.plot(pred.values*100)
ax.plot(unc.values*100)
ax.set_xticks(np.arange(0,80,10))
ax.set_xticklabels(np.arange(1950,2030,10))
ax.set_xlabel('year')
ax.set_ylabel('number of stations')
plt.show()
