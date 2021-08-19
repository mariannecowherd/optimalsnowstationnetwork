import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# TODO add QF removed values as grey points?

# open data map
largefilepath = '/net/so4/landclim/bverena/large_files/'
data = xr.open_dataarray(f'{largefilepath}df_gaps.nc')
data = data.sortby('country')

# simplify some countries to continents for less labels on x axis
countrydict = {'Alaska': 'USA',
               'Austria': 'Europe',
               'Denmark': 'Europe',
               'Finland': 'Europe',
               'France': 'Europe',
               'Germany': 'Europe',
               'Italy': 'Europe',
               'Poland': 'Europe',
               'Romania': 'Europe',
               'Spain': 'Europe',
               'UK': 'Europe',
               'Benin,Niger,Mali': 'Africa',
               'CotedIvoire,Nigeria,Ghana,Uganda,Rwanda,Kenya': 'Africa',
               'Senegal': 'Africa'}
continent = []
for c in data.country:
    try:
        continent.append(countrydict[c.item()])
    except KeyError:
        continent.append(c.item())
data = data.assign_coords(continent=('stations',continent))
data = data.sortby('continent')

# calculate anomalies
mean = data.mean(dim='time')
std = data.std(dim='time')
data = (data - mean) / std

# plot
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(111)
fs = 25
fig.suptitle('temporal coverage ISMN stations', fontsize=fs)
ax.imshow(np.flipud(~np.isnan(data.values)), vmin=-2, vmax=2, cmap='coolwarm_r', aspect=0.8)

# set ticklabels
xticks = [0] + np.cumsum(np.unique(data.continent.values, return_counts=True)[1])[:-1].tolist()
xticklabels = np.unique(data.continent.values)
# remove ticklabels overlapping with small sample size
xticklabels[3] = ''
xticklabels[7] = ''
xticklabels[9] = ''
xticklabels[11] = ''
xticklabels[12] = ''
xticklabels[-2] = ''
yticks = np.arange(0,732,12*5)
yticklabels = np.arange(1960,2021,5)[::-1]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=15)
ax.set_yticklabels(yticklabels, fontsize=15)
plt.subplots_adjust(bottom=0.4, left=0.05, right=0.95)
#data.plot(ax=ax, vmin=-2, vmax=2, cmap='coolwarm_r')
plt.savefig('stationtime.png')
#plt.show()
