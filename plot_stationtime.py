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
fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('monthly aggregated soil moisture standardised anomalies for all ISMN stations')
ax.imshow(np.flipud(data.values), vmin=-2, vmax=2, cmap='coolwarm_r')

# set ticklabels
xticks = [0] + np.cumsum(np.unique(data.continent.values, return_counts=True)[1])[:-1].tolist()
xticklabels = np.unique(data.continent.values)
yticks = np.arange(0,732,12*5)
yticklabels = np.arange(1960,2021,5)[::-1]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(xticklabels, rotation=90)
ax.set_yticklabels(yticklabels)
#data.plot(ax=ax, vmin=-2, vmax=2, cmap='coolwarm_r')
plt.show()
