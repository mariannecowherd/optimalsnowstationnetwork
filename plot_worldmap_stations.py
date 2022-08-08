"""
number of stations per year vs prediction uncertainty and error
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd

# open data
largefilepath = '/net/so4/landclim/bverena/large_files/'
data = xr.open_dataset(f'{largefilepath}df_gaps.nc')['mrso']
koeppen_simple = xr.open_dataset(f'{largefilepath}koeppen_simple.nc')['__xarray_dataarray_variable__']

# select the stations still active
inactive_networks = ['HOBE','PBO_H20','IMA_CAN1','SWEX_POLAND','CAMPANIA',
                     'HiWATER_EHWSN', 'METEROBS', 'UDC_SMOS', 'KHOREZM',
                     'ICN','AACES','HSC_CEOLMACHEON','MONGOLIA','RUSWET-AGRO',
                     'CHINA','IOWA','RUSWET-VALDAI','RUSWET-GRASS']
data = data.where(~data.network.isin(inactive_networks), drop=True)

# plot worldmap with all & still active stations on koeppen climates
colors = ['white', 'darkgreen', 'forestgreen', 'darkseagreen', 'linen', 'tan', 
          'gold', 'lightcoral', 'peru', 'yellowgreen', 'olive', 'olivedrab', 
          'lightgrey', 'whitesmoke']
fs = 25
cmap = LinearSegmentedColormap.from_list('koeppen', colors, N=len(colors))
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(21,15))
ax = fig.add_subplot(111, projection=proj)
ax.coastlines()
cbar_kwargs = {'orientation': 'horizontal', 'label': ''}
koeppen_simple.plot(ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, transform=transf)
ax.scatter(data.lon, data.lat, transform=transf, c='black', marker='v', s=10)
ax.set_title('ISMN station network', fontsize=fs)
plt.savefig('worldmap_stations.png')
