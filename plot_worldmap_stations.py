"""
number of stations per year vs prediction uncertainty and error
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd

# open data
largefilepath = '/net/so4/landclim/bverena/large_files/'
data = xr.open_dataset(f'{largefilepath}df_gaps.nc')['mrso']
koeppen_simple = xr.open_dataset(f'{largefilepath}opscaling/koeppen_simple.nc')
koeppen_simple = koeppen_simple.to_array()

# select the stations still active
inactive_networks = ['HOBE','PBO_H20','IMA_CAN1','SWEX_POLAND','CAMPANIA',
                     'HiWATER_EHWSN', 'METEROBS', 'UDC_SMOS', 'KHOREZM',
                     'ICN','AACES','HSC_CEOLMACHEON','MONGOLIA','RUSWET-AGRO',
                     'CHINA','IOWA','RUSWET-VALDAI','RUSWET-GRASS']
data = data.where(~data.network.isin(inactive_networks), drop=True)

# create koeppen legend
colors = ['white', 'darkgreen', 'forestgreen', 'darkseagreen', 'linen', 'tan', 
          'gold', 'lightcoral', 'peru', 'yellowgreen', 'olive', 'olivedrab', 
          'lightgrey', 'whitesmoke']
names = ['Active station', 'Af Tropical rainforest','Am Tropical monsoon','Aw Tropical savannah',
         'BW Arid desert','BS Arid steppe','Cs Temperate, dry summer',
         'Cw Temperate, dry winter','Cf Temperate, no dry season',
         'Ds Cold, dry summer','Dw Cold, dry winter', 'Df Cold, no dry season',
         'ET Polar tundra','EF Polar frost']
legend_station = Line2D([0], [0], marker='v', color='w', 
                 label='Active station',markerfacecolor='black', markersize=15)
legend_elements = [Patch(facecolor=col, label=lab) for col, lab in zip(colors[1:],names[1:])]                   
legend_elements = [legend_station] + legend_elements


# plot worldmap with still active stations on koeppen climates
fs = 25
cmap = LinearSegmentedColormap.from_list('koeppen', colors, N=len(colors))
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(30,10))
ax = fig.add_subplot(111, projection=proj)
ax.coastlines()
#cbar_kwargs = {'orientation': 'horizontal', 'label': ''}
#koeppen_simple.plot(ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, transform=transf)
koeppen_simple.plot(ax=ax, cmap=cmap, add_colorbar=False, transform=transf)
ax.scatter(data.lon, data.lat, transform=transf, c='black', marker='v', s=10)
ax.set_title('(a) ISMN station network and Koppen-Geiger climates', fontsize=fs)
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.03,1),
          fontsize=fs)
plt.savefig('worldmap_stations.png')
