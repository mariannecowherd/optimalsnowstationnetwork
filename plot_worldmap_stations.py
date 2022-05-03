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

# calculate simplified koeppen classes
#koeppen = xr.open_rasterio(f'{largefilepath}Beck_KG_V1_present_0p5.tif')
#koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()
#koeppen_simple = xr.full_like(koeppen, np.nan)
#k_reduced = [0,1,2,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,13]
#kdict = dict(zip(range(31),k_reduced))
#for lat in koeppen.lat:
#    for lon in koeppen.lon:
#        koeppen_simple.loc[lat,lon] = kdict[koeppen.loc[lat,lon].item()]
#    print(lat.item())
#koeppen_simple.to_netcdf(f'{largefilepath}koeppen_simple.nc')

# create recommended koeppen colormap
#koeppen_legend = pd.read_csv('koeppen_legend.txt', delimiter=';', skipinitialspace=True)
#colors = []
#for color in koeppen_legend.Color:
#    clist = color[1:-1].split(' ')
#    clist = clist = (int(clist[0])/255., int(clist[1])/255., int(clist[2])/255.)
#    colors.append(clist)
#cmap = LinearSegmentedColormap.from_list('koeppen', colors, N=len(colors))
#import IPython; IPython.embed()

# select the stations still active
inactive_networks = ['HOBE','PBO_H20','IMA_CAN1','SWEX_POLAND','CAMPANIA',
                     'HiWATER_EHWSN', 'METEROBS', 'UDC_SMOS', 'KHOREZM',
                     'ICN','AACES','HSC_CEOLMACHEON','MONGOLIA','RUSWET-AGRO',
                     'CHINA','IOWA','RUSWET-VALDAI','RUSWET-GRASS']
data = data.where(~data.network.isin(inactive_networks), drop=True)

# plot worldmap with all & still active stations on koeppen climates
colors = ['white', 'darkgreen', 'forestgreen', 'darkseagreen', 'linen', 'tan', 'gold', 'lightcoral', 'peru', 'yellowgreen', 'olive', 'olivedrab', 'lightgrey', 'whitesmoke']
fs = 25
cmap = LinearSegmentedColormap.from_list('koeppen', colors, N=len(colors))
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(21,15))
ax = fig.add_subplot(111, projection=proj)
ax.coastlines()
cbar_kwargs = {'orientation': 'horizontal', 'label': ''}
koeppen_simple.plot(ax=ax, cmap=cmap, cbar_kwargs=cbar_kwargs, transform=transf)
ax.scatter(data.lon, data.lat, transform=transf, c='blue', marker='v', s=10)
ax.set_title('ISMN station network', fontsize=fs)
#ax.set_ticklabels(['a','b'])
plt.savefig('worldmap_stations.png')
