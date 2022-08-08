"""
plot station density per ar6 region
"""

import numpy as np
import pandas as pd
import xarray as xr
import regionmask

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# station data
largefilepath = '/net/so4/landclim/bverena/large_files/'
df = xr.open_dataset(f'{largefilepath}df_gaps.nc').load()
df = df['mrso']

# select the stations still active
inactive_networks = ['HOBE','PBO_H20','IMA_CAN1','SWEX_POLAND','CAMPANIA',
                     'HiWATER_EHWSN', 'METEROBS', 'UDC_SMOS', 'KHOREZM',
                     'ICN','AACES','HSC_CEOLMACHEON','MONGOLIA','RUSWET-AGRO',
                     'CHINA','IOWA','RUSWET-VALDAI','RUSWET-GRASS']
df = df.where(~df.network.isin(inactive_networks), drop=True)

# koeppen
regions = xr.open_dataarray(f'{largefilepath}opscaling/koeppen_simple.nc')

# area per grid point
res = np.abs(np.diff(regions.lat)[0]) # has to be resolution of "regions" for correct grid area calc
grid = xr.Dataset({'lat': (['lat'], regions.lat.data),
                   'lon': (['lon'], regions.lon.data)})
shape = (len(grid.lat),len(grid.lon))
earth_radius = 6371*1000 # in m
weights = np.cos(np.deg2rad(grid.lat))
area = earth_radius**2 * np.deg2rad(res) * weights * np.deg2rad(res)
area = np.repeat(area.values, shape[1]).reshape(shape)
grid['area'] = (['lat','lon'], area)
grid = grid.to_array() / (1000*1000) # to km**2
grid = grid / (1000*1000) # to Mio km**2

# assign each station its  koeppen climate
station_regions = np.full((df.shape[1]), np.nan)
for i, (lat, lon) in enumerate(zip(df.lat, df.lon)):
    region = regions.sel(lat=lat.item(), lon=lon.item(), method='nearest').item()
    station_regions[i] = region

# histogram info
res = []
for region in np.arange(1,12):
    no_stations = (station_regions == region).sum() # unitless
    area_region = grid.where(regions == region).sum().values.item()
    if no_stations != 0:
        res.append(no_stations / area_region)
    else:
        res.append(0)
    print(f'region no {region}, station density {np.round(no_stations / area_region,2)} stations per Mio km^2')
n = len(res)
region_names = ['Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df']

# histogram plot
fs = 25
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = fs 
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.barh(np.arange(n), res[::-1], color='darkgreen')
ax.set_yticks(np.arange(n))
ax.set_yticklabels(region_names[::-1], fontsize=fs)#, rotation=90) 
ax.set_ylabel('Koppen-Geiger climate', fontsize=fs)
ax.set_xlabel('station density [stations per Mio $km^2$]', fontsize=fs)
ax.set_title('(b) station density per Koppen-Geiger climate', fontsize=fs)
plt.savefig('stationdensity_koeppen.png')
