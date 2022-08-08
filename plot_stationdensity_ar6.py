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

# ar6 regions
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(grid.lon, grid.lat)
regions = regionmask.defined_regions.ar6.land.mask(grid.lon, grid.lat) 
regions = regions.where(~np.isnan(landmask))

# assign each station its ar6 region
station_regions = np.full((df.shape[1]), np.nan)
for i, (lat, lon) in enumerate(zip(df.lat, df.lon)):
    region = regions.sel(lat=lat.item(), lon=lon.item(), method='nearest').item()
    station_regions[i] = region

#  calculate station density per region
res = []
for region in range(int(regions.max().item())):
    no_stations = (station_regions == region).sum() # unitless
    area_region = grid.where(regions == region).sum().values.item() # / (1000*1000) # km**2
    if no_stations != 0:
        res.append(no_stations / area_region)
    else:
        res.append(0)
    print(f'region no {region}, station density {np.round(no_stations / area_region,2)} stations per Mio km^2')

# create world map
density = xr.full_like(regions, 0)
for region, d in zip(range(int(regions.max().item())), res):
    density = density.where(regions != region, d) # unit stations per bio square km
density = density.where(~np.isnan(landmask))

# plot
cmap = plt.get_cmap('Greens').copy()
bad_color = 'lightgrey'
cmap.set_under(bad_color)
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection=proj)
im = density.plot(ax=ax, add_colorbar=False, cmap=cmap, transform=transf, 
             vmin=1, vmax=200)#, cbar_kwargs=cbar_kwargs)
regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False, proj=transf)
ax.set_title('station density per AR6 region') 
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('stations per million $km^2$')
ax.coastlines()
#plt.show()
