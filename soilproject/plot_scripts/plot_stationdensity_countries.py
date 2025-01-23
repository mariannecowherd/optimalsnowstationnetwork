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

# select surface layer soil moisture
lat = df.lat.groupby('station_id').first()
lon = df.lon.groupby('station_id').first()
network = df.network.groupby('station_id').first()
country = df.country.groupby('station_id').first()
df = df.groupby('station_id').first()
df = df.assign_coords(network=('station_id',network.data))
df = df.assign_coords(country=('station_id',country.data))
df = df.assign_coords(lat=('station_id',lat.data))
df = df.assign_coords(lon=('station_id',lon.data))
df = df.sortby('country')

# area per grid point
#res = 0.01
res = 0.1
grid = xr.Dataset({'lat': (['lat'], np.arange(-90,90,res)),
                   'lon': (['lon'], np.arange(-180,180,res))})
shape = (len(grid.lat),len(grid.lon))
earth_radius = 6371*1000 # in m
weights = np.cos(np.deg2rad(grid.lat))
area = earth_radius**2 * np.deg2rad(res) * weights * np.deg2rad(res)
area = np.repeat(area.values, shape[1]).reshape(shape)
grid['area'] = (['lat','lon'], area)

# ar6 regions
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(grid.lon, grid.lat)
regions = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(grid.lon, grid.lat)
regions = regions.where(~np.isnan(landmask))

# assign each station its ar6 region
station_regions = np.full((df.shape[1]), np.nan)
for i, (lat, lon) in enumerate(zip(df.lat, df.lon)):
    region = regions.sel(lat=lat.item(), lon=lon.item(), method='nearest').item()
    station_regions[i] = region

# histogram info
res = []
region_names = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.names
region_no = []
for region, name in enumerate(region_names):
    no_stations = (station_regions == region).sum() # unitless
    area_region = grid['area'].where(regions == region).sum().values.item() / (1000*1000) # km**2
    area_region = area_region / (1000*1000) # Mio km**2
    if no_stations != 0:
        res.append(no_stations / area_region)
    else:
        res.append(0)
    region_no.append(region)
    print(region, no_stations / area_region)
res = np.array(res)
region_no = np.array(region_no)
region_names = np.array(region_names)

# select only countries with stations
region_names = region_names[res != 0]
region_no = region_no[res != 0]
res = res[res != 0]
n = len(res)

# histogram plot
from matplotlib.lines import Line2D
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.bar(np.arange(n), res)
ax.set_xticks(np.arange(n))
ax.set_xticklabels(region_names, rotation=90)
ax.set_xlabel('country')
ax.set_ylabel('station density [1 station per $x^2 km^2$]')
ax.set_title('(c) station density per country')
ax.hlines(182, -10, 100, colors='orange')
ax.hlines(84, -10, 100, colors='brown')
ax.set_xlim([0,n+1])
ax.set_ylim([0,2000])

# calc density
density = xr.full_like(regions, 0)
for region, d in zip(region_no, res):
    density = density.where(regions != region, d) # unit stations per bio square km
density = density.where(~np.isnan(landmask))

# plot
fs = 25
cmap = plt.get_cmap('Greens').copy()
bad_color = 'lightgrey'
cmap.set_under(bad_color)
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
levels = np.arange(0,220,20)
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection=proj)
#cbar_kwargs = {'label': 'stations per million $km^2$'}
#landmask.plot(ax=ax, add_colorbar=False, cmap='binary', transform=transf, vmin=0, vmax=10)
im = density.plot(ax=ax, add_colorbar=False, cmap=cmap, transform=transf, 
                  vmin=1, vmax=200, levels=levels)
regionmask.defined_regions.natural_earth_v5_0_0.countries_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False, proj=transf)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('stations per million $km^2$', fontsize=fs)
ax.set_title('(b) station density per country', fontsize=fs) 
ax.coastlines()
ax.set_facecolor('aliceblue')
#plt.show()
print('h')
plt.savefig('stationdensity_country.png')
