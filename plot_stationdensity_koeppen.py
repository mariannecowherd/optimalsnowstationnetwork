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

# koeppen
filename = f'{largefilepath}Beck_KG_V1_present_0p083.tif'
koeppen = xr.open_rasterio(filename)
regions = koeppen.rename({'x':'lon','y':'lat'}).squeeze()

# area per grid point
#res = 0.01
res = 0.1
grid = xr.Dataset({'lat': (['lat'], regions.lat.data),
                   'lon': (['lon'], regions.lon.data)})
shape = (len(grid.lat),len(grid.lon))
earth_radius = 6371*1000 # in m
weights = np.cos(np.deg2rad(grid.lat))
area = earth_radius**2 * np.deg2rad(res) * weights * np.deg2rad(res)
area = np.repeat(area.values, shape[1]).reshape(shape)
grid['area'] = (['lat','lon'], area)

# ar6 regions
landmask = regionmask.defined_regions.natural_earth.land_110.mask(grid.lon, grid.lat)
regions = regions.where(~np.isnan(landmask))

# assign each station its ar6 region
station_regions = np.full((df.shape[1]), np.nan)
for i, (lat, lon) in enumerate(zip(df.lat, df.lon)):
    region = regions.sel(lat=lat.item(), lon=lon.item(), method='nearest').item()
    station_regions[i] = region

# histogram info
res = []
for region in range(31):
    print(region)
    no_stations = (station_regions == region).sum() # unitless
    area_region = grid['area'].where(regions == region).sum().values.item() / (1000*1000) # km**2
    area_region = area_region / (1000*1000) # Mio km**2
    if no_stations != 0:
        res.append(no_stations / area_region)
    else:
        res.append(0)
n = len(res)
legend = pd.read_csv('koeppen_legend.txt', delimiter=';')
region_names = legend.Short.values

# histogram plot
from matplotlib.lines import Line2D
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.bar(np.arange(n), res)
ax.set_xticks(np.arange(n))
ax.set_xticklabels(region_names, rotation=90)
ax.set_xlabel('AR6 region')
#ax.set_ylabel('station density [1 station per $x^2 km^2$]')
ax.set_ylabel('station density [stations per Mio $km^2$]')
ax.set_title('Station density')
#ax.hlines(182, -10, 100, colors='orange')
#ax.hlines(84, -10, 100, colors='brown')
#ax.set_xlim([0,n])
#ax.set_ylim([0,2000])
#kloster = [Line2D([0], [0], marker='s', color='white', linewidth=0, markersize=20, label='Kloster et al (2012)')]
#gruber = [Line2D([0], [0], marker='s', color='white', linewidth=0, markersize=20, label='Gruber et al (2018)')]
#ax.legend(handles=kloster, bbox_to_anchor=(0.9, 0.88), loc='center left', borderaxespad=0., frameon=False)
#ax.legend(handles=gruber, bbox_to_anchor=(0.9, 0.88), loc='center left', borderaxespad=0., frameon=False)
plt.show()

# create world map
density = xr.full_like(regions, 0)
for region, d in zip(range(31), res):
    density = density.where(regions != region, d) # unit stations per bio square km
density = density.where(~np.isnan(landmask))

# plot
cmap = plt.get_cmap('Greens').copy()
bad_color = 'lightsteelblue'
cmap.set_bad(bad_color)
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(111, projection=proj)
cbar_kwargs = {'label': 'stations per million $km^2$'}
#landmask.plot(ax=ax, add_colorbar=False, cmap='binary', transform=transf, vmin=0, vmax=10)
density.plot(ax=ax, add_colorbar=True, cmap=cmap, transform=transf, 
             vmin=0, vmax=200, cbar_kwargs=cbar_kwargs)
ax.set_title('station density per koeppen-geiger climate') 
ax.coastlines()
plt.show()
#plt.savefig('ar6_stationdensity.png')
