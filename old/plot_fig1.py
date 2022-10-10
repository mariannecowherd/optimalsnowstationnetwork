"""
Plot ISMN station locations and station density per Koppen-Geiger climate, per AR6 region and per country
"""

import numpy as np
import pandas as pd
import xarray as xr
import regionmask

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl

from calc_worldarea import calc_area

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

# landmask
landmask = xr.open_dataarray(f'{largefilepath}opscaling/landmask.nc')

# calc land area
grid = calc_area(landmask)

# get regions from regionmask
ar6regions = regionmask.defined_regions.ar6.land.mask(grid.lon, grid.lat) 
koeppenreg = xr.open_dataarray(f'{largefilepath}opscaling/koeppen_simple.nc')
countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(grid.lon, grid.lat)

# remove ocean
ar6regions = ar6regions.where(~np.isnan(landmask))
koeppenreg = koeppenreg.where(~np.isnan(landmask))
countries = countries.where(~np.isnan(landmask))

# assign each station its region
def assign_station(df, regions):
    station_regions = np.full((df.shape[1]), np.nan)
    for i, (lat, lon) in enumerate(zip(df.lat, df.lon)):
        region = regions.sel(lat=lat.item(), lon=lon.item(), method='nearest').item()
        station_regions[i] = region
    return station_regions

stations_ar6 = assign_station(df, ar6regions)
stations_koeppen = assign_station(df, koeppenreg)
stations_countries = assign_station(df, countries)

#  calculate station density per region
def calc_station_density(station_regions, regions, grid):
    res = []
    for region in range(int(regions.max().item())+1):
        no_stations = (station_regions == region).sum() # unitless
        area_region = grid.where(regions == region).sum().values.item() # / (1000*1000) # km**2
        if no_stations != 0:
            res.append(no_stations / area_region)
        else:
            res.append(0)
        print(f'region no {region}, station density {np.round(no_stations / area_region,2)} stations per Mio km^2')
    return res

density_ar6 = calc_station_density(stations_ar6, ar6regions, grid)
density_koeppen = calc_station_density(stations_koeppen, koeppenreg, grid)
density_countries = calc_station_density(stations_countries, countries, grid)

density_ar6 = np.round(density_ar6, 3)
density_countries = np.round(density_countries, 3)

# drop koeppen climates Ocean, ET, EF, BW: 0, 13, 12, 4
region_names = ['Af','Am','Aw','BS','Cs','Cw','Cf','Ds','Dw','Df']
density_koeppen = np.array(density_koeppen)[[1,2,3,5,6,7,8,9,10,11]]

# create world map
def to_worldmap(res, regions):
    density = xr.full_like(regions, np.nan)
    for region, d in zip(range(int(regions.max().item())), res):
        density = density.where(regions != region, d) # unit stations per bio square km
    density = density.where(landmask)
    return density

density_ar6 = to_worldmap(density_ar6, ar6regions)
density_countries = to_worldmap(density_countries, countries)

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

# set ocean negative number
density_ar6 = density_ar6.where(density_ar6 != 0, np.nan)

density_ar6 = density_ar6.where(landmask, -10)
density_countries = density_countries.where(landmask, -10)

# plot constants
fs = 25
cmap = plt.get_cmap('Greens').copy()
bad_color = 'lightgrey'
cmap.set_under('aliceblue')
cmap_koeppen = LinearSegmentedColormap.from_list('koeppen', colors, N=len(colors))
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
levels = np.arange(0,280,20)

# initiate figure
fig = plt.figure(figsize=(40,15))
gs = gridspec.GridSpec(2,3)

ax1 = fig.add_subplot(gs[0,:], projection=proj)
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1], projection=proj)
ax4 = fig.add_subplot(gs[1,2], projection=proj)

# plot koeppen worldmap
ax1.coastlines()
koeppenreg.plot(ax=ax1, cmap=cmap_koeppen, 
                add_colorbar=False, transform=transf)
ax1.scatter(df.lon, df.lat, transform=transf, c='black', marker='v', s=10)
ax1.set_title('(a) ISMN station network and Koppen-Geiger climates', fontsize=fs)
ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.03,1),
          fontsize=fs-5)

# plot koeppen barplot
n = len(density_koeppen)
ax2.barh(np.arange(n), density_koeppen[::-1], color='darkgreen')
ax2.set_yticks(np.arange(n))
ax2.set_yticklabels(region_names[::-1], fontsize=fs)
ax2.set_ylabel('Koppen-Geiger climate', fontsize=fs)
ax2.set_xlabel('station density [stations per Mio $km^2$]', fontsize=fs)
ax2.set_title('(b) station density per Koppen-Geiger climate', fontsize=fs)

# plot ar6 regions worldmap
im = density_ar6.plot(ax=ax3, add_colorbar=False, cmap=cmap, transform=transf, 
             vmin=0, vmax=280, levels=levels)
text_kws = dict(bbox=dict(color="none"), fontsize=fs-5,)
regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='lightcoral', linewidth=2), 
                    ax=ax3, label='abbrev', projection=transf, text_kws=text_kws)
ax3.set_title('(c) station density per AR6 region', fontsize=fs)
ax3.set_facecolor(bad_color)

# plot countries worldmap
density_countries.plot(ax=ax4, add_colorbar=False, cmap=cmap, transform=transf, 
             vmin=0, vmax=280, levels=levels)
text_kws = dict(bbox=dict(color="none"), fontsize=fs-5,)
regionmask.defined_regions.natural_earth_v5_0_0.countries_110.plot(line_kws=dict(color='lightcoral', linewidth=2), 
            ax=ax4, add_label=False, projection=transf)
ax4.set_title('(d) station density per country', fontsize=fs)
ax4.set_facecolor(bad_color)

# global plot args and save
cbar_ax = fig.add_axes([0.5, 0.09, 0.3, 0.03]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=fs)
cbar.set_label('station density [stations per million $km^2$]', fontsize=fs)
mpl.rcParams['xtick.labelsize'] = fs 
plt.savefig('fig1.png', dpi=300)
