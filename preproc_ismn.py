"""
extract lat lon and time information from ISMN network
"""

import glob
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd

# TODO add vegetation of each station
# TODO use Martins apporach (see Email)

ismnpath = '/net/exo/landclim/data/variable/soil-moisture/ISMN/20210211/point-scale_none_0.5h/original/'
largefilepath = '/net/so4/landclim/bverena/large_files/'
station_folders = glob.glob(f'{ismnpath}**/', recursive=True)

# create empty pandas dataframe for data
index = range(2827)
columns = ['lat','lon','start','end']
df = pd.DataFrame(index=index, columns=columns)

i = 0
for folder in station_folders:
    print(folder)
    try:
        onefile = glob.glob(f'{folder}*.stm', recursive=True)[0]
    except IndexError: # list index out of range -> no .stm files in this directory
        continue # skip this directory
    with open(onefile, 'r') as f:
        lines = f.readlines()
    firstline = lines[0]
    first_entry = lines[2]
    last_entry = lines[-1]
    lat, lon = float(firstline.split()[3]), float(firstline.split()[4])
    station_start = datetime.strptime(first_entry.split()[0], '%Y/%m/%d')
    station_end = datetime.strptime(last_entry.split()[0], '%Y/%m/%d')
    df.lat[i] = lat
    df.lon[i] = lon
    df.start[i] = station_start
    df.end[i] = station_end
    i += 1

# interpolate station locations on era5 grid
filepath = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'
filename = f'{filepath}era5_deterministic_recent.lsm.025deg.time-invariant.nc'
data = xr.open_dataset(filename)
def find_closest(list_of_values, number):
    return min(list_of_values, key=lambda x:abs(x-number))
station_grid_lat = []
station_grid_lon = []
for lat, lon in zip(df.lat,df.lon):
    print(lat, lon)
    station_grid_lat.append(find_closest(data.lat, lat))
    station_grid_lon.append(find_closest(data.lon, lon))
df['lat_grid'] = station_grid_lat
df['lon_grid'] = station_grid_lon

# koeppen climate class per location
filename = f'{largefilepath}Beck_KG_V1_present_0p0083.tif'
koeppen = xr.open_rasterio(filename)
koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()

koeppen_class = []
for lat, lon in zip(station_grid_lat,station_grid_lon):
    print(lat, lon)
    koeppen_class.append(koeppen.sel(lon=lon, lat=lat).values.item())
df['koeppen_class'] = koeppen_class
station_coords.to_csv(f'{largefilepath}station_info_grid.csv')

quit()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
z,x,y = pltarr.values.nonzero()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, -z, zdir='z', c= 'red')
plt.show()

# plot 3D lat lon time plot!
for lat, lon, start, end in zip(df.lat, df.lon, df.start, df.end):
    pltarr.loc[slice(start,end),lat,lon] = 1
