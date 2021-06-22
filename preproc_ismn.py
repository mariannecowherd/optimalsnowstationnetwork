"""
extract lat lon and time information from ISMN network
"""

import glob
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd

# add vegetation of each station; done, funny vegetation descriptions
# use Martins apporach (see Email) # not done because less stations (soil moisture layer depths are not standardised)
# see for example "find . -type f" in ismnpath
# remove stations that are placed in ocean koeppen class; done, 27 stations removed, all coastal (no obvious wrong coords)
# add reduced koeppen class; done
# TODO see whether there are measurement gaps
# started with: 
# test = pd.read_csv(onefile, skiprows=1, header=None, delimiter='  ', engine='python', parse_dates=[0])
# test.columns = ['date','value']
# test = test.set_index('date')
## test.iloc[:,0] = [datetime.strptime(date, '%Y/%m/%d %H:%M').date() for date in test.iloc[:,0]]
## test.iloc[:,1] = [float(value.split(' ')[1]) for value in test.iloc[:,1]]
# test.value = [float(value.split(' ')[1]) for value in test.value]
# test = test.resample('1m').mean()

ismnpath = '/net/exo/landclim/data/variable/soil-moisture/ISMN/20210211/point-scale_none_0.5h/original/'
largefilepath = '/net/so4/landclim/bverena/large_files/'
station_folders = glob.glob(f'{ismnpath}**/', recursive=True)

# create empty pandas dataframe for data
print('find all station information')
index = range(2816)
columns = ['lat','lon','start','end', 'land_cover_class']
df = pd.DataFrame(index=index, columns=columns)
daterange = pd.date_range(start='1979-01-01', end='2021-01-01', freq='1m')
df_gaps = pd.DataFrame(index=daterange, columns=index)

i = 0
for folder in station_folders:
    try:
        onefile = glob.glob(f'{folder}*sm*.stm', recursive=True)[0]
    except IndexError: # list index out of range -> no .stm files in this directory
        continue # skip this directory
    print(folder) # DEBUG
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

    # get info on which months are measures per station
    test = pd.read_csv(onefile, skiprows=1, header=None, # first line is not header
                       delimiter=r'\s+', # delimiter any no of whitespaces 
                       engine='python', # future warnung suppressed
                       parse_dates=[0], # first row is dates convert to format
                       error_bad_lines=False) # ignore all lines with more columns
    test = test[[0,2]]
    test.columns = ['date','value']
    test = test.set_index('date')
    test = test.resample('1m').mean()
    df_gaps.loc[df_gaps.index.isin(test.index),i] = test.value

    # get metadata
    try:
        infofile = glob.glob(f'{folder}*static_variables.csv')[0]
    except IndexError: # file does not exist
        df.land_cover_class[i] = 'NaN'
    else:
        info = pd.read_csv(infofile, delimiter=';')
        try:
            df.land_cover_class[i] = str(info[info.quantity_name == 'land cover classification'].description.iloc[-1])
        except AttributeError: # column names are different
            df.land_cover_class[i] = 'NaN'

    # counter
    i += 1

# save as netcdf 
df_gaps = xr.DataArray(df_gaps, dims=['time','stations'])
df_gaps = df_gaps.assign_coords(lon=('stations',df.lon))
df_gaps = df_gaps.assign_coords(lat=('stations',df.lat))
df_gaps.to_netcdf(f'{largefilepath}df_gaps.nc')

# interpolate station locations on era5 grid
print('interpolate station locations on era5 grid')
filepath = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'
filename = f'{filepath}era5_deterministic_recent.lsm.025deg.time-invariant.nc'
data = xr.open_dataset(filename)
def find_closest(list_of_values, number):
    return min(list_of_values, key=lambda x:abs(x-number))
station_grid_lat = []
station_grid_lon = []
for lat, lon in zip(df.lat,df.lon):
    station_grid_lat.append(find_closest(data.lat.values, lat))
    station_grid_lon.append(find_closest(data.lon.values, lon))
df['lat_grid'] = station_grid_lat
df['lon_grid'] = station_grid_lon

# koeppen climate class per location
print('get koeppen class of station location')
filename = f'{largefilepath}Beck_KG_V1_present_0p0083.tif'
koeppen = xr.open_rasterio(filename)
koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()

koeppen_class = []
for lat, lon in zip(station_grid_lat,station_grid_lon):
    klat = find_closest(koeppen.lat.values, lat)
    klon = find_closest(koeppen.lon.values, lon)
    koeppen_class.append(koeppen.sel(lon=klon, lat=klat).values.item())
df['koeppen_class'] = koeppen_class

# calculate reduced koeppen classes
print('get simplified koeppen class of station location')
k_reduced = [0,1,2,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,13]
kdict = dict(zip(range(31),k_reduced))
stations_koeppen_class = []
for s, station in df.iterrows():
    stations_koeppen_class.append(kdict[station.koeppen_class])
df['simplified_koeppen_class'] = stations_koeppen_class

# remove stations that are in ocean koeppen class
df = df[df.koeppen_class != 0]

# save
print(df.head())
df.to_csv(f'{largefilepath}station_info_grid.csv')
