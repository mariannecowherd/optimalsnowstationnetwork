"""
extract lat lon and time information from ISMN network
"""

import glob
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd

ismnpath = ('/net/exo/landclim/data/variable/soil-moisture/ISMN/'
             '20210211/point-scale_none_0.5h/original/')
largefilepath = '/net/so4/landclim/bverena/large_files/'
station_folders = glob.glob(f'{ismnpath}**/', recursive=True)

# create empty pandas dataframe for data
print('find all station information')
index = range(10930)
index = range(2816)
columns = ['lat','lon','start','end', 'land_cover_class', 'network', 
           'country','koeppen_class','simplified_koeppen_class']
df = pd.DataFrame(index=index, columns=columns)
daterange = pd.date_range(start='1960-01-01', end='2021-12-01', freq='1D')
df_gaps = pd.DataFrame(index=daterange, columns=index)

stations_without_valid_meas = []
list_lat = []
list_lon = []
list_land_cover_class = []
list_koeppen_class = []
list_depth_start = []
list_depth_end = []
list_network = []
station_id = []
list_stationname = []

i = 0
j = 0
for folder in station_folders:  # loop over stations

    filenames = glob.glob(f'{folder}*sm*.stm', recursive=True)
    filenames = sorted(filenames)
    filenames = filenames[:1] # DEBUG: only first soil layer

    for filename in filenames: # loop over obs within station
        #print(filename.split('/')[-1])

        # get coordinates of station
        with open(filename, 'r') as f:
            firstline = f.readline()
        lat, lon = float(firstline.split()[3]), float(firstline.split()[4])
        list_lat.append(lat)
        list_lon.append(lon)

        # get depth of measurement
        list_depth_start.append(float(filename.split('/')[-1].split('_')[4]))
        list_depth_end.append(float(filename.split('/')[-1].split('_')[5]))

        # get name of network
        list_network.append(firstline.split()[1])

        # get name of station
        list_stationname.append(firstline.split()[2])

        # numerate stations
        station_id.append(i)

        # open data table of observation
        station_obs = pd.read_csv(filename, skiprows=1, header=None, # first line is not header
                           delim_whitespace=True, # delimiter any no of whitespaces
                           parse_dates=[0], # first row is dates convert to format
                           usecols=range(4)) # only first four columns since 5th column
                                             # is network qf which may contain whitespaces
        station_obs.columns = ['date', 'hour' ,'value', 'qf_ismn']
        station_obs = station_obs.set_index('date')
        station_obs = station_obs[['value','qf_ismn']]

        # only entries where quality flag is 'good'
        #import IPython; IPython.embed()
        #station_obs = station_obs[station_obs.qf_ismn == 'G'] 
        #station_obs = station_obs[station_obs.qf_ismn.isin(['G','U','M'])]

        # calculate monthly means
        station_obs = station_obs.resample('1D').mean()
        print(i, j, station_obs.shape[0], filename.split('/')[-1])
# only first 28xx entries have values... something wrong in item settings?

        # check if any valid measurements are avail
        if station_obs.size == 0:
            stations_without_valid_meas.append(i)

        # write into xarray
        df_gaps.loc[df_gaps.index.isin(station_obs.index),j] = station_obs.value

        # get metadata
        try:
            infofile = glob.glob(f'{folder}*static_variables.csv')[0]
        except IndexError: # file does not exist
            list_land_cover_class.append('NaN')
            list_koeppen_class.append('NaN')
        else:
            info = pd.read_csv(infofile, delimiter=';')

            try:
                list_land_cover_class.append(str(info[info.quantity_name == 'land cover classification'].description.iloc[-1]))
            except AttributeError: # column name does not exist
                list_land_cover_class.append('NaN')

            try:
                list_koeppen_class.append(info[info.quantity_name == 'climate classification'].value.item())
            except (AttributeError, ValueError): # column name does not exist or more than one climate classification given
                list_koeppen_class.append('NaN')

        # counter for all meas
        j += 1
    # counter for all stations
    i += 1

# remove stations without valid measurements (from qf)
#df_gaps = df_gaps.T[~df_gaps.T.index.isin(stations_without_valid_meas)].T

# interpolate station locations on cmip6-ng grid
print('interpolate station locations on cmip6-ng grid')
filepath = f'/net/atmos/data/cmip6-ng/mrso/mon/g025/'
filename = f'{filepath}mrso_mon_CanESM5_historical_r1i1p1f1_g025.nc'
cmip6 = xr.open_dataset(filename)
cmip6.coords['lon'] = (cmip6.coords['lon'] + 180) % 360 - 180
cmip6 = cmip6.sortby('lon')
lat_cmip = []
lon_cmip = []
latlon_cmip = []
for lat, lon in zip(list_lat, list_lon):
    point = cmip6.sel(lat=lat, lon=lon, method='nearest')
    lat_cmip.append(point.lat.item())
    lon_cmip.append(point.lon.item())
    latlon_cmip.append(f'{point.lat.item()} {point.lon.item()}')

# translate koeppen class string into number
legend = pd.read_csv('koeppen_legend.txt', delimiter=';', skipinitialspace=True)
koeppen_no = []
for station in list_koeppen_class:
    if (station != 'NaN'):
        if (station != 'W'):
            koeppen_no.append(legend[legend.Short == station].No.item())
        else:
            koeppen_no.append(np.nan)
    else:
        koeppen_no.append(np.nan)
koeppen_no = np.array(koeppen_no)

# fill remaining koeppen class info from koeppen map
print('get koeppen class of station location')
filename = f'{largefilepath}Beck_KG_V1_present_0p0083.tif'
koeppen = xr.open_rasterio(filename)
koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()
koeppen_class = []
list_lat, list_lon = np.array(list_lat), np.array(list_lon)
for lat, lon in zip(list_lat[np.isnan(koeppen_no)], list_lon[np.isnan(koeppen_no)]):
    point = koeppen.sel(lat=lat, lon=lon, method='nearest')
    klat, klon = point.lat.item(), point.lon.item()
    koeppen_class.append(koeppen.sel(lon=klon, lat=klat).values.item())
list_koeppen_class = np.array(list_koeppen_class)
list_koeppen_class[np.isnan(koeppen_no)] = koeppen_class
koeppen_no[np.isnan(koeppen_no)] = koeppen_class

# calculate reduced koeppen classes
print('get simplified koeppen class of station location')
k_reduced = [0,1,2,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,13]
kdict = dict(zip(range(31),k_reduced))
simplified_koeppen_class = []
for station in koeppen_no:
    simplified_koeppen_class.append(kdict[station])

# add country to station
networks = pd.read_csv('ISMN_station_countries.txt', delim_whitespace=True)
networks = networks.set_index('Name')
countries = []
for net in list_network:
    countries.append(networks.loc[net].Country)

# to xarray
df_gaps = xr.DataArray(df_gaps, dims=['time','stations'])
df_gaps = df_gaps.assign_coords(lon=('stations',list_lon))
df_gaps = df_gaps.assign_coords(station_id=('stations',station_id))
df_gaps = df_gaps.assign_coords(lat=('stations',list_lat))
df_gaps = df_gaps.assign_coords(lat_cmip=('stations',lat_cmip))
df_gaps = df_gaps.assign_coords(lon_cmip=('stations',lon_cmip))
df_gaps = df_gaps.assign_coords(latlon_cmip=('stations',latlon_cmip))
df_gaps = df_gaps.assign_coords(koeppen=('stations',list_koeppen_class))
df_gaps = df_gaps.assign_coords(koeppen_simple=('stations',simplified_koeppen_class))
df_gaps = df_gaps.assign_coords(stationname=('stations',list_stationname))
df_gaps = df_gaps.assign_coords(network=('stations',list_network))
df_gaps = df_gaps.assign_coords(country=('stations',countries))
df_gaps = df_gaps.assign_coords(depth_start=('stations',list_depth_start))
df_gaps = df_gaps.assign_coords(depth_end=('stations',list_depth_end))

# remove ocean stations
df_gaps = df_gaps.where(df_gaps.koeppen != 0, drop=True)

# remove stations without valid measurements (from qf)
# TODO

# save as netcdf
df_gaps = df_gaps.rename('mrso')
df_gaps.to_netcdf(f'{largefilepath}df_gaps.nc')

# save cmip table as netcdf
df_gaps = df_gaps.groupby('latlon_cmip').mean()

# add lat and lon again to grouped station data
lat_cmip = []
lon_cmip = []
for latlon in df_gaps.latlon_cmip:
    lat, lon = latlon.item().split()
    lat, lon = float(lat), float(lon)
    lat_cmip.append(lat)
    lon_cmip.append(lon)
df_gaps = df_gaps.assign_coords(lat_cmip=('latlon_cmip',lat_cmip))
df_gaps = df_gaps.assign_coords(lon_cmip=('latlon_cmip',lon_cmip))

# save as netcdf
df_gaps.to_netcdf(f'{largefilepath}df_gaps_cmip.nc')
