"""
extract lat lon and time information from ISMN network
"""

import glob
from datetime import datetime
import numpy as np
import pandas as pd

# TODO add time of each station
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

df.to_csv(f'{largefilepath}station_info.csv')
