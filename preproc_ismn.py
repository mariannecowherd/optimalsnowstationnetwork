"""
extract lat lon and time information from ISMN network
"""

import glob
import numpy as np

# TODO add time of each station
# TODO add vegetation of each station

ismnpath = '/net/exo/landclim/data/variable/soil-moisture/ISMN/20210211/point-scale_none_0.5h/original/'
largefilepath = '/net/so4/landclim/bverena/large_files/'
station_folders = glob.glob(f'{ismnpath}**/', recursive=True)
station_locations = []
for folder in station_folders:
    try:
        onefile = glob.glob(f'{folder}*.stm', recursive=True)[0]
    except IndexError: # list index out of range -> no .stm files in this directory
        continue # skip this directory
    with open(onefile, 'r') as f:
        firstline = f.readline()
    lat, lon = float(firstline.split()[3]), float(firstline.split()[4])
    station_locations.append([lat,lon])
station_locations = np.array(station_locations)  
station_locations.dump(largefilepath + 'ISMN_station_locations.npy')
