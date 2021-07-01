import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import xesmf as xe
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

largefilepath = '/net/so4/landclim/bverena/large_files/'

def _calc_slope(x, y):
    '''wrapper that returns the slop from a linear regression fit of x and y'''
    slope = stats.linregress(x, y)[0]  # extract slope only
    return slope

def linear_trend(obj):
    time_nums = xr.DataArray(obj['time'].values.astype(float),
                             dims='time',
                             coords={'time': obj['time']},
                           name='time_nums')
    trend = xr.apply_ufunc(_calc_slope, time_nums, obj,
                           vectorize=True,
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[[]],
                           output_dtypes=[float],
                           dask='parallelized')
    return trend

def rename_vars(data):
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({'mrso': f'{modelname}_{ensemblename}'})
    data = data.drop_dims('bnds')
    return data

cmip6path = '/net/atmos/data/cmip6-ng/mrso/ann/g025/'
ssp = 5
rcp = '85'
cmip = xr.open_mfdataset(f'{cmip6path}mrso*_ssp{ssp}{rcp}_*.nc', # shoyer says better use instead of glob
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars,  # rename vars to model name
                         join='left') # timestamp problem
cmip = cmip.to_array()
cmip.coords['lon'] = (cmip.coords['lon'] + 180) % 360 - 180
cmip = cmip.sortby('lon')

# calculate MM trend
mmm = cmip.mean(dim='variable').load()
mmm = (mmm - mmm.mean(dim = 'time')) / mmm.std(dim='time')
trend = linear_trend(mmm)
#mmm.mean(dim=('lat','lon')).plot()
#trend.plot(cmap='coolwarm_r')

# now only at stations
df = pd.read_csv(f'{largefilepath}station_info_grid.csv')

def find_closest(list_of_values, number):
    return min(list_of_values, key=lambda x:abs(x-number))
station_grid_lat = []
station_grid_lon = []
df = df[df.end > '2016-01-01'] # include only stations still running
for lat, lon in zip(df.lat,df.lon):
    station_grid_lat.append(find_closest(mmm.lat.values, lat))
    station_grid_lon.append(find_closest(mmm.lon.values, lon))
coords_unique = np.unique(np.array([station_grid_lat, station_grid_lon]), axis=1)

stationdata = xr.DataArray(np.full((mmm.shape[0], coords_unique.shape[1]), np.nan),
                   coords=[mmm.coords['time'], range(coords_unique.shape[1])], 
                   dims=['time','stations'])
stationdata = stationdata.assign_coords(lat=('stations',coords_unique[0,:]))
stationdata = stationdata.assign_coords(lon=('stations',coords_unique[1,:]))
for s in range(coords_unique.shape[1]):
    lat, lon = coords_unique[:,s]
    mmm.sel(lat=lat, lon=lon)
    stationdata[:,s] = mmm.sel(lat=lat, lon=lon)

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
trend.plot(cmap='coolwarm_r', ax=ax)
plt.scatter(coords_unique[1,:], coords_unique[0,:], marker='x', s=1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
mmm.mean(dim=('lat','lon')).plot(ax=ax)
stationdata.mean(dim='stations').plot(ax=ax)
plt.show()

# open koeppen and regrid mmm
filename = f'{largefilepath}Beck_KG_V1_present_0p5.tif'
koeppen = xr.open_rasterio(filename)
koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()
regridder = xe.Regridder(mmm, koeppen, 'bilinear', reuse_weights=False)
mmm = regridder(mmm)

# convert to simplified koeppen class
k_reduced = [0,1,2,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,13]
kdict = dict(zip(range(31),k_reduced))
koeppen.values = np.vectorize(kdict.get)(koeppen.values)

# get koeppen class for station pixel at koeppen resolution (way to do this more elegantly?)
def find_closest(list_of_values, number):
    return min(list_of_values, key=lambda x:abs(x-number))
station_grid_lat = []
station_grid_lon = []
df = df[df.end > '2016-01-01'] # include only stations still running
for lat, lon in zip(df.lat,df.lon):
    station_grid_lat.append(find_closest(mmm.lat.values, lat))
    station_grid_lon.append(find_closest(mmm.lon.values, lon))
coords_unique = np.unique(np.array([station_grid_lat, station_grid_lon]), axis=1)

stationdata = xr.DataArray(np.full((mmm.shape[0], coords_unique.shape[1]), np.nan),
                   coords=[mmm.coords['time'], range(coords_unique.shape[1])], 
                   dims=['time','stations'])
stationdata = stationdata.assign_coords(lat=('stations',coords_unique[0,:]))
stationdata = stationdata.assign_coords(lon=('stations',coords_unique[1,:]))
koeppen_class = []
for s in range(coords_unique.shape[1]):
    lat, lon = coords_unique[:,s]
    mmm.sel(lat=lat, lon=lon)
    stationdata[:,s] = mmm.sel(lat=lat, lon=lon)
    koeppen_class.append(koeppen.sel(lat=lat, lon=lon, method='nearest').item())
stationdata = stationdata.assign_coords(koeppen_simple=('stations',koeppen_class))

# plot trend per koeppen climate
from matplotlib.pyplot import cm
color=cm.rainbow(np.linspace(0,1,11))
reduced_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','ET','EF']
for k in range(14):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #mmm.where(koeppen == k).mean(dim=('lat','lon')).plot(ax=ax, c=color[i,:])
    #stationdata.where(stationdata.koeppen_simple == k, drop=True).mean(dim='stations').plot(ax=ax, c=color[i,:], linestyle='--')
    #plt.show()
#ax.legend(reduced_names[1:-2])
#plt.show()
