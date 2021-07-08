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
df = df[df.end > '2016-01-01'] # only stations still running
n_stations = xr.full_like(mmm[0,:,:].squeeze(), 0)
for lat, lon in zip(df.lat,df.lon):
    gridpoint = n_stations.sel(lat=lat, lon=lon, method='nearest')
    gridlat, gridlon = gridpoint.lat.item(), gridpoint.lon.item()
    n_stations.loc[gridlat, gridlon] = n_stations.loc[gridlat, gridlon] + 1

# plot number of stations per grid cell
#proj = ccrs.PlateCarree()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection=proj)
#ax.set_title('number of active stations per CMIP6-ng grid cell')
#n_stations.plot(cmap='Greens', ax=ax)
#ax.coastlines()
#plt.show()

# plot station density per grid cell
gridarea = xr.open_dataset(f'{largefilepath}gridarea_cmip6ng.nc')['cell_area']
gridarea.coords['lon'] = (gridarea.coords['lon'] + 180) % 360 - 180
gridarea = gridarea.sortby('lon')
gridarea = (gridarea/1000) / 1e6 # convert to mio km^2
station_density = n_stations/gridarea
proj = ccrs.PlateCarree()
fig = plt.figure()
ax = fig.add_subplot(111, projection=proj)
station_density.plot(cmap='Greens', ax=ax, cbar_kwargs={'label': 'stations per Mio km^2'})
#ax.scatter(df.lon, df.lat , marker='x', s=1)
ax.set_title('density of active stations per CMIP6-ng grid cell')
ax.coastlines()
plt.show()
import IPython; IPython.embed()

# select gridpoints with station density above certain threshold
tmp = station_density.stack(gridpoints=('lat','lon'))
tmp = tmp.where(tmp > 0.1, drop=True)
stationdata = xr.DataArray(np.full((mmm.shape[0], tmp.shape[0]), np.nan),
                   coords=[mmm.coords['time'], range(tmp.shape[0])], 
                   dims=['time','stations'])
stationdata = stationdata.assign_coords(lat=('stations',tmp.lat))
stationdata = stationdata.assign_coords(lon=('stations',tmp.lon))
for l, (lat, lon) in enumerate(zip(stationdata.lat, stationdata.lon)):
    stationdata.loc[:,l] = mmm.loc[:,lat,lon]

# create xarray with cmip6 data at stations
#def find_closest(list_of_values, number):
#    return min(list_of_values, key=lambda x:abs(x-number))
#station_grid_lat = []
#station_grid_lon = []
#df = df[df.end > '2016-01-01'] # include only stations still running
#for lat, lon in zip(df.lat,df.lon):
#    station_grid_lat.append(find_closest(mmm.lat.values, lat))
#    station_grid_lon.append(find_closest(mmm.lon.values, lon))
#coords_unique = np.unique(np.array([station_grid_lat, station_grid_lon]), axis=1)
#
#stationdata = xr.DataArray(np.full((mmm.shape[0], coords_unique.shape[1]), np.nan),
#                   coords=[mmm.coords['time'], range(coords_unique.shape[1])], 
#                   dims=['time','stations'])
#stationdata = stationdata.assign_coords(lat=('stations',coords_unique[0,:]))
#stationdata = stationdata.assign_coords(lon=('stations',coords_unique[1,:]))
#for s in range(coords_unique.shape[1]):
#    lat, lon = coords_unique[:,s]
#    mmm.sel(lat=lat, lon=lon)
#    stationdata[:,s] = mmm.sel(lat=lat, lon=lon)

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
trend.plot(cmap='coolwarm_r', ax=ax, vmin=-0.3, vmax=0.3)
plt.scatter(tmp.lon, tmp.lat, marker='x', s=1)
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
station_density = regridder(station_density)

# convert to simplified koeppen class
k_reduced = [0,1,2,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,13]
kdict = dict(zip(range(31),k_reduced))
koeppen.values = np.vectorize(kdict.get)(koeppen.values)

# select gridpoints with station density above certain threshold
tmp = station_density.stack(gridpoints=('lat','lon'))
tmp = tmp.where(tmp > 0.1, drop=True)
stationdata = xr.DataArray(np.full((mmm.shape[0], tmp.shape[0]), np.nan),
                   coords=[mmm.coords['time'], range(tmp.shape[0])], 
                   dims=['time','stations'])
stationdata = stationdata.assign_coords(lat=('stations',tmp.lat))
stationdata = stationdata.assign_coords(lon=('stations',tmp.lon))
koeppen_class = []
for l, (lat, lon) in enumerate(zip(stationdata.lat, stationdata.lon)):
    stationdata.loc[:,l] = mmm.loc[:,lat,lon]
    koeppen_class.append(koeppen.sel(lat=lat, lon=lon).item())
stationdata = stationdata.assign_coords(koeppen_simple=('stations',koeppen_class))

# plot trend per koeppen climate
from matplotlib.pyplot import cm
color=cm.rainbow(np.linspace(0,1,11))
reduced_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','ET','EF']
slope_stationdata = []
slope_alldata = []
for k in range(14):
    slope_alldata.append(_calc_slope(np.arange(86),mmm.where(koeppen == k).mean(dim=('lat','lon'))))
    slope_stationdata.append(_calc_slope(np.arange(86),stationdata.where(stationdata.koeppen_simple == k, drop=True).mean(dim='stations')))
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #mmm.where(koeppen == k).mean(dim=('lat','lon')).plot(ax=ax, c=color[i,:])
    #stationdata.where(stationdata.koeppen_simple == k, drop=True).mean(dim='stations').plot(ax=ax, c=color[i,:], linestyle='--')
    #plt.show()
#ax.legend(reduced_names[1:-2])
#plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(14), slope_alldata, marker='x')
ax.scatter(range(14), slope_stationdata, marker='x')
ax.set_title('fitted linear trend in mean soil moisture 2020-2100 in MMM CMIP6 SSP5 8.5')
ax.legend(['whole area of koeppen class', 'area of koeppen class with station density larger than threshold'])
ax.axhline()
ax.set_xticks(range(14))
ax.set_xticklabels(reduced_names)
ax.set_ylabel('fitted linear trend slope')
ax.set_xlabel('simplified koeppen-geiger climate class')
plt.show()

# scatter error in trend estimate with station density
error_estimate = np.abs(np.array(slope_alldata) - np.array(slope_stationdata))
density = [3.3126400217852,
             2.4572254418477,
             3.5573204219480,
             9.6494712179372,
             23.995941152118,
             117.36169912495,
             2.8483855557724,
             41.292729367904,
             60.665451036883,
             29.642355656222,
             37.449535740312,
             8.3342134994285,
             0.2127171364699]
reduced_names = ['Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','ET','EF']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(density, error_estimate[1:])
for n, name in enumerate(reduced_names):
    ax.annotate(name, xy=(density[n], error_estimate[1:][n]))
ax.set_xlabel('station density [stations per bio km^2]')
ax.set_ylabel('error in slope estimate from observed grid points only')
