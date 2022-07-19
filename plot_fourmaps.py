import glob
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import cftime
import cartopy.crs as ccrs
import pickle
import numpy as np

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221, projection=proj)
ax2 = fig.add_subplot(222, projection=proj)
ax3 = fig.add_subplot(223, projection=proj)
ax4 = fig.add_subplot(224, projection=proj)

largefilepath = '/net/so4/landclim/bverena/large_files/opscaling/'
#crop = rxr.open_rasterio(f'{largefilepath}Global_cropland_3km_2019.tif')
#crop = crop.rename({'x': 'lon', 'y': 'lat'})
#pop = xr.open_dataset(f'{largefilepath}gpw_v4_population_density_rev11_1_deg.nc').to_array()
#pop = pop.sel(raster=5) # pop density 2020 identifier
pop = xr.open_dataarray(f'{largefilepath}population_density_regridded.nc')
crop = xr.open_dataarray(f'{largefilepath}cropland_regridded.nc')

def rename_vars(data):
    varname = list(data.keys())[0]
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({varname: f'{modelname}_{ensemblename}'})
    try:
        data = data.drop_vars('file_qf')
    except ValueError:
        pass
    try:
        data = data.drop_dims('height')
    except ValueError:
        pass
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap):
        data['time'] = data.indexes['time'].to_datetimeindex()
    if isinstance(data.time[0].item(), cftime._cftime.Datetime360Day):
        data['time'] = data.indexes['time'].to_datetimeindex()
    return data

experimentname = 'ssp585'
ensemblename = '*i1*' 
cmip6_path = f'/net/atmos/data/cmip6-ng/mrso/mon/g025/'
filenames = glob.glob(f'{cmip6_path}mrso_mon_*_{experimentname}_{ensemblename}_*.nc')
data = xr.open_mfdataset(filenames, 
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars)  # rename vars to model name
data = data.to_array(dim='model').rename('mrso')
data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
data = data.sortby('lon')
data = data.sel(time=slice('2015','2050'))
data = data.mean(dim='model').load() # here for now
data = data.resample(time='1y').mean()
ms_to_year = 365*24*3600*10**9
trend = data.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year

crop.plot(ax=ax4, cmap='Greys', transform=transf)
pop.plot(ax=ax3, cmap='Greys', transform=transf)
trend.plot(ax=ax2, cmap='coolwarm_r', transform=transf)

method = 'systematic'
modelname = 'MIROC6'
metric = 'corr'
testcase = '_savemap'

with open(f"lats_systematic_{modelname}_{metric}_new.pkl", "rb") as f:
    latlist = pickle.load(f)

with open(f"lons_systematic_{modelname}_{metric}_new.pkl", "rb") as f:
    lonlist = pickle.load(f)

added = xr.full_like(trend, np.nan)
for l, (lats, lons) in enumerate(zip(latlist, lonlist)):
    for lat, lon in zip(lats, lons):
        added.loc[lat,lon] = l

added.plot(ax=ax1, cmap='Greys', transform=transf)

ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax4.coastlines()

ax1.set_global()
ax2.set_global()
ax3.set_global()
ax4.set_global()

plt.show()


obsmask = xr.open_dataarray(f'{largefilepath}obsmask.nc')
agpop = ((crop > 50) | (pop > 30))
doubling = xr.full_like(obsmask, False)
doubling = doubling.where(~obsmask, True)
doubling = (doubling | (added <= 2))

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221, projection=proj)
ax2 = fig.add_subplot(222, projection=proj)
ax3 = fig.add_subplot(223, projection=proj)
ax4 = fig.add_subplot(224, projection=proj)

agpop.plot(ax=ax1, transform=transf, cmap='Greys')
obsmask.plot(ax=ax2, transform=transf, cmap='Greys')
(agpop & obsmask).plot(ax=ax3, transform=transf, cmap='Greys')
doubling.plot(ax=ax4, transform=transf, cmap='Greys')

ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax4.coastlines()

ax1.set_title('(a) AgPop region')
ax2.set_title('(b) Currently observed area')
ax3.set_title('(c) AgPop currently observed')
ax4.set_title('(d) AgPop observed after doubling')

plt.show()
