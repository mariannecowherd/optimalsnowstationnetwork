import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# open data map
era5path = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1m/processed/regrid/'
years = list(np.arange(1979,2021))
varnames = ['swvl1','swvl2','swvl3']
filenames = [f'{era5path}era5_deterministic_recent.{varname}.025deg.1m.{year}.nc' for year in years for varname in varnames]
data = xr.open_mfdataset(filenames, combine='by_coords')
data = data.to_array().mean(dim='variable')
data = data.resample(time='1y').mean()

icalc = True
if icalc:
    # load feature space X
    print('load data')
    largefilepath = '/net/so4/landclim/bverena/large_files/'
    era5path_invariant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'
    invarnames = ['lsm','z','slor','cvl','cvh', 'tvl', 'tvh']
    freq = '1y'
    filenames_constant = [f'{era5path_invariant}era5_deterministic_recent.{varname}.025deg.time-invariant.nc' for varname in invarnames]
    filenames_variable = [f'{largefilepath}era5_deterministic_recent.temp.025deg.{freq}.max.nc', 
                         f'{largefilepath}era5_deterministic_recent.temp.025deg.{freq}.min.nc',
                         f'{largefilepath}era5_deterministic_recent.var.025deg.{freq}.mean.nc',
                         f'{largefilepath}era5_deterministic_recent.var.025deg.{freq}.roll.nc',
                         f'{largefilepath}era5_deterministic_recent.precip.025deg.{freq}.sum.nc']
    #filenames_variable = [f'{largefilepath}era5_deterministic_recent.var.025deg.{freq}.mean.nc']
    constant = xr.open_mfdataset(filenames_constant, combine='by_coords').load()
    variable = xr.open_mfdataset(filenames_variable, combine='by_coords').load()
    landlat, landlon = np.where((constant['lsm'].squeeze() > 0.8).load()) # land is 1, ocean is 0
    #landlat, landlon = [], []
    #for ilat, ilon in zip(idxlat, idxlon):
    #    landlat.append(constant['lsm'].squeeze()[ilat,ilon].lat.item())
    #    landlon.append(constant['lsm'].squeeze()[ilat,ilon].lon.item())
    #constant = constant.isel(lon=xr.DataArray(landlon, dims='landpoints'),
    #                         lat=xr.DataArray(landlat, dims='landpoints')).squeeze()
    #variable = variable.isel(lon=xr.DataArray(landlon, dims='landpoints'),
    #                         lat=xr.DataArray(landlat, dims='landpoints')).squeeze()
    #constant['latdat'] = ('landpoints', constant.lat.values)
    #constant['londat'] = ('landpoints', constant.lon.values)
    constant = constant.squeeze()

    # normalise data # for now omitted because expensive
    #print('normalise data')
    #mean = data.mean(dim='time')
    #std = data.std(dim='time')
    #data = (data - mean) / std
    #
    #mean = variable.mean(dim='time')
    #std = variable.std(dim='time')
    #variable = (variable - mean) / std

    # stack constant maps and merge with variables
    print('stack data')
    ntimesteps = variable.coords['time'].size # TODO use climfill package
    constant = constant.expand_dims({'time': ntimesteps}, axis=1)
    constant['time'] = variable['time']
    variable = variable.merge(constant)

    # select only station points in X (variable) data
    #print('select stations')
    #import IPython; IPython.embed()
    #variable = variable.sel(lat=data.lat_grid, lon=data.lon_grid)

    # rf settings
    n_trees = 50
    kwargs = {'n_estimators': n_trees,
              'min_samples_leaf': 2,
              'max_features': 'auto', 
              'max_samples': None, 
              'bootstrap': True,
              'warm_start': True,
              'n_jobs': 50, # set to number of trees
              'verbose': 0}

    # stack dimensions time and stations and remove not measured time points per station
    import IPython; IPython.embed()
    soil = data.isel(lat=xr.DataArray(landlat, dims='landpoints'),
                     lon=xr.DataArray(landlat, dims='landpoints'))
    variable = variable.isel(lat=xr.DataArray(landlat, dims='landpoints'),
                             lon=xr.DataArray(landlat, dims='landpoints'))
    soil = soil.stack(datapoints=('landpoints','time'))
    variable = variable.stack(datapoints=('landpoints','time')).to_array().T
    pred = xr.full_like(soil, np.nan)
    #stations = np.arange(0, 522547200, 504)
    #soil = soil.where(~np.isnan(soil), drop=True)
    #variable = variable.where(~np.isnan(soil), drop=True).to_array().T
    landpoints = soil.landpoints.values
    soil = soil.values # numpy is faster
    variable = variable.values

    tmp = []

    #for l in range(len(landlat)):
    #for s in range(len(stations)):
    for landpoint in range(len(landlat)):
        now = datetime.now()
        print(now, landpoint)
        #start, end = stations[s], stations[s] + 504
        landpoint_mask = landpoints != landpoint
        idx_test, idx_train = np.where(~landpoint_mask), np.where(landpoint_mask) # index-list indexing is faster than boolean indexing. source https://stackoverflow.com/questions/57783029/is-indexing-with-a-bool-array-or-index-array-faster-in-numpy-pytorch

        # define test and train data
        #y_test = soil[idx_test]
        y_train = soil[idx_train]
        X_test = variable[idx_test,:].squeeze()
        X_train = variable[idx_train,:].squeeze()
        #y_train = soil.where(soil.landpoints != landpoint, drop=True)
        #X_train = variable[:,:start], variable[:,end:]
        #X_test = variable[:,start:end]
        #y_train = data[:,:start], data[:,end:]
        #X_test = variable.loc[lon,lat]
        #y_test = data.loc[:,lat,lon]
        #X_train = variable.where((variable.lat == lat) & (variable.lon == lon), drop=True)
        #X_test = variable.where((variable.lat != lat) & (variable.lon != lon), drop=True)
        #y_train = data.where((data.lat == lat) & (data.lon == lon), drop=True).squeeze()
        #y_test = data.where((data.lat != lat) & (data.lon != lon), drop=True).squeeze()

        #X_test = variable.where(soil.stations == station, drop=True)

        #y_train = soil.where(soil.stations != station, drop=True)
        #X_train = variable.where(soil.stations != station, drop=True)
        #if X_test.size < 1:
        #    tmp.append(s)
        #    print(f'station {station.item()} skipped no values')
        #    continue

        # train QRF
        rf = RandomForestRegressor(**kwargs)
        import IPython; IPython.embed()
        rf.fit(X_train, y_train)

        y_predict = rf.predict(X_test)
        import IPython; IPython.embed()
        pred.loc[X_test.time,station] = y_predict
        #res = np.zeros((n_trees, X_test.shape[0]))
        #for t, tree in enumerate(rf.estimators_):
        #    res[t,:] = tree.predict(X_test)
        #mean = np.mean(res, axis=0)
        #upper, lower = np.percentile(res, [95 ,5], axis=0)

    pred.to_netcdf(f'{largefilepath}pred_era.nc') # TODO DEBUG REMOVE
else:
    pred = xr.open_dataset(f'{largefilepath}pred_era.nc')
    pred = pred['__xarray_dataarray_variable__']

# calculate RMSE of CV # TODO data not yet normalised
#rmse = np.sqrt(((data - pred)**2).mean(dim='time'))
pcorr = xr.corr(data,pred, dim='time')

# plot
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection=proj)
ax.set_title('Correlation of leave-one-out-CV and original station data')
im = ax.scatter(pred.lon, pred.lat, c=pcorr.values, marker='o', s=2, cmap='autumn_r', vmin=0, vmax=1)
cax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('correlation')
ax.coastlines()
plt.savefig('leave_one_out.png')
plt.close()

# plot per koeppen climate
koeppen_rmse = []
koeppen_minmax = np.zeros((2,14))
for i in range(14):
    tmp = pcorr.where(data.koeppen_simple == i)
    tmp_median = tmp.median().item()
    koeppen_rmse.append(tmp_median)
    koeppen_minmax[:,i] = tmp_median - np.nanpercentile(tmp, 10), np.nanpercentile(tmp,90) - tmp_median
reduced_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','ET','EF']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(reduced_names, koeppen_rmse)
ax.set_ylabel('median correlation')
ax.set_ylim([-0.1,1])
plt.savefig('LOO-CV_koeppen.png')
plt.close()

# scatter with station density
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
 37.449535740312]

koeppen_rmse = koeppen_rmse[1:-2]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0,1])
ax.scatter(density, koeppen_rmse)
#ax.errorbar(density, koeppen_rmse, yerr=koeppen_minmax[:,1:], fmt='none')
for n, name in enumerate(reduced_names[1:-2]):
    ax.annotate(name, xy=(density[n], koeppen_rmse[n]))
ax.set_ylabel('median correlation')
ax.set_xlabel('station density [station per bio km^2]')
plt.show()
