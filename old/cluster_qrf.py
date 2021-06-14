"""
cluster landpoints along environmental similiarity, then count how many in-situ observations are in each cluster
quantile random forest from https://medium.com/dataman-in-ai/a-tutorial-on-quantile-regression-quantile-random-forests-and-quantile-gbm-d3c651af7516

    @author: verena bessenbacher
    @date: 08 03 2021
"""

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING) # no matplotlib logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import concurrent
import cartopy.crs as ccrs

# internal functions
def find_closest(list_of_values, number):
    return min(list_of_values, key=lambda x:abs(x-number))

def to_latlon(data):
    lsm = xr.open_dataset(largefilepath + f'landmask_idebug_{idebugspace}_None.nc')
    lsm = lsm.to_array().squeeze()
    shape = lsm.shape
    landlat, landlon = np.where(lsm)
    tmp = xr.DataArray(np.full((data.coords['time'].size,shape[0],shape[1]),np.nan), coords=[data.coords['time'], lsm.coords['latitude'], lsm.coords['longitude']], dims=['time','latitude','longitude'])
    tmp.values[:,landlat,landlon] = data
    return tmp

# internal settings
largefilepath = '/net/so4/landclim/bverena/large_files/'
plotpath = "/home/bverena/python/plots/"
idebugspace = False
varnames = ['skt', 'swvl1', 'tp', 'tws']

#import IPython; IPython.embed()

# read feature table
logging.info(f'read feature table...')
data = xr.open_dataset(largefilepath + f'features_init_None_None_idebug_{idebugspace}.nc').load()
data = data['data']

# read data coordinates
data_lat = np.unique(data['latitude'])
data_lon = np.unique(data['longitude'])

# read fluxnet station coordinates
logging.info(f'read fluxnet stations data...')
station_coords = pd.read_csv(largefilepath + 'fluxnet_station_coords.csv')
stations_lat = station_coords.LOCATION_LAT.values
stations_lon = station_coords.LOCATION_LONG.values

# find gridpoint where fluxnet station is
logging.info(f'find gridpoint that fluxnet station contains...')
station_grid_lat = []
station_grid_lon = []
for lat, lon in zip(stations_lat,stations_lon):
    station_grid_lat.append(find_closest(data_lat, lat))
    station_grid_lon.append(find_closest(data_lon, lon))

icalc = False
if icalc:

    # unstack data from (variable, datapoint) to (variable, time, landpoint) to (variable, time, latitude, longitude)
    data = data.set_index(datapoints=('time','landpoints')).unstack('datapoints')

    # renormalise globally
    datamean = xr.open_dataarray(largefilepath + f'datamean_None_None_idebug_{idebugspace}.nc').load()
    datastd = xr.open_dataarray(largefilepath + f'datastd_None_None_idebug_{idebugspace}.nc').load()
    datamean = datamean.sel(variable=varnames)
    datastd = datastd.sel(variable=varnames)
    data = data * datastd + datamean

    # normalise per grid point
    datamean = data.mean(dim=('time'))
    datastd = data.std(dim=('time'))
    data = (data - datamean) / datastd

    # find gridpoint where fluxnet station is
    logging.info(f'find gridpoint that fluxnet station contains...')
    station_grid_lat = []
    station_grid_lon = []
    for lat, lon in zip(stations_lat,stations_lon):
        station_grid_lat.append(find_closest(data_lat, lat))
        station_grid_lon.append(find_closest(data_lon, lon))

    lat_landpoints = data.latitude.values[0,:]
    lon_landpoints = data.longitude.values[0,:]
    selected_landpoints = []
    for l, (lat, lon) in enumerate(zip(station_grid_lat,station_grid_lon)):
        # get index of landpoint with this lat and lon
        try:
            selected_landpoints.append(np.intersect1d(np.where(lat_landpoints == lat)[0], np.where(lon_landpoints == lon)[0])[0])
        except:
            pass
    selected_landpoints = np.unique(selected_landpoints) # some grid points are chosen more than one because contain more than one FLUXNET station
    other_landpoints = np.arange(data.shape[2]).tolist()
    for pt in selected_landpoints:
        other_landpoints.remove(pt)

    # divide data into training data (where fluxnet stations are) and tsssesc data (where arent)
    # data.sel does not work on non-dim coordinates, data.where only makes them nan and does not reduce, np.in1d+reshape is too expensive. ooping it is
    logging.info(f'divide into test and train data...')
    train = data.isel(landpoints=selected_landpoints)
    test = data.isel(landpoints=other_landpoints)
    del(data)
    train = train.stack(datapoints=('time','landpoints')).reset_index('datapoints').T
    test = test.stack(datapoints=('time','landpoints')).reset_index('datapoints').T

    # take random variable for testing
    varname = 'swvl1'

    # divide into X and y
    logging.info(f'divide into X and y ...')
    notyvars = test.coords['variable'].values.tolist()
    notyvars.remove(varname)
    y_test = test.loc[:,varname]
    X_test = test.loc[:,notyvars]

    y_train = train.loc[:,varname]
    X_train = train.loc[:,notyvars]

    # train quantile random forest regressor on observed gridpoints
    logging.info(f'train all trees ...')
    from sklearn.ensemble import RandomForestRegressor
    n_trees = 100
    kwargs = {'n_estimators': n_trees,
              'min_samples_leaf': 2,
              'max_features': 0.5, 
              'max_samples': 0.5, 
              'bootstrap': True,
              'warm_start': True,
              'n_jobs': 100, # set to number of trees
              'verbose': 0}
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train, y_train)
    del(X_train)

    #logging.info(f'predict from all trees ...')
    #pred_trees = np.zeros((y_test.shape[0], n_trees))
    #for t, pred in enumerate(rf.estimators_):
    #    logging.info(f'predict tree {t}...')
    #    pred_trees[:,t] = pred.predict(X_test)

    # concurrent version
    logging.info(f'predict from all trees...')
    def predict_one_sample(trees, sample): # 5.41 ms +- 381 us per loop (mean +- std. dev. of 7 runs, 100 loops each)
        res = np.zeros(len(trees))
        for t, tree in enumerate(trees):
            res[t] = tree.predict(sample.values.reshape(1,-1))[0]
        upper, lower = np.percentile(res, [95 ,5])
        return upper - lower

    def predict_subsample(subsample, trees): 
        res = np.zeros((len(trees), subsample.shape[0]))
        for t, tree in enumerate(trees):
            res[t,:] = tree.predict(subsample)
        upper, lower = np.percentile(res, [95 ,5], axis=0)
        #arr = xr.full_like(subsample[:,0], np.nan) # subsample not a dataarray anymore by now
        #arr[:] = upper - lower
        #return arr
        return upper - lower
    #X_test = X_test.chunk({'datapoints': 20}) # change for full res
    #y_predict = xr.apply_ufunc(predict_subsample, X_test[:10,:], input_core_dims=[['datapoints','variable']],output_core_dims=[['datapoints']]) # takes too long (hours)

    y_predict = []
    for a, arr in enumerate(np.array_split(X_test, 12000, axis=0)): #12000
        y_predict.append(predict_subsample(arr, rf.estimators_))
        print(a)
    y_predict = np.concatenate(y_predict)

    y_train[:] = np.nan
    y_test[:] = y_predict
    res = xr.concat([y_test,y_train], dim='datapoints')

    # reshape to worldmap
    logging.info(f'reshape to worldmap...')
    res = res.set_index(datapoints=('time','landpoints')).unstack('datapoints')
    res = to_latlon(res)
    res.to_netcdf(largefilepath + f'optimal_res_idebug_{idebugspace}.nc')
    #quit()

    #y_predict = np.full_like(y_test, np.nan)
    #for s in range(y_predict.shape[0]):
    #    y_predict[s] = predict_one_sample(rf.estimators_, X_test[s,:])
    #    if (s % 1000000==0):
    #        print(s)

    #    
    #def one_tree(X_test, tree, t, idebugspace, res):
    #    logging.info(f'tree {t} started')
    #    res[:] = tree.predict(X_test)
    #    res.to_netcdf(largefilepath + f'res_tree_{t}_idebug_{idebugspace}.nc')
    #    #res.dump(largefilepath + f'res_tree_{t}_idebug_{idebugspace}.nc')
    #    del(res) # keep this otherwise memory fills up
    #    logging.info(f'tree {t} finished')
    #    return t 

    ##max_workers = 1
    ##with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #for t, tree in enumerate(rf.estimators_):
    #    kwargs = {'tree': tree,
    #              'X_test': X_test,
    #              'idebugspace': idebugspace,
    #              'res': xr.full_like(y_test, np.nan),
    #              't': t}
    #    #results.append(executor.submit(one_tree, **kwargs))
    #    one_tree(**kwargs)
    ##for result in results: 
    ##    print(result, result.result())
    #del(X_test)
    #y_trees = np.zeros((y_test.shape[0], n_trees))
    #for t in range(n_trees):
    #    y_trees[:,t] = xr.open_dataset(largefilepath + f'res_tree_{t}_idebug_{idebugspace}.nc').to_array()

    ## quantile accross 1st dim
    #logging.info(f'get quantile range...')
    #upper, lower = np.percentile(y_trees, [95 ,5], axis=1)
    #del(y_trees)
    #iqr = upper - lower
    #del(upper, lower)

    #y_train[:] = np.nan
    #y_test[:] = iqr
    #del(iqr)
    #res = xr.concat([y_test,y_train], dim='datapoints')
    #del(y_test, y_train)
    #res.to_netcdf(largefilepath + f'optimal_res_idebug_{idebugspace}.nc')
else:
    res = xr.open_dataarray(largefilepath + f'optimal_res_idebug_{idebugspace}.nc')

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection=proj)
ax.coastlines()
res.median(axis=0).plot(cmap='Greys', ax=ax, transform=transf)
plt.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred', transform=transf)
plt.savefig(plotpath + f'iqr_norm_idebug_{idebugspace}.png', dpi=300)
