import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

# load data
largefilepath = '/net/so4/landclim/bverena/large_files/'
modelname = 'CanESM5'
experimentname = 'ssp585'
ensemblename = 'r1i1p1f1'
ifut = True
if ifut:
    orig = xr.open_dataset(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc')
    pred = xr.open_dataset(f'{largefilepath}mrso_upscale_{modelname}_future_{ensemblename}.nc')
    new = xr.open_dataset(f'{largefilepath}mrso_newstations_{modelname}_future_{ensemblename}.nc')
    benchmark = xr.open_dataset(f'{largefilepath}mrso_benchmark_{modelname}_future_{ensemblename}_test.nc')

    # regrid pred and benchmark to worldmap
    tmp = xr.full_like(orig.sel(time=slice('2025','2035'))['mrso'], np.nan)
    tmp.loc[dict(lat=pred.lat,lon=pred.lon)] = pred['mrso']    
    pred = tmp

    tmp = xr.full_like(orig.sel(time=slice('2025','2035'))['mrso'], np.nan)
    tmp.loc[dict(lat=benchmark.lat,lon=benchmark.lon)] = benchmark['mrso']    
    benchmark = tmp

    tmp = xr.full_like(orig.sel(time=slice('2025','2035'))['mrso'], np.nan)
    tmp.loc[dict(lat=new.lat,lon=new.lon)] = new['mrso']    
    new = tmp

    # select identical time period
    orig = orig.sel(time=slice('2025','2035'))['mrso']
    pred = pred.sel(time=slice('2025','2035'))
    benchmark = benchmark.sel(time=slice('2025','2035'))
    new = new.sel(time=slice('2025','2035'))

else: # historical
    orig = xr.open_dataset(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc')['mrso']
    pred = xr.open_dataset(f'{largefilepath}mrso_fut_{modelname}_{experimentname}_{ensemblename}.nc')['mrso']
    experimentname = 'historical'
    benchmark = xr.open_dataset(f'{largefilepath}mrso_benchmark_{modelname}_{experimentname}_{ensemblename}_euler.nc')['mrso'] # debug USE EULER

    # select identical time period
    orig = orig.sel(time=slice('1960','2014'))
    pred = pred.sel(time=slice('1960','2014'))
    benchmark = benchmark.sel(time=slice('1960','2014'))

# normalise
def standardised_anom(data):
    data_seasonal_mean = data.groupby('time.month').mean()
    data_seasonal_std = data.groupby('time.month').std()
    data = (data.groupby('time.month') - data_seasonal_mean) 
    data = data.groupby('time.month') / data_seasonal_std
    return data
import IPython; IPython.embed()

orig = standardised_anom(orig)
#pred = standardised_anom(pred)
#benchmark = standardised_anom(benchmark)

proj = ccrs.PlateCarree()
fig = plt.figure()
fig.suptitle('Difference between RMSE of mean soil moisture anomaly (10% driest values globally) of CMIP-U and CMIP-B')
ax1 = fig.add_subplot(121, projection=proj)
ax2 = fig.add_subplot(122)

orig_drought = orig.where(orig < orig.quantile(0.1))
pred_drought = pred.where(pred < pred.quantile(0.1))
benchmark_drought = benchmark.where(benchmark < benchmark.quantile(0.1))
new_drought = new.where(new < new.quantile(0.1))

rmse_irreducible = np.sqrt(((orig_drought - benchmark_drought)**2).mean(dim='time'))
rmse_repr = np.sqrt(((orig_drought - pred_drought)**2).mean(dim='time'))
rmse_new = np.sqrt(((orig_drought - new_drought)**2).mean(dim='time'))
rmse_diff = rmse_repr - rmse_irreducible
rmse_diff1 = rmse_new - rmse_irreducible # TODO debug
rmse_diff.plot(ax=ax1, cbar_kwargs={'orientation': 'horizontal', 'label': 'RMSE difference'})
ax1.coastlines()
ax1.set_title('')

koeppen_simple = xr.open_dataset(f'{largefilepath}koeppen_simple.nc')['__xarray_dataarray_variable__']
import xesmf as xe
regridder = xe.Regridder(rmse_diff, koeppen_simple, 'bilinear', reuse_weights=False)
koeppen_names = ['Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df']
rmse_diff = regridder(rmse_diff)
rmse_diff1 = regridder(rmse_diff1)
koeppen_rmse = []
koeppen_rmse1 = []
n_koeppen = list(range(1,12))
koeppen_density = [3.3126, 2.4572, 3.5573, 9.6494, 23.995, 117.3616, 2.8483, 41.2927, 60.6654, 29.6423, 37.4495]
for k in n_koeppen:
    mae_tmp = rmse_diff.where(koeppen_simple == k, np.nan)
    koeppen_rmse.append(mae_tmp.mean().item())
    mae_tmp = rmse_diff1.where(koeppen_simple == k, np.nan)
    koeppen_rmse1.append(mae_tmp.mean().item())
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
#fig.suptitle('koeppen climate classes')
ax2.scatter(koeppen_density, koeppen_rmse, c='blue')
ax2.scatter(koeppen_density, koeppen_rmse1, c='green')
for i,(x,y) in enumerate(zip(koeppen_density,koeppen_rmse)):
    ax2.annotate(koeppen_names[i], xy=(x,y))
for i,(x,y) in enumerate(zip(koeppen_density,koeppen_rmse1)):
    ax2.annotate(koeppen_names[i], xy=(x,y))
ax2.set_ylabel('RMSE difference')
ax2.set_xlabel('station density [bio km^2]')
ax2.set_title('Per Koeppen climate')
plt.show()
import IPython; IPython.embed()
