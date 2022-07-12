import numpy as np
import xarray as xr
import regionmask
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

modelnames = ['HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR','IPSL-CM6A-LR',
              'ACCESS-ESM1-5','BCC-CSM2-MR','CESM2','CMCC-ESM2',
              'CNRM-ESM2-1','CanESM5','E3SM-1-1','FGOALS-g3',
              'GFDL-ESM4','GISS-E2-1-H','INM-CM4-8',#'MCM-UA-1-0', # MCM does not have pr
              'UKESM1-0-LL'] #NorESM does only have esm-ssp585
metrics = ['r2','seasonality','corr','trend']
method = 'systematic'
#model = 'HadGEM3-GC31-MM'
#metric = 'corr'
testcase = '_savemap'
for metric in metrics:
    for model in modelnames:
        try:
            data = xr.open_mfdataset(f'corrmap_???_{method}_{model}_{metric}{testcase}.nc',
                                     combine='nested', concat_dim='frac_observed')
        except OSError as e: 
            print(f'{model} {metric} {e}')
        regions = regionmask.defined_regions.ar6.land.mask(data.lon, data.lat)

        # spaghetti plot for regions
        for region in np.unique(regions):
            data.mrso.where(regions == region).mean(dim=('lat','lon')).plot()
        #plt.show()

        # map of doubling station density
        landmask = xr.open_dataarray('/net/so4/landclim/bverena/large_files/opscaling/landmask_cmip6-ng.nc')
        landmask = landmask.squeeze().drop(['time','month'])
        min_frac = min(data.mrso.frac_observed)
        double_frac = min_frac*2
        doublemap = xr.full_like(regions, np.nan)
        resmap = data.mrso.sel(frac_observed=double_frac, method='nearest') - data.mrso.sel(frac_observed=min_frac, method='nearest')
        for region in np.unique(regions):
            #mincorr = data.mrso.where(regions == region).mean(dim=('lat','lon')).sel(frac_observed=min_frac, method='nearest').values
            #doublecorr =  data.mrso.where(regions == region).mean(dim=('lat','lon')).sel(frac_observed=double_frac, method='nearest').values
            diff_frac = resmap.where(regions != region).mean()
            doublemap = doublemap.where(regions != region, diff_frac)
        doublemap = doublemap.where(landmask)
        doublemap = doublemap.drop_vars(['variable','band']).squeeze()
        doublemap.to_netcdf(f'doublemap_{model}_{metric}.nc')
        resmap.to_netcdf(f'resmap_{model}_{metric}.nc')

# plot result
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,5))
fig.suptitle(f'Increase of correlation with doubling of stations')
ax1 = fig.add_subplot(241, projection=proj)
ax2 = fig.add_subplot(242, projection=proj)
ax3 = fig.add_subplot(243, projection=proj)
ax4 = fig.add_subplot(244, projection=proj)
for ax, metric in zip((ax1,ax2,ax3,ax4),metrics):
    data = xr.open_mfdataset(f'doublemap*_{metric}.nc', combine='nested', concat_dim='model')

    data.mask.mean(dim='model').plot(ax=ax, cmap='Greens', transform=transf, vmin=0, vmax=0.3, add_colorbar=False)
    ax.coastlines()
    ax.set_global()
ax1.set_title('(a) Absolute values')
ax2.set_title('(b) Mean seasonal cycle')
ax3.set_title('(c) Anomalies')
ax4.set_title('(d) Trend')
plt.savefig('doubling.png')

# plot koeppen classes
ax5 = fig.add_subplot(245)
ax6 = fig.add_subplot(246)
ax7 = fig.add_subplot(247)
ax8 = fig.add_subplot(248)
koeppen_res = []
for ax, metric in zip((ax5,ax6,ax7,ax8),metrics):
    data = xr.open_mfdataset(f'resmap*_{metric}.nc', combine='nested', concat_dim='model')
    res = []
    for i in koeppen_ints:
        res.append(data.where(koeppen == i).mean().item())
    koeppen_res.append(res)
ax5.bar(np.arange(len(koeppen_res[0])), koeppen_res[0], color='darkred')
ax6.bar(np.arange(len(koeppen_res[1])), koeppen_res[1], color='darkred')
ax7.bar(np.arange(len(koeppen_res[2])), koeppen_res[2], color='darkred')
ax8.bar(np.arange(len(koeppen_res[3])), koeppen_res[3], color='darkred')
plt.show()
