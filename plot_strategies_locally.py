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
testcase = '_savemap'
largefilepath = '/net/so4/landclim/bverena/large_files/'

landmask = xr.open_dataarray(f'{largefilepath}opscaling/landmask_cmip6-ng.nc')
landmask = landmask.squeeze().drop(['time','month'])

for metric in metrics:
    for model in modelnames:
        data = xr.open_mfdataset(f'corrmap_???_{method}_{model}_{metric}{testcase}.nc',
                                 combine='nested', concat_dim='frac_observed')

        min_frac = min(data.mrso.frac_observed)
        double_frac = min_frac*2

        resmap = data.mrso.sel(frac_observed=double_frac, method='nearest') - \
                 data.mrso.sel(frac_observed=min_frac, method='nearest')
        import IPython; IPython.embed()
        resmap = resmap.drop(['band','variable'])
        resmap.to_netcdf(f'resmap_{model}_{metric}.nc')

# calc and plot ar6 regions
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(15, 7))
fig.suptitle(f'Increase of correlation with doubling of stations')
ax1 = fig.add_subplot(241, projection=proj)
ax2 = fig.add_subplot(242, projection=proj)
ax3 = fig.add_subplot(243, projection=proj)
ax4 = fig.add_subplot(244, projection=proj)
regions = regionmask.defined_regions.ar6.land.mask(data.lon, data.lat)
vmaxdict = {'trend':3,'r2':0.3,'corr':0.3,'seasonality':0.3}
ims = []
for ax, metric in zip((ax1,ax2,ax3,ax4),metrics):
    data = xr.open_mfdataset(f'resmap*_{metric}.nc', combine='nested', concat_dim='model').mrso
    doublemap = xr.full_like(regions, np.nan)
    for region in np.unique(regions):
        diff_frac = data.where(regions == region).mean().values
        doublemap = doublemap.where(regions != region, diff_frac)
    doublemap = doublemap.where(landmask)
    ims.append(doublemap.plot(ax=ax, cmap='Greens', transform=transf, vmin=0, vmax=vmaxdict[metric], 
                   add_colorbar=False))
    ax.coastlines()
    ax.set_global()
ax1.set_title('(a) Absolute values')
ax2.set_title('(b) Mean seasonal cycle')
ax3.set_title('(c) Anomalies')
ax4.set_title('(d) Trend')

cbar_ax_corr = fig.add_axes([0.16, 0.53, 0.5, 0.03]) # left bottom width height
cbar_ax_mae = fig.add_axes([0.77, 0.53, 0.1, 0.03]) # left bottom width height
cbar_corr = fig.colorbar(ims[0], cax=cbar_ax_corr, orientation='horizontal')
cbar_mae = fig.colorbar(ims[-1], cax=cbar_ax_mae, orientation='horizontal')
cbar_corr.set_label('difference in correlation')
cbar_mae.set_label('difference in mae')

# calc and plot koeppen classes
ax5 = fig.add_subplot(245)
ax6 = fig.add_subplot(246)
ax7 = fig.add_subplot(247)
ax8 = fig.add_subplot(248)
filename = f'{largefilepath}koeppen_simple.nc'
koeppen = xr.open_dataarray(filename)
koeppen_res = []
koeppen_classes = ['Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df']
koeppen_ints = np.arange(1,12).astype(int)
for ax, metric in zip((ax5,ax6,ax7,ax8),metrics):
    data = xr.open_mfdataset(f'resmap*_{metric}.nc', combine='nested', concat_dim='model').mrso.load().load()
    res = []
    for i in koeppen_ints:
        res.append(data.where(koeppen == i).mean().item())
    koeppen_res.append(res)

ax5.bar(np.arange(len(koeppen_res[0])), koeppen_res[0], color='darkgreen')
ax6.bar(np.arange(len(koeppen_res[1])), koeppen_res[1], color='darkgreen')
ax7.bar(np.arange(len(koeppen_res[2])), koeppen_res[2], color='darkgreen')
ax8.bar(np.arange(len(koeppen_res[3])), koeppen_res[3], color='darkgreen')
ax5.set_ylim([0,0.3])
ax6.set_ylim([0,0.3])
ax7.set_ylim([0,0.3])
ax8.set_ylim([0,3])
ax5.set_xticks(np.arange(len(res)))
ax5.set_xticklabels(koeppen_classes)
ax6.set_xticks(np.arange(len(res)))
ax6.set_xticklabels(koeppen_classes)
ax7.set_xticks(np.arange(len(res)))
ax7.set_xticklabels(koeppen_classes)
ax8.set_xticks(np.arange(len(res)))
ax8.set_xticklabels(koeppen_classes)
ax5.set_ylabel('difference in metric')

plt.savefig('doubling.png')
