import pickle
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]

upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'
#modelnames = ['IPSL-CM6A-LR','HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR']
modelnames = ['HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR','IPSL-CM6A-LR',
              'ACCESS-ESM1-5','BCC-CSM2-MR','CESM2','CMCC-ESM2',
              'CNRM-ESM2-1','CanESM5','E3SM-1-1','FGOALS-g3',
              'GFDL-ESM4','GISS-E2-1-H','INM-CM4-8','UKESM1-0-LL'] 
metrics = ['_r2','_seasonality','_corr','_trend']

largefilepath = '/net/so4/landclim/bverena/large_files/'
#filename = f'{largefilepath}Beck_KG_V1_present_0p5.tif'
filename = f'{largefilepath}koeppen_simple.nc'
koeppen = xr.open_dataarray(filename)
#koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()

mrso = xr.open_dataset(f'{upscalepath}mrso_{modelnames[0]}.nc')['mrso'].load().squeeze()
mrso = mrso.reset_coords("model", drop=True)
niter = xr.full_like(mrso.mean(dim='time'), np.nan)
niter = niter.expand_dims({'model':len(modelnames)}).copy()
niter = niter.assign_coords({"model": modelnames})
niter = niter.expand_dims({'metric':len(metrics)}).copy()
niter = niter.assign_coords({"metric": metrics})


koeppen_res = []
for metric in metrics:

    for modelname in modelnames:
        try:
            with open(f"corr_systematic_{modelname}{metric}_new.pkl", "rb") as f:
                corr = pickle.load(f)

            with open(f"nobs_systematic_{modelname}{metric}_new.pkl", "rb") as f:
                nobs = pickle.load(f)

            with open(f"lats_systematic_{modelname}{metric}_new.pkl", "rb") as f:
                latlist = pickle.load(f)

            with open(f"lons_systematic_{modelname}{metric}_new.pkl", "rb") as f:
                lonlist = pickle.load(f)

        except FileNotFoundError:
            continue
        else:
            print(metric, modelname)

        for i, (lats, lons) in enumerate(zip(latlist, lonlist)):
            for lat, lon in zip(lats, lons):
                niter.loc[metric,modelname,lat,lon] = i

# calc rank percentages from iter
niter = niter / niter.max(dim=("lat", "lon")) # removed 1 - ...

# delete points that are desert in any model
meaniter = niter.mean(dim='model')
meaniter = meaniter.where(~np.isnan(niter).any(dim='model'))

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(241, projection=proj)
ax2 = fig.add_subplot(242, projection=proj)
ax3 = fig.add_subplot(243, projection=proj)
ax4 = fig.add_subplot(244, projection=proj)
ax5 = fig.add_subplot(245)
ax6 = fig.add_subplot(246)
ax7 = fig.add_subplot(247)
ax8 = fig.add_subplot(248)

meaniter[0,:,:].plot.contourf(ax=ax1, add_colorbar=False, cmap='Reds_r', vmin=0, vmax=1, transform=transf)
meaniter[1,:,:].plot.contourf(ax=ax2, add_colorbar=False, cmap='Reds_r', vmin=0, vmax=1, transform=transf)
meaniter[2,:,:].plot.contourf(ax=ax3, add_colorbar=False, cmap='Reds_r', vmin=0, vmax=1, transform=transf)
im = meaniter[3,:,:].plot.contourf(ax=ax4, add_colorbar=False, cmap='Reds_r', vmin=0, vmax=1, transform=transf)

ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax4.coastlines()

ax1.set_title('(a) Absolute values')
ax2.set_title('(b) Mean seasonal cycle')
ax3.set_title('(c) Anomalies')
ax4.set_title('(d) Trend')

cbar_ax = fig.add_axes([0.91, 0.53, 0.02, 0.3]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('mean rank percentile')


import xesmf as xe
regridder = xe.Regridder(niter, koeppen, 'bilinear', reuse_weights=False)
niter = regridder(niter)
koeppen_classes = ['Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df']
koeppen_ints = np.arange(1,12).astype(int)

#fig = plt.figure(figsize=(10, 10))
#ax1 = fig.add_subplot(221)
#ax2 = fig.add_subplot(222)
#ax3 = fig.add_subplot(223)
#ax4 = fig.add_subplot(224)

koeppen_res = []
for metric in metrics:
    res = []
    for i in koeppen_ints:
        res.append(niter.sel(metric=metric).where(koeppen == i).mean().item())
        #tmp = niter.sel(metric=metric).where(koeppen == i)
        #tmp.mean(dim=('lat','lon')).plot()
    koeppen_res.append(res)

ax5.bar(np.arange(len(koeppen_res[0])), koeppen_res[0], color='darkred')
ax6.bar(np.arange(len(koeppen_res[1])), koeppen_res[1], color='darkred')
ax7.bar(np.arange(len(koeppen_res[2])), koeppen_res[2], color='darkred')
ax8.bar(np.arange(len(koeppen_res[3])), koeppen_res[3], color='darkred')

ax5.set_ylim([0,0.8])
ax6.set_ylim([0,0.8])
ax7.set_ylim([0,0.8])
ax8.set_ylim([0,0.8])

ax5.set_xticks(np.arange(len(res)))
ax5.set_xticklabels(koeppen_classes)
ax6.set_xticks(np.arange(len(res)))
ax6.set_xticklabels(koeppen_classes)
ax7.set_xticks(np.arange(len(res)))
ax7.set_xticklabels(koeppen_classes)
ax8.set_xticks(np.arange(len(res)))
ax8.set_xticklabels(koeppen_classes)
#ax5.set_title('(d) Absolute values')
#ax6.set_title('(b) Mean seasonal cycle')
#ax7.set_title('(a) Anomalies')
#ax8.set_title('(c) Trend')
ax5.set_ylabel('mean rank percentile')
plt.savefig('metrics_maps.png')
