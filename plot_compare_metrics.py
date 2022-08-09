import pickle
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import regionmask

colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]

upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'
modelnames = ['HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR','IPSL-CM6A-LR',
              'ACCESS-ESM1-5','BCC-CSM2-MR','CESM2','CMCC-ESM2',
              'CNRM-ESM2-1','CanESM5','E3SM-1-1','FGOALS-g3',
              'GFDL-ESM4','GISS-E2-1-H','INM-CM4-8','UKESM1-0-LL'] 
#modelnames = ['ACCESS-CM2 ','ACCESS-ESM1-5 ','BCC-CSM2-MR ','CESM2-WACCM ',
#              'CESM2 ','CMCC-CM2-SR5 ','CMCC-ESM2 ','CNRM-CM6-1-HR ',
#              'CNRM-CM6-1 ','CNRM-ESM2-1 ','CanESM5-CanOE ','CanESM5 E3SM-1-1 ',
#              'EC-Earth3-AerChem ','EC-Earth3-Veg-LR ','EC-EARTH3-Veg ',
#              'FGOALS-f3-L ','FGOALS-g3 ','GFDL-ESM4 ','GISS-E2-1-G ',
#              'GISS-E2-1-H ','GISS-E2-2-G ','HadGEM3-GC31-MM ','INM-CM4-8 ',
#              'INM-CM5-0 ','IPSL-CM6A-LR ','MIROC-ES2L ','MIROC6 ',
#              'MPI-ESM1-2-HR ','MPI-ESM1-2-LR ','MRI-ESM2-0 ','UKESM1-0-LL']
metrics = ['_r2','_seasonality','_corr','_trend']

largefilepath = '/net/so4/landclim/bverena/large_files/'
filename = f'{largefilepath}opscaling/koeppen_simple.nc'
koeppen = xr.open_dataarray(filename)

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
meaniter.to_netcdf('meaniter.nc')

# regrid to koeppen grid for bar plot
import xesmf as xe
regridder = xe.Regridder(niter, koeppen, 'bilinear', reuse_weights=False)
niter = regridder(niter)
koeppen_classes = ['Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df']
koeppen_classes = ['Af','Am','Aw','BS','Cs','Cw','Cf','Ds','Dw','Df'] # BW is removed
koeppen_ints = np.arange(1,12).astype(int)
koeppen_ints = [1,2,3,5,6,7,8,9,10,11]

koeppen_res = []
for metric in metrics:
    res = []
    for i in koeppen_ints:
        res.append(niter.sel(metric=metric).where(koeppen == i).mean().item())
    koeppen_res.append(res)

# percentiles across models:
# groupby_bins with bin int: TypeError: cannot perform reduce with flexible type
#                       arr: ValueError: None of the data falls within bins with edges [0.5, 0.6]
perc = np.floor(niter*10)*10 # np.floor only rounds to integers
koeppen_res = []
koeppen_res = np.zeros((4,10,10))
for m, metric in enumerate(metrics):
    for k, i in enumerate(koeppen_ints):
        tmp = niter.sel(metric=metric).where(koeppen == i).stack(land=('lat','lon')).dropna('land', how='all')
        tmp = tmp.values.flatten()
        tmp = np.histogram(tmp, bins=np.arange(0,1.1,0.1))[0]
        koeppen_res[m,k,:] = tmp / tmp.sum()
koeppen_res[0,:,:] = koeppen_res[0,:,:] / koeppen_res[0,:,:].sum(axis=0)
koeppen_res[1,:,:] = koeppen_res[1,:,:] / koeppen_res[1,:,:].sum(axis=0)
koeppen_res[2,:,:] = koeppen_res[2,:,:] / koeppen_res[2,:,:].sum(axis=0)
koeppen_res[3,:,:] = koeppen_res[3,:,:] / koeppen_res[3,:,:].sum(axis=0)

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('Reds_r')

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(421, projection=proj)
ax2 = fig.add_subplot(423, projection=proj)
ax3 = fig.add_subplot(425, projection=proj)
ax4 = fig.add_subplot(427, projection=proj)
ax5 = fig.add_subplot(422)
ax6 = fig.add_subplot(424)
ax7 = fig.add_subplot(426)
ax8 = fig.add_subplot(428)

meaniter[0,:,:].plot.contourf(ax=ax1, add_colorbar=False, cmap=cmap, vmin=0, vmax=1, transform=transf)
meaniter[1,:,:].plot.contourf(ax=ax2, add_colorbar=False, cmap=cmap, vmin=0, vmax=1, transform=transf)
meaniter[2,:,:].plot.contourf(ax=ax3, add_colorbar=False, cmap=cmap, vmin=0, vmax=1, transform=transf)
im = meaniter[3,:,:].plot.contourf(ax=ax4, add_colorbar=False, cmap=cmap, vmin=0, vmax=1, transform=transf)

ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax4.coastlines()

ax1.set_title('(a) Mean monthly values')
ax2.set_title('(b) Mean seasonal cycle')
ax3.set_title('(c) Monthly anomalies')
ax4.set_title('(d) Long-term trend')

cbar_ax = fig.add_axes([0.15, 0.07, 0.3, 0.02]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('mean rank percentile')

for i in range(10):
    ax5.bar(np.arange(len(koeppen_ints)), koeppen_res[0,i,:], bottom=koeppen_res[0,:i,:].sum(axis=0), color=cmap(i/10))
    ax6.bar(np.arange(len(koeppen_ints)), koeppen_res[1,i,:], bottom=koeppen_res[1,:i,:].sum(axis=0), color=cmap(i/10))  
    ax7.bar(np.arange(len(koeppen_ints)), koeppen_res[2,i,:], bottom=koeppen_res[2,:i,:].sum(axis=0), color=cmap(i/10))
    ax8.bar(np.arange(len(koeppen_ints)), koeppen_res[3,i,:], bottom=koeppen_res[3,:i,:].sum(axis=0), color=cmap(i/10))

#ax5.set_ylim([0,0.98])
#ax6.set_ylim([0,0.98])
#ax7.set_ylim([0,0.98])
#ax8.set_ylim([0,0.98])

ax5.set_xticks(np.arange(len(res)))
ax5.set_xticklabels(koeppen_classes)
ax6.set_xticks(np.arange(len(res)))
ax6.set_xticklabels(koeppen_classes)
ax7.set_xticks(np.arange(len(res)))
ax7.set_xticklabels(koeppen_classes)
ax8.set_xticks(np.arange(len(res)))
ax8.set_xticklabels(koeppen_classes)
ax5.set_ylabel('mean rank percentile')
ax6.set_ylabel('mean rank percentile')
ax7.set_ylabel('mean rank percentile')
ax8.set_ylabel('mean rank percentile')

plt.savefig('metrics_maps.png')
