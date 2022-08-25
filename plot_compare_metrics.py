import pickle
import xarray as xr
import xesmf as xe
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import regionmask

upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'
metrics = ['r2','seasonality','corr','trend']

# read files
largefilepath = '/net/so4/landclim/bverena/large_files/'
filename = f'{largefilepath}opscaling/koeppen_simple.nc'
koeppen = xr.open_dataarray(filename)
testcase = 'new'
niter = xr.open_mfdataset(f'niter_systematic*{testcase}.nc', coords='minimal').squeeze().mrso

# calc rank percentages from iter
niter = niter / niter.max(dim=("lat", "lon")) # removed 1 - ...

# calc model mean
meaniter = niter.mean(dim='model')

# delete points that are desert in any model # not necessary anymore bec harmonised desert mask
#meaniter = meaniter.where(~np.isnan(niter).any(dim='model'))
#niter = niter.where(~np.isnan(niter).any(dim='model'))

# regrid to koeppen grid for bar plot
regridder = xe.Regridder(niter, koeppen, 'bilinear', reuse_weights=False)
niter = regridder(niter)
koeppen_classes = ['Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df']
koeppen_classes = ['Af','Am','Aw','BS','Cs','Cw','Cf','Ds','Dw','Df'] # BW is removed
koeppen_ints = np.arange(1,12).astype(int)
koeppen_ints = [1,2,3,5,6,7,8,9,10,11]

# percentiles across models:
# groupby_bins with bin int: TypeError: cannot perform reduce with flexible type
#                       arr: ValueError: None of the data falls within bins with edges [0.5, 0.6]
#perc = np.floor(niter*10)*10 # np.floor only rounds to integers
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

colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(421, projection=proj)
ax2 = fig.add_subplot(423, projection=proj)
ax3 = fig.add_subplot(425, projection=proj)
ax4 = fig.add_subplot(427, projection=proj)
ax5 = fig.add_subplot(422)
ax6 = fig.add_subplot(424)
ax7 = fig.add_subplot(426)
ax8 = fig.add_subplot(428)

levels = np.arange(0.0,1.1,0.1)
meaniter[0,:,:].plot(ax=ax1, add_colorbar=False, cmap=cmap, vmin=0, vmax=1, transform=transf, levels=levels)
meaniter[1,:,:].plot(ax=ax2, add_colorbar=False, cmap=cmap, vmin=0, vmax=1, transform=transf, levels=levels)
meaniter[2,:,:].plot(ax=ax3, add_colorbar=False, cmap=cmap, vmin=0, vmax=1, transform=transf, levels=levels)
im = meaniter[3,:,:].plot(ax=ax4, add_colorbar=False, cmap=cmap, vmin=0, vmax=1, transform=transf, levels=levels)

ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax4.coastlines()

ax1.set_title('(a)')
ax2.set_title('(c)')
ax3.set_title('(e)')
ax4.set_title('(g)')
ax5.set_title('(b)')
ax6.set_title('(d)')
ax7.set_title('(f)')
ax8.set_title('(h)')
ax1.text(-0.4, 0.5,'Monthly mean',transform=ax1.transAxes, va='center')
ax2.text(-0.4, 0.5,'Mean \nseasonal cycle',transform=ax2.transAxes, va='center')
ax3.text(-0.4, 0.5,'Monthly \nanomalies',transform=ax3.transAxes, va='center')
ax4.text(-0.4, 0.5,'Long-term trend',transform=ax4.transAxes, va='center')

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

ax5.set_xticks(np.arange(len(koeppen_ints)))
ax5.set_xticklabels(koeppen_classes)
ax6.set_xticks(np.arange(len(koeppen_ints)))
ax6.set_xticklabels(koeppen_classes)
ax7.set_xticks(np.arange(len(koeppen_ints)))
ax7.set_xticklabels(koeppen_classes)
ax8.set_xticks(np.arange(len(koeppen_ints)))
ax8.set_xticklabels(koeppen_classes)
ax5.set_ylabel('mean rank percentile')
ax6.set_ylabel('mean rank percentile')
ax7.set_ylabel('mean rank percentile')
ax8.set_ylabel('mean rank percentile')

fig.subplots_adjust(hspace=0.3)

plt.savefig('metrics_maps.png')
