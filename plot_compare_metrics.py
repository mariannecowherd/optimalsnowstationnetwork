import pickle
import xarray as xr
import xesmf as xe
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import regionmask

upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'

# read files
largefilepath = '/net/so4/landclim/bverena/large_files/'
filename = f'{largefilepath}opscaling/koeppen_simple.nc'
koeppen = xr.open_dataarray(filename)
testcase = 'smmask2'
niter = xr.open_mfdataset(f'niter_systematic*{testcase}.nc', coords='minimal').squeeze().mrso
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(niter.lon, niter.lat)

# calc rank percentages from iter
niter = niter / niter.max(dim=("lat", "lon")) # removed 1 - ...

# calc stats 
meaniter = niter.mean(dim='model')
stditer = niter.std(dim='model')

# set ocean negative such that blue on map
meaniter = meaniter.where(~np.isnan(landmask), -10)
stditer = stditer.where(~np.isnan(landmask), -10)

# plot
fs = 25
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap_r = plt.get_cmap('Reds_r').copy()
cmap = plt.get_cmap('Reds').copy()
bad_color = 'lightgrey'
cmap_r.set_under('aliceblue')
cmap.set_under('aliceblue')

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(221, projection=proj)
ax2 = fig.add_subplot(222, projection=proj)
ax3 = fig.add_subplot(223, projection=proj)
ax4 = fig.add_subplot(224, projection=proj)

levels = np.arange(0.0,1.1,0.1)
stdmin=0.2
stdmax=0.45
levels_std = np.arange(stdmin,stdmax,0.05)
meaniter[0,:,:].plot(ax=ax1, add_colorbar=False, cmap=cmap_r, vmin=0, vmax=1, transform=transf, levels=levels)
stditer[0,:,:].plot(ax=ax2, add_colorbar=False, cmap=cmap, vmin=stdmin, vmax=stdmax, transform=transf, levels=levels_std)
im1 = meaniter[1,:,:].plot(ax=ax3, add_colorbar=False, cmap=cmap_r, vmin=0, vmax=1, transform=transf, levels=levels)
im2 = stditer[1,:,:].plot(ax=ax4, add_colorbar=False, cmap=cmap, vmin=stdmin, vmax=stdmax, transform=transf, levels=levels_std)

ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax4.coastlines()

ax1.set_title('(a)')
ax2.set_title('(b)')
ax3.set_title('(c)')
ax4.set_title('(d)')
ax1.text(-0.3, 0.5,'Interannual \nvariability',transform=ax1.transAxes, va='center')
ax3.text(-0.3, 0.5,'Long-term \ntrend',transform=ax3.transAxes, va='center')

cbar_ax = fig.add_axes([0.13, 0.1, 0.3, 0.02]) # left bottom width height
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Multi-model mean rank percentile')

cbar_ax2 = fig.add_axes([0.59, 0.1, 0.3, 0.02]) # left bottom width height
cbar = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
cbar.set_label('Multi-model standard deviation rank percentile')

ax1.set_facecolor(bad_color)
ax2.set_facecolor(bad_color)
ax3.set_facecolor(bad_color)
ax4.set_facecolor(bad_color)

fig.subplots_adjust(hspace=0.1, bottom=0.15, wspace=0.4)

#plt.show()
plt.savefig('metrics.png', dpi=300)
