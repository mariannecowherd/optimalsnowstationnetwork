import numpy as np
import cartopy.crs as ccrs
import xarray as xr
import regionmask
import matplotlib.pyplot as plt

largefilepath = '/net/so4/landclim/bverena/large_files/'

# load data
testcase = 'smmask'
#metric = 'trend'
metric = 'corr'
niter = xr.open_mfdataset(f'niter_systematic_*_{metric}_{testcase}.nc', coords='minimal')
corrmaps = xr.open_mfdataset(f'corrmap_systematic_*_{metric}_{testcase}.nc', coords='minimal')
obsmask = xr.open_dataarray(f'{largefilepath}opscaling/obsmask.nc').squeeze()

# calc change in pearson when doubling stations
min_frac = min(corrmaps.mrso.frac_observed)
double_frac = min_frac*2
orig = corrmaps.mrso.sel(frac_observed=min_frac, method='nearest')
double = corrmaps.mrso.sel(frac_observed=double_frac, method='nearest')
corr_increase = double - orig
#corr_increase = orig - double # DEBUG
corr_increase = corr_increase.mean(dim='model').squeeze().load()

# calc rank percentages from iter
niter = niter / niter.max(dim=("lat", "lon")) # removed 1 - ...

# calc model mean
meaniter = niter.mean(dim='model').squeeze().mrso

# select threshold
meaniter = meaniter < double_frac

# ar6 regions
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(obsmask.lon, obsmask.lat)
regions = regionmask.defined_regions.ar6.land.mask(obsmask.lon, obsmask.lat)
regions = regions.where(~np.isnan(landmask))

# area per grid point
res = np.abs(np.diff(regions.lat)[0]) # has to be resolution of "regions" for correct grid area calc
grid = xr.Dataset({'lat': (['lat'], regions.lat.data),
                   'lon': (['lon'], regions.lon.data)})
shape = (len(grid.lat),len(grid.lon))
earth_radius = 6371*1000 # in m
weights = np.cos(np.deg2rad(grid.lat))
area = earth_radius**2 * np.deg2rad(res) * weights * np.deg2rad(res)
area = np.repeat(area.values, shape[1]).reshape(shape)
grid['area'] = (['lat','lon'], area)
grid = grid.to_array() / (1000*1000) # to km**2
grid = grid / (1000*1000) # to Mio km**2
grid = grid.squeeze()

# groupby regions
density_current = obsmask.squeeze().groupby(regions).sum() / grid.groupby(regions).sum()
density_future = (obsmask.squeeze() | meaniter).groupby(regions).sum() / grid.groupby(regions).sum()
corr_increase = corr_increase.groupby(regions).mean()

# round to get rid of 1 slightly below zero number (non significant)
corr_increase = np.round(corr_increase, 3)
density_future[0] = 0 # fringe station that isn't really on greenland
corr_increase[0] = 0 # fringe station that isn't really on greenland

# create world map
density_c = xr.full_like(regions, 0)
for region, d in zip(range(int(regions.max().item())), density_current):
    density_c = density_c.where(regions != region, d) # unit stations per bio square km

density_f = xr.full_like(regions, 0)
for region, d in zip(range(int(regions.max().item())), density_future):
    density_f = density_f.where(regions != region, d) # unit stations per bio square km

doubling = xr.full_like(regions, 0)
for region, d in zip(range(int(regions.max().item())), corr_increase):
    doubling = doubling.where(regions != region, d) # unit stations per bio square km

# calc station density difference
density = density_f - density_c

# set no change to nan
density = density.where(density != 0, np.nan)
doubling = doubling.where(doubling != 0, np.nan)

# set ocean negative number
doubling = doubling.where(~np.isnan(landmask), -10)
density = density.where(~np.isnan(landmask), -10)

# plot difference
fs = 25
cmap = plt.get_cmap('Greens').copy()
bad_color = 'lightgrey'
cmap.set_under('aliceblue')
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(25,7))
ax1 = fig.add_subplot(121, projection=proj)
ax2 = fig.add_subplot(122, projection=proj)
#fig.suptitle('Impact of doubling station number', fontsize=fs)

im = density.plot(ax=ax1, add_colorbar=False, cmap=cmap, 
                                  transform=transf, vmin=0, vmax=12)
regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='red', linewidth=1), 
                                         ax=ax1, add_label=False, projection=transf)
cbar_ax = fig.add_axes([0.48, 0.15, 0.02, 0.7]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=fs)
cbar.set_label('stations per million $km^2$', fontsize=fs)
ax1.set_title('(a) change in station density', fontsize=fs) 

im = doubling.plot(ax=ax2, add_colorbar=False, cmap=cmap, 
                                  transform=transf, vmin=0, vmax=0.3)
                                  #transform=transf, vmin=0, vmax=1.0) # DEBUG
regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='red', linewidth=1), 
                                         ax=ax2, add_label=False, projection=transf)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=fs)
cbar.set_label('pearson correlation', fontsize=fs)
ax2.set_title('(b) change in pearson correlation', fontsize=fs) 

ax1.set_facecolor(bad_color)
ax2.set_facecolor(bad_color)

ax1.coastlines()
ax2.coastlines()
#plt.show()
plt.savefig(f'change_stationdensity_{testcase}.png')
