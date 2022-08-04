import numpy as np
import cartopy.crs as ccrs
import xarray as xr
import regionmask
import matplotlib.pyplot as plt

largefilepath = '/net/so4/landclim/bverena/large_files/'

# load data
meaniter = xr.open_dataarray('meaniter.nc')
meaniter = meaniter.sel(metric='_corr') < 0.25
#obsmask = xr.open_mfdataset(f'{largefilepath}opscaling/obsmask_*.nc', combine='nested', compat='override')
obsmask = xr.open_dataarray(f'{largefilepath}opscaling/obsmask.nc') # DEBUG later all models

# ar6 regions
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(obsmask.lon, obsmask.lat)
regions = regionmask.defined_regions.ar6.land.mask(obsmask.lon, obsmask.lat)
regions = regions.where(~np.isnan(landmask))

# area per grid point
res = 2.5 # has to be resolution of "regions" for correct grid area calc
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

# groupby regions
density_current = obsmask.groupby(regions).sum() / grid.groupby(regions).sum()
density_future = (obsmask | meaniter).groupby(regions).sum() / grid.groupby(regions).sum()

# create world map
density_c = xr.full_like(regions, 0)
for region, d in zip(range(int(regions.max().item())), density_current):
    density_c = density_c.where(regions != region, d) # unit stations per bio square km
density_c = density_c.where(~np.isnan(landmask))

density_f = xr.full_like(regions, 0)
for region, d in zip(range(int(regions.max().item())), density_future):
    density_f = density_f.where(regions != region, d) # unit stations per bio square km
density_f = density_f.where(~np.isnan(landmask))

# plot
cmap = plt.get_cmap('Greens').copy()
bad_color = 'lightgrey'
cmap.set_under(bad_color)
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(121, projection=proj)
ax2 = fig.add_subplot(122, projection=proj)

im = density_c.plot(ax=ax1, add_colorbar=False, cmap=cmap, transform=transf, 
             vmin=1, vmax=20)#, cbar_kwargs=cbar_kwargs)
im = density_f.plot(ax=ax2, add_colorbar=False, cmap=cmap, transform=transf, 
             vmin=1, vmax=20)#, cbar_kwargs=cbar_kwargs)
regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), ax=ax1, add_label=False, proj=transf)
regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), ax=ax2, add_label=False, proj=transf)
ax1.set_title('current station density per AR6 region') 
ax2.set_title('future station density per AR6 region') 
ax1.coastlines()
ax2.coastlines()
plt.show()
