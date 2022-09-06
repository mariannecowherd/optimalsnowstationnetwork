"""
plot ismn station locations and cmip6 observed and unobserved grid points

"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

largefilepath = '/net/so4/landclim/bverena/large_files/'
smmask = xr.open_dataarray(f'{largefilepath}opscaling/smcoup_agpop_mask.nc').squeeze()

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection=proj)
smmask.plot(ax=ax, cmap='Greys', vmin=0, vmax=2, transform=transf, add_colorbar=False)
ax.coastlines()
ax.set_title('Regions relevant for soil moisture observations')
plt.savefig(f'smcoupmask.pdf', bbox_inches='tight')
