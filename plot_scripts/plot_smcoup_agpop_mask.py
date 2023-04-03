"""
plot ismn station locations and cmip6 observed and unobserved grid points

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

largefilepath = '/net/so4/landclim/bverena/large_files/'
smmask = xr.open_dataarray(f'{largefilepath}opscaling/landmask.nc').squeeze()
obsmask = xr.open_dataarray(f'{largefilepath}opscaling/obsmask.nc').squeeze()

mask = np.logical_and(smmask, np.logical_not(obsmask))

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection=proj)
mask.plot(ax=ax, cmap='Greys', vmin=0, vmax=2, transform=transf, add_colorbar=False)
ax.coastlines()
ax.set_title('Regions considered for station placement')
plt.savefig(f'smcoupmask.pdf', bbox_inches='tight')
