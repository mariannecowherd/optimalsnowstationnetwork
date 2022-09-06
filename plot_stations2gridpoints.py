"""
plot ismn station locations and cmip6 observed and unobserved grid points

"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

largefilepath = '/net/so4/landclim/bverena/large_files/'

obsmask = xr.open_dataarray(f'{largefilepath}opscaling/obsmask.nc').squeeze()
stations = xr.open_dataset(f'{largefilepath}df_gaps.nc')['mrso']

inactive_networks = ['HOBE','PBO_H20','IMA_CAN1','SWEX_POLAND','CAMPANIA',
                     'HiWATER_EHWSN', 'METEROBS', 'UDC_SMOS', 'KHOREZM',
                     'ICN','AACES','HSC_CEOLMACHEON','MONGOLIA','RUSWET-AGRO',
                     'CHINA','IOWA','RUSWET-VALDAI','RUSWET-GRASS']
stations = stations.where(~stations.network.isin(inactive_networks), drop=True)

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection=proj)
obsmask.plot(ax=ax, cmap='Greys', vmin=0, vmax=2, transform=transf, add_colorbar=False)
ax.scatter(stations.lon, stations.lat, transform=transf, c='black', marker='v', s=2) 
ax.coastlines()
plt.savefig(f'obsmask.pdf', bbox_inches='tight')
