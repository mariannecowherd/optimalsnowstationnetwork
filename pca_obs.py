"""
PCA on obs data
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
import regionmask
import cartopy.crs as ccrs

largefilepath = '/net/so4/landclim/bverena/large_files/'
df = xr.open_dataset(f'{largefilepath}df_gaps.nc').load()

# delete all-nan stations
df = df['mrso'].dropna('stations', how='all')

# network list
network = df.network.groupby('station_id').first()
lat = df.lat.groupby('station_id').first()
lon = df.lon.groupby('station_id').first()

# mean over all soil layers / take "root zone" sm
df = df.groupby('station_id').first()

# correlation matrix
#corrmatrix = df.to_pandas().corr()
#plt.imshow(corrmatrix, cmap='PiYG', vmin=0, vmax=0.2)
#plt.show()

# corr matrix on climatology
clim = df.groupby('time.month').mean()
corrmatrix = clim.to_pandas().corr()
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(corrmatrix, cmap='PiYG', vmin=0, vmax=1.0)
ticklabels, ticks = np.unique(network.values, return_counts=True)
ticks = ticks.cumsum()
ticks = [0] + list(ticks[:-1])
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels, rotation=90)
plt.show()

# mean corr of each station with each other station
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False)
ax.scatter(lon, lat, c=corrmatrix.mean(axis=0), transform=transf, cmap='PiYG', marker='x')
plt.show()

# normalise
datamean = clim.mean(dim='month')
datastd = clim.std(dim='month')
clim = (clim - datamean) / datastd

datamean = clim.mean(dim='station_id')
datastd = clim.std(dim='station_id')
clim = (clim - datamean) / datastd

# gapfilling (needs to be more sophisticated than this)
# even troyayanska use CLIMFILL analogue before SVDImpute
#df = df.fillna(df.mean(dim='time'))
#df = df.fillna(df.mean()) # stations where never any value observed

# remove stations with missing values
network = network[~np.isnan(clim.values).any(axis=0)]
clim = clim.dropna('station_id', how='any')

# run PCA
pca = PCA(n_components=2)
dims = pca.fit_transform(clim.T)

# plot
plt.scatter(dims[:,0], dims[:,1], c= pd.factorize(network)[0], cmap='tab20')
#for i, txt in enumerate(clim.station_id):
for i, txt in enumerate(network.values):
    plt.annotate(txt.item(), (dims[i,0], dims[i,1]))
plt.show()
