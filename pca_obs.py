"""
PCA on obs data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
import regionmask
import cartopy.crs as ccrs

largefilepath = '/net/so4/landclim/bverena/large_files/'
df = xr.open_dataset(f'{largefilepath}df_gaps.nc').load()
df = df['mrso']
print(df.shape)

# remove all stations with less than 365 days of meas
#df = df[:,(~np.isnan(df)).sum(dim="time") >= 365*20]
print(df.shape)
count = (~np.isnan(df)).resample(time="1m").sum()
df = df.resample(time='1m').mean()
df = df.where(count >= 16)
#df = df[:,(~np.isnan(df)).sum(dim="time") >= 10*12]
df = df.where(np.isnan(df).sum(dim="time") >= 10 * 12)

# delete all-nan stations
#df = df.dropna('stations', how='all')
print(df.shape)

# select only still active stations (from dorigo 2021 at least irr. updated)
inactive_networks = ['HOBE','PBO_H20','IMA_CAN1','SWEX_POLAND','CAMPANIA',
                     'HiWATER_EHWSN', 'METEROBS', 'UDC_SMOS', 'KHOREZM',
                     'ICN','AACES','HSC_CEOLMACHEON','MONGOLIA','RUSWET-AGRO',
                     'CHINA','IOWA','RUSWET-VALDAI','RUSWET-GRASS']
df = df.where(~df.network.isin(inactive_networks), drop=True)

# normalise
datamean = df.mean(dim='time')
datastd = df.std(dim='time')
df = (df - datamean) / datastd

# select surface layer soil moisture
lat = df.lat.groupby('station_id').first()
lon = df.lon.groupby('station_id').first()
network = df.network.groupby('station_id').first()
country = df.country.groupby('station_id').first()
df = df.groupby('station_id').first()
df = df.assign_coords(network=('station_id',network.data))
df = df.assign_coords(country=('station_id',country.data))
df = df.sortby('country')

# calculate anomalies
clim = df.groupby('time.month').mean()
df = df.groupby("time.month") - clim

# get driest 10% at each station
#df = (df - df.mean(dim="time")) / df.std(dim="time")
#df = df.groupby('time.dayofyear') - df.groupby('time.dayofyear').mean()
#drought = df < df.quantile(dim="time", q=0.05)
#corrmatrix = drought.to_pandas().corr()
#np.fill_diagonal(corrmatrix.values, np.nan)
#proj = ccrs.Robinson()
#transf = ccrs.PlateCarree()
#fig = plt.figure(figsize=(10,5))
#ax = fig.add_subplot(111, projection=proj)
#regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False)
#ax.scatter(df.lon, df.lat, c=corrmatrix.max(axis=0), transform=transf, cmap='PiYG', marker='x')
#plt.show()

# network list
#country = df.country.groupby('station_id').first()
#
## mean over all soil layers / take "root zone" sm
#df = df.groupby('station_id').first()

# correlation matrix
#corrmatrix = df.to_pandas().corr()
#plt.imshow(corrmatrix, cmap='PiYG', vmin=0, vmax=0.2)
#plt.show()

# corr matrix on climatology
#clim = df.groupby('time.dayofyear').mean()
#corrmatrix = clim.to_pandas().corr()
cmap = plt.get_cmap('viridis').copy()
bad_color = 'lightgrey'
cmap.set_bad(bad_color)
corrmatrix = df.to_pandas().corr()
np.fill_diagonal(corrmatrix.values, np.nan) # inplace
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(corrmatrix, cmap=cmap, vmin=0, vmax=1.0)
ticklabels, ticks = np.unique(df.country.values, return_counts=True)
ticks = ticks.cumsum()
ticks = [0] + list(ticks[:-1])
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels, rotation=90)
ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)
plt.show()

# mean corr of each station with each other station
fs = 12
from matplotlib.lines import Line2D
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False)
im = ax.scatter(lon, lat, c=corrmatrix.max(axis=0), transform=transf, cmap=cmap, marker='.', vmin=0, vmax=1, s=5)
cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.5]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('pearson correlation', fontsize=fs)
legend_colors = [Line2D([0], [0], marker='s', color=bad_color, linewidth=0, markersize=20, label='not enough obs')]
ax.legend(handles=legend_colors, bbox_to_anchor=(0.9, 0.98), loc='center left', borderaxespad=0., frameon=False, fontsize=fs)
plt.show()

# agglomerative clustering
#df = df.dropna('station_id', how='all') # all nan stations gone
#df = df.resample(time="1y").mean() # yearly mean
#df = df.dropna('time', how='all') # all nan years drop
# no years overlapping for all stations!!
lat = lat[~np.isnan(clim.values).any(axis=0)]
lon = lon[~np.isnan(clim.values).any(axis=0)]
clim = clim.dropna("station_id", how="any") # all station with mis vals drop
clim = clim.assign_coords(station_no=("station_id", np.arange(1767)))
clim = clim.swap_dims({"station_id": "station_no"})

from sklearn.cluster import AgglomerativeClustering
n_clusters = 20
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
clustering = clustering.fit(1 - clim.to_pandas().corr())
labels = clustering.labels_

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
clim.plot(ax=ax,cbar_kwargs={"label": "surface layer soil moisture"})
#ax.scatter(np.arange(len(labels)),((labels+1)/n_clusters)*12)
ticklabels, ticks = np.unique(clim.country.values, return_counts=True)
ticks = ticks.cumsum()
ticks = [0] + list(ticks[:-1])
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels, rotation=90)
plt.show()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False)
im = ax.scatter(lon, lat, c=labels, transform=transf, 
                cmap='tab20', marker='.', 
                s=20)
plt.show()

for cluster in range(n_clusters):
    clat, clon = lat[labels == cluster], lon[labels == cluster]
    clabel = labels[labels == cluster]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection=proj)
    regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False)
    im = ax.scatter(clon, clat, c=clabel, transform=transf, 
                    cmap=cmap, marker='.', 
                    s=10)
    for i, txt in enumerate(clabel):
        plt.annotate(txt, (clon[i].item(), clat[i].item()), transform=transf)
    plt.show()

quit()
# normalise
datamean = clim.mean(dim='dayofyear')
datastd = clim.std(dim='dayofyear')
clim = (clim - datamean) / datastd

datamean = clim.mean(dim='station_id')
datastd = clim.std(dim='station_id')
clim = (clim - datamean) / datastd

# gapfilling (needs to be more sophisticated than this)
# even troyayanska use CLIMFILL analogue before SVDImpute
#df = df.fillna(df.mean(dim='time'))
#df = df.fillna(df.mean()) # stations where never any value observed

# plot raw data
clim.plot(cmap='coolwarm_r', vmin=-5, vmax=5)
plt.show()

# remove stations with missing values
network = network[~np.isnan(clim.values).any(axis=0)]
import IPython; IPython.embed()
country = country[~np.isnan(clim.values).any(axis=0)]
lat = lat[~np.isnan(clim.values).any(axis=0)]
lon = lon[~np.isnan(clim.values).any(axis=0)]
clim = clim.dropna('station_id', how='any')
print(clim.shape)

# max corr of each station with each other station
corrmatrix = np.corrcoef(clim.values.T)
np.fill_diagonal(corrmatrix, np.nan)
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False)
ax.scatter(lon, lat, c=np.nanmax(corrmatrix,axis=0), transform=transf, cmap='PiYG', marker='x', vmin=0.35, vmax=1)
plt.show()

# plot raw data
clim.plot(cmap='coolwarm_r', vmin=-5, vmax=5)
plt.show()

# run PCA
pca = PCA(n_components=2)
dims = pca.fit_transform(clim.T)

# plot
plt.scatter(dims[:,0], dims[:,1], c= pd.factorize(country)[0], cmap='tab20')
#for i, txt in enumerate(clim.station_id):
for i, txt in enumerate(country.values):
    plt.annotate(txt.item(), (dims[i,0], dims[i,1]))
plt.show()

# 3dim PCA
pca = PCA(n_components=3)
dims = pca.fit_transform(clim.T)

# plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dims[:,0], dims[:,1], dims[:,2], c= pd.factorize(network)[0], cmap='tab20')
plt.show()
