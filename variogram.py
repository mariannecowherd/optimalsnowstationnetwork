import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import regionmask

# station data
largefilepath = '/net/so4/landclim/bverena/large_files/'
df = xr.open_dataset(f'{largefilepath}df_gaps.nc').load()
df = df['mrso']

# select the stations still active
inactive_networks = ['HOBE','PBO_H20','IMA_CAN1','SWEX_POLAND','CAMPANIA',
                     'HiWATER_EHWSN', 'METEROBS', 'UDC_SMOS', 'KHOREZM',
                     'ICN','AACES','HSC_CEOLMACHEON','MONGOLIA','RUSWET-AGRO',
                     'CHINA','IOWA','RUSWET-VALDAI','RUSWET-GRASS']
df = df.where(~df.network.isin(inactive_networks), drop=True)

# remove all stations with less than 365 days of meas
count = (~np.isnan(df)).resample(time="1m").sum()
df = df.resample(time='1m').mean()
df = df.where(count >= 16)
df = df.where(np.isnan(df).sum(dim="time") >= 10 * 12)

# normalise
datamean = df.mean(dim='time')
datastd = df.std(dim='time')
df = (df - datamean) / datastd

# calculate anomalies
clim = df.groupby('time.month').mean()
df = df.groupby("time.month") - clim

# select surface layer soil moisture
#lat = df.lat.groupby('station_id').first()
#lon = df.lon.groupby('station_id').first()
#network = df.network.groupby('station_id').first()
#country = df.country.groupby('station_id').first()
#df = df.groupby('station_id').first()
#df = df.assign_coords(network=('station_id',network.data))
#df = df.assign_coords(country=('station_id',country.data))
#df = df.assign_coords(lat=('station_id',lat.data))
#df = df.assign_coords(lon=('station_id',lon.data))
df = df.sortby('country')

# distance matrix # TODO in km, not latlon
from scipy.spatial.distance import pdist, squareform
dist = squareform(pdist(np.array(list([df.lat.values, df.lon.values])).T))
np.fill_diagonal(dist, np.nan) # inplace

# corr matrix
corrmatrix = df.to_pandas().corr(method='spearman',min_periods=12).values
np.fill_diagonal(corrmatrix, np.nan) # inplace

# bin corr and dist by dist
bins = np.arange(0,300,1)
n_obs = []
mean_corr = []
for b in range(len(bins)-1):
    mask = ((dist > bins[b]) & (dist < bins[b+1]))
    n_obs.append(mask.sum() / 2) # bec diagonal
    mean_corr.append(np.nanmean(corrmatrix[mask]))

# plot
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(mean_corr)
ax2.bar(np.arange(len(mean_corr)), n_obs)
ax2.set_xlabel('distance in lat/lon space between two stations')
ax1.set_ylabel('pearson correlation')
ax2.set_ylabel('number of station pairs')
plt.show()

# from visual inspection, set max distance for anomalies
# to correlate to 15 in lat/lon space
threshold = 15
dist[dist > threshold] = np.nan
corrmatrix[corrmatrix > threshold] = np.nan

# plot settings
cmap = plt.get_cmap('viridis').copy()
bad_color = 'lightgrey'
cmap.set_bad(bad_color)
fs = 12
from matplotlib.lines import Line2D
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()

# plot dist
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False)
im = ax.scatter(df.lon, df.lat, c=np.nanmin(dist, axis=0), transform=transf, cmap=cmap, marker='.', s=5, vmin=0, vmax=threshold)
cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.5]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('pearson correlation', fontsize=fs)
legend_colors = [Line2D([0], [0], marker='s', color=bad_color, linewidth=0, markersize=20, label='not enough obs')]
ax.legend(handles=legend_colors, bbox_to_anchor=(0.9, 0.98), loc='center left', borderaxespad=0., frameon=False, fontsize=fs)
plt.show()

# plot corrmatrix
mosaic = '''
AAB
AAC
'''
fig, axs = plt.subplot_mosaic(mosaic, subplot_kw={'projection':proj})
#fig = plt.figure(figsize=(10,5))
#ax = fig.add_subplot(111, projection=proj)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=axs['A'], add_label=False)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=axs['B'], add_label=False)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=axs['C'], add_label=False)
axs['A'].scatter(df.lon, df.lat, c=np.nanmax(corrmatrix, axis=0), transform=transf, cmap=cmap, vmin=0.7, vmax=1, s=5)#, marker='.')
axs['C'].scatter(df.lon, df.lat, c=np.nanmax(corrmatrix, axis=0), transform=transf, cmap=cmap, vmin=0.7, vmax=1, s=5)#, marker='.')
im = axs['B'].scatter(df.lon, df.lat, c=np.nanmax(corrmatrix, axis=0), transform=transf, cmap=cmap, vmin=0.7, vmax=1, s=5)#, marker='.')
axs['B'].set_extent([-120, -60, 25, 52], crs=transf)
axs['C'].set_extent([-10, 30, 35, 72], crs=transf)
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.5]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('pearson correlation', fontsize=fs)
legend_colors = [Line2D([0], [0], marker='s', color=bad_color, linewidth=0, markersize=20, label='not \nenough \nobs')]
axs['B'].legend(handles=legend_colors, bbox_to_anchor=(1.05, 0.9), loc='center left', borderaxespad=0., frameon=False, fontsize=fs)
plt.show()

# convert corrmatrix to distance
corrmatrix = -corrmatrix + 1 # perfect corr is 0, perfect anticorr is 2
corrmatrix[np.isnan(corrmatrix)] = 3 # nan corr is 3

# perform clustering and plot dendrogram
from sklearn.cluster import AgglomerativeClustering
n_clusters = 50
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='single')
clustering = clustering.fit(corrmatrix)
labels = clustering.labels_
print(np.unique(labels, return_counts=True))

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False)
im = ax.scatter(df.lon, df.lat, c=labels, transform=transf, 
                cmap='hsv', marker='.', 
                s=20)
plt.show()

#lat, lon = df.lat, df.lon
#for cluster in range(n_clusters):
#    clat, clon = lat[labels == cluster], lon[labels == cluster]
#    clabel = labels[labels == cluster]
#    fig = plt.figure(figsize=(10,5))
#    ax = fig.add_subplot(111, projection=proj)
#    regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=ax, add_label=False)
#    im = ax.scatter(clon, clat, c=clabel, transform=transf, 
#                    cmap=cmap, marker='.', 
#                    s=10)
#    ax.set_title(f'{cluster}: {len(clabel)} stations')
#    for i, txt in enumerate(clabel):
#        plt.annotate(txt, (clon[i].item(), clat[i].item()), transform=transf)
#    plt.show()
