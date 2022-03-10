import numpy as np
from scipy.spatial.distance import pdist, squareform
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import regionmask
from calc_geodist import calc_geodist_exact as calc_geodist

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

# remove all stations with less than 10 years of meas
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

# sort by country
df = df.sortby('country')

# distance matrix # TODO in km, not latlon
#dist = squareform(pdist(np.array(list([df.lat.values, df.lon.values])).T))
#np.fill_diagonal(dist, np.nan) # inplace
dist = calc_geodist(df.lon.values, df.lat.values)

# corr matrix
corrmatrix = df.to_pandas().corr(method='spearman',min_periods=24).values
np.fill_diagonal(corrmatrix, np.nan) # inplace

# similarity cutoff
dist[np.isnan(corrmatrix)] = np.nan
#dist[corrmatrix < 0.7] = np.nan
allcorr = corrmatrix.copy()
corrmatrix[corrmatrix < 0.7] = np.nan

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()

hull = []
for s in range(df.shape[1]):
    corr = corrmatrix[s, :]
    d = dist[s, :]
    no_stations = d[~np.isnan(corr)].size
    allc = allcorr[s,:]
    print(no_stations)

    idxs = np.where(~np.isnan(corr))[0]
    idxs2 = np.where(~np.isnan(allc))[0]
    #fig = plt.figure(figsize=(10,5))
    #ax = fig.add_subplot(111, projection=proj)
    #ax.set_title(f'number of similar stations: {no_stations}')
    ##ax.set_extent([-180, 180, -90, 90], crs=proj)
    #ax.coastlines()
    #ax.scatter(df.lon, df.lat, c='lightgrey', transform=transf)
    #ax.scatter(df.lon[idxs2], df.lat[idxs2], c='lightblue', transform=transf, edgecolors='black')
    #ax.scatter(df.lon[idxs], df.lat[idxs], c='lightgreen', transform=transf, edgecolors='black')
    #ax.scatter(df.lon[s], df.lat[s], c='red', transform=transf, edgecolors='black')
    #plt.show()
    ##import IPython; IPython.embed()
    ##plt.savefig(f'corrmaps_{s:04}.png')
    ##plt.close()

    # rigid spatial repr formulation
    #if no_stations < 3: # exact formulation
    if no_stations == 0:
        hull.append(0)
    elif no_stations == 1:
        d_corrstations = d[~np.isnan(corr)]
        hull.append(d_corrstations.max())
    else:
        no_close_corr_stations = False
        d_corrstations = d[~np.isnan(corr)]
        d_close = d[d <= d_corrstations.max()]
        frac_closecorr = d_corrstations.size / d_close.size
        while frac_closecorr <= 0.9:
            #print(d_corrstations)
            #print(d_close)
            #print(frac_closecorr)
            # remove furthest distant correlated station
            try:
                d_corrstations = np.delete(d_corrstations, d_corrstations.argmax())
                d_close = d[d <= d_corrstations.max()]
                frac_closecorr = d_corrstations.size / d_close.size
            except ValueError: # closest station is not corr
                no_close_corr_stations = True 
                break # break always only breaks inner loop
        #print(d_corrstations)
        #print(d_close)
        #print(frac_closecorr)
        if no_close_corr_stations:
            hull.append(0)
        else:
            hull.append(d_corrstations.max())

    #continue
    #if no_stations == 0:
    #    hull.append(0)
    #elif no_stations == 1:
    #    tmp = d[~np.isnan(corr)].item()
    #    hull.append(tmp)
    #else:
    #    tmp1 = d[~np.isnan(corr)].max()
    #    hull.append(tmp1)
        # TODO: missing hull correction 90%
        #allc = allcorr[s,:]
        #tmp2 = allc[(d < tmp1) & (~np.isnan(allc))]
        #print('neighbors:', tmp2.size)
        #import IPython; IPython.embed()

        ## plot
        #fig = plt.figure(figsize=(10,5))
        #ax = fig.add_subplot(111)


#fig = plt.figure(figsize=(10,5))
#ax = fig.add_subplot(111, projection=proj)
#ax.coastlines()
#im = ax.scatter(df.lon, df.lat, c='lightgrey', transform=transf)
#hull = np.array(hull)
#lats = df.lat[hull != 0]
#lons = df.lon[hull != 0]
#hull = hull[hull != 0]
#im = ax.scatter(lons, lats, c=hull, transform=transf, cmap='Wistia')
#cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.5]) # left bottom width height
#cbar = fig.colorbar(im, cax=cbar_ax)
#cbar.set_label('distance in km')
#plt.show()

mosaic = '''
AAB
AAC
'''
fig, axs = plt.subplot_mosaic(mosaic, subplot_kw={'projection':proj})
#fig = plt.figure(figsize=(10,5))
#ax = fig.add_subplot(111, projection=proj)
cmap = 'Wistia'
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=axs['A'], add_label=False)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=axs['B'], add_label=False)
regionmask.defined_regions.natural_earth.land_110.plot(line_kws=dict(color='black', linewidth=1), ax=axs['C'], add_label=False)
axs['A'].scatter(df.lon, df.lat, c='lightgrey', transform=transf)
axs['C'].scatter(df.lon, df.lat, c='lightgrey', transform=transf)
im = axs['B'].scatter(df.lon, df.lat, c='lightgrey', transform=transf)
hull = np.array(hull)
lats = df.lat[hull != 0]
lons = df.lon[hull != 0]
hull = hull[hull != 0]
axs['A'].scatter(lons, lats, c=hull, transform=transf, cmap=cmap)
axs['C'].scatter(lons, lats, c=hull, transform=transf, cmap=cmap)
im = axs['B'].scatter(lons, lats, c=hull, transform=transf, cmap=cmap)
axs['B'].set_extent([-120, -60, 25, 52], crs=transf)
axs['C'].set_extent([-10, 30, 35, 72], crs=transf)
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.5]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('distance in km')
plt.show()
