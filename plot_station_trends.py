import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt

# no overall pattern visible in trends, no overall trend visible
# could be due to errors in preprocessing (values different units?)
# could be due to short timespan of data
# could be because no trend is emerging yet

def _calc_slope(x, y):
    '''wrapper that returns the slop from a linear regression fit of x and y'''
    slope = stats.linregress(x, y)[0]  # extract slope only
    return slope

largefilepath = '/net/so4/landclim/bverena/large_files/'

data = xr.open_dataset(f'{largefilepath}df_gaps.nc')
data = data['__xarray_dataarray_variable__']
data = (data - data.mean(dim='time')) / data.std(dim='time')

data.resample(time='1y').mean().mean(dim='stations').plot() # no global trend 

data = data.resample(time='1y').mean()


fig = plt.figure()
ax = fig.add_subplot(111)
trends = np.full(data.shape[1], np.nan)
for s in range(data.stations.shape[0]):
    #data.loc[:,station].plot(ax=ax, c='grey', alpha=0.1) # too busy
    station = data.loc[:,s]
    station = station.where(~np.isnan(station), drop=True)
    time = np.arange(station.shape[0]) 
    time = time[~np.isnan(station)]
    try:
        trends[s] = _calc_slope(time, station)
    except ValueError:
        continue
ax.boxplot(trends[~np.isnan(trends)]) # no overall trend visible
plt.show()

plt.scatter(data.lon, data.lat, c=trends, cmap='coolwarm_r', vmin=-1, vmax=1, s=1)
plt.show()
