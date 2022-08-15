import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

# load data
testcase = 'new'
corrmap = xr.open_mfdataset(f'corrmap_*_A*_{testcase}.nc') # TODO all models
corrmap = corrmap.mean(dim=('lat','lon')).mrso
frac = corrmap.frac_observed

metrics = ['r2','seasonality','corr','trend']
strategies = ['systematic','random','interp']
strategies = ['systematic','random']

# plot constants
colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]
col = [col_random, col_swaths, col_real]
a = 0.5
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()

for modelname in corrmap.model:
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    for metric, ax in zip(metrics, (ax1,ax2,ax3,ax4,)):
        for s, strategy in enumerate(strategies):

            ax.plot(frac, corrmap.sel(metric=metric, model=modelname, strategy=strategy),
                    c=col[s], alpha=a, linewidth=0.5)

    fig.suptitle(modelname.item())
    plt.show()

meancorr = corrmap.mean(dim='model')
