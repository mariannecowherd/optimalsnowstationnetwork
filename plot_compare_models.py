import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

# load data
def drop_time(data): # not necessary after rerun
    filename = data.encoding['source']
    try:
        data = data.drop_vars('time')
        print(filename, 'time succesfully removed')
    except ValueError as e:
        print(filename, e)
    return data
testcase = 'new'
#corrmap = xr.open_mfdataset(f'corrmap_*_A*_{testcase}.nc') # TODO all models
corrmap = xr.open_mfdataset(f'corrmap_*_{testcase}.nc',
                            compat='override',
                            coords='minimal',
                            preprocess=drop_time).load() # not necessary after rerun
corrmap = corrmap.mean(dim=('lat','lon')).mrso
corrmap = corrmap.interpolate_na(dim='frac_observed')
corrmap = corrmap.sel(frac_observed=slice(0,0.9)) # TODO change with higher res
frac = corrmap.frac_observed

metrics = ['r2','seasonality','corr','trend']
strategies = ['systematic','random','interp']
#strategies = ['systematic','random']

# plot constants
colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]
col = [col_real, col_random, col_swaths]
a = 0.5
legend_colors = [Line2D([0], [0], marker='None', color=col_random, linewidth=2, label='random'),
                 Line2D([0], [0], marker='None', color=col_swaths, linewidth=2, label='geographical distance'),
                 Line2D([0], [0], marker='None', color=col_real, label='skill-based')]
fontsize=15

# plot
plt.rcParams.update({'font.size': fontsize})
fig, axes = plt.subplots(nrows=15, ncols=4, figsize=(25,30))
axes[-1,-1].legend(handles=legend_colors, loc='lower center', bbox_to_anchor=(0.1,-1), ncol=3)
#axes[-1,0].axis('off')
#axes[-1,1].axis('off')
#axes[-1,2].axis('off')
#axes[-1,3].axis('off')

axes[0,0].set_title('(d) Monthly mean')
axes[0,1].set_title('(b) Mean seasonal cycle')
axes[0,2].set_title('(a) Monthly Anomalies')
axes[0,3].set_title('(c) Long-term trend')

for m, modelname in enumerate(corrmap.model[15:]):

    for e, metric in enumerate(metrics):
        for s, strategy in enumerate(strategies):
            axes[m,e].plot(frac, corrmap.sel(metric=metric, model=modelname, strategy=strategy),
                           c=col[s], alpha=1, linewidth=0.5)

    axes[m,0].set_ylabel('correlation')
    axes[m,1].set_ylabel('correlation')
    axes[m,2].set_ylabel('correlation')
    axes[m,3].set_ylabel('MAE')
    axes[m,0].grid(alpha=a)
    axes[m,1].grid(alpha=a)
    axes[m,2].grid(alpha=a)
    axes[m,3].grid(alpha=a)

    axes[m,0].text(-0.7, 0.5,modelname.item(),transform=axes[m,0].transAxes, va='center')

plt.savefig(f'compare_models_2.png')
#meancorr = corrmap.mean(dim='model')
