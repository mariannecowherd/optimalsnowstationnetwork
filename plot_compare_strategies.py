import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# set colors
colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]
a = 0.2

def drop_time(data): # not necessary after rerun
    filename = data.encoding['source']
    try:
        data = data.drop_vars('time')
        print(filename, 'time succesfully removed')
    except ValueError as e:
        print(filename, e)
    return data

# load data
testcase = 'smmask2'
corrmap = xr.open_mfdataset(f'corrmap_*_{testcase}.nc',
                            compat='override',
                            coords='minimal',
                            preprocess=drop_time).load() # not necessary after rerun
#corrmap = xr.open_mfdataset(filenames, preprocess=drop_time, coords='minimal', compat='override')
corrmap = corrmap.mean(dim=('lat','lon')).mrso

# interpolate to regular frac_observed intervals
#frac_observed = np.arange(0.12,1.0,0.05)
#corrmap = corrmap.interp(frac_observed=frac_observed)
corrmap = corrmap.interpolate_na(dim='frac_observed')
corrmap = corrmap.sel(frac_observed=slice(0,0.9)) # TODO change with higher res

# defining those is necessary because order from reading files is not predictable
metrics = ['corr','trend']
strategies = ['systematic','random','interp']
col = [col_real, col_random, col_swaths]

# start figure
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# plot individual models
frac = corrmap.frac_observed
for metric, ax in zip(metrics, (ax1,ax2)):
    for modelname in corrmap.model:
        for s, strategy in enumerate(strategies):
            ax.plot(frac, corrmap.sel(metric=metric, model=modelname, strategy=strategy),
                    c=col[s], alpha=a, linewidth=0.5)

# plot multi model mean
meancorr = corrmap.mean(dim='model')
meancorr = meancorr.transpose('metric','strategy','frac_observed')
for metric, ax in zip(metrics, (ax1,ax2)):
    for s, strategy in enumerate(strategies):
        ax.plot(frac, meancorr.sel(metric=metric, strategy=strategy),
                c=col[s])

ax1.vlines(frac[0], ymin=0, ymax=1.1, colors='grey')
ax2.vlines(frac[0], ymin=0, ymax=8, colors='grey')
ax1.set_title('(a) Inter-annual variability')
ax2.set_title('(b) Trend')
ax1.set_ylabel('pearson correlation')
ax2.set_ylabel('MAE [$kg\;m^{-2}$]')
ax1.set_xlabel('percentage observed points')
ax2.set_xlabel('percentage observed points')
ax1.text(0.19, 0.92, 'current ISMN')
ax1.grid(alpha=a)
ax2.grid(alpha=a)
ax1.set_ylim([0,1])
ax2.set_ylim([0,4])

legend_colors = [Line2D([0], [0], marker='None', color=col_random, linewidth=2, label='random'),
                 Line2D([0], [0], marker='None', color=col_swaths, linewidth=2, label='geographical distance'),
                 Line2D([0], [0], marker='None', color=col_real, label='skill-based')]
ax2.legend(handles=legend_colors, loc='upper right', borderaxespad=0.1)

#plt.show()
plt.savefig('strategies.pdf')
