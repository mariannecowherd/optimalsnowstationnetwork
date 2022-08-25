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

# DEBUG
#testcase = 'new'
#import glob
#filenames = glob.glob(f'corrmap_*_{testcase}.nc')
#for filename in filenames:
#    tmp = xr.open_dataarray(filename)
#    try:
#        if tmp.frac_observed.values[0] != 0.12403100775193798:
#            print(tmp.frac_observed.values[0], filename)
#            filenames.remove(filename)
#    except AttributeError as e:
#        print(filename, e)

# load data
testcase = 'new'
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
metrics = ['r2','seasonality','corr','trend']
strategies = ['systematic','random','interp']
col = [col_real, col_random, col_swaths]

# start figure
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# plot individual models
frac = corrmap.frac_observed
for metric, ax in zip(metrics, (ax1,ax2,ax3,ax4)):
    for modelname in corrmap.model:
        for s, strategy in enumerate(strategies):
            ax.plot(frac, corrmap.sel(metric=metric, model=modelname, strategy=strategy),
                    c=col[s], alpha=a, linewidth=0.5)

# plot multi model mean
meancorr = corrmap.mean(dim='model')
meancorr = meancorr.transpose('metric','strategy','frac_observed')
for metric, ax in zip(metrics, (ax1,ax2,ax3,ax4)):
    for s, strategy in enumerate(strategies):
        ax.plot(frac, meancorr.sel(metric=metric, strategy=strategy),
                    c=col[s])

#ax1.plot(frac, meancorr[0,0,:].values, c=col[0])
#ax1.plot(frac, meancorr[0,1,:].values, c=col[1])
#ax1.plot(frac, meancorr[0,2,:].values, c=col[2])
#ax2.plot(frac, meancorr[1,0,:].values, c=col[0])
#ax2.plot(frac, meancorr[1,1,:].values, c=col[1])
#ax2.plot(frac, meancorr[1,2,:].values, c=col[2])
#ax3.plot(frac, meancorr[2,0,:].values, c=col[0])
#ax3.plot(frac, meancorr[2,1,:].values, c=col[1])
#ax3.plot(frac, meancorr[2,2,:].values, c=col[2])
#ax4.plot(frac, meancorr[3,0,:].values, c=col[0])
#ax4.plot(frac, meancorr[3,1,:].values, c=col[1])
#ax4.plot(frac, meancorr[3,2,:].values, c=col[2])

ax1.vlines(frac[0], ymin=0, ymax=1.1, colors='grey')
ax2.vlines(frac[0], ymin=0, ymax=1.1, colors='grey')
ax3.vlines(frac[0], ymin=0, ymax=1.1, colors='grey')
ax4.vlines(frac[0], ymin=0, ymax=8.1, colors='grey')
ax1.set_title('(d) Monthly mean')
ax2.set_title('(b) Mean seasonal cycle')
ax3.set_title('(a) Monthly Anomalies')
ax4.set_title('(c) Long-term trend')
ax1.set_ylabel('pearson correlation')
ax2.set_ylabel('pearson correlation')
ax3.set_ylabel('pearson correlation')
ax4.set_ylabel('MAE')
ax3.set_xlabel('percentage observed points')
ax4.set_xlabel('percentage observed points')
ax1.text(0.14, 0.92, 'current ISMN')
ax1.grid(alpha=a)
ax2.grid(alpha=a)
ax3.grid(alpha=a)
ax4.grid(alpha=a)
ax1.set_ylim([0,1])
ax2.set_ylim([0,1])
ax3.set_ylim([0,1])
ax4.set_ylim([0,8])

legend_colors = [Line2D([0], [0], marker='None', color=col_random, linewidth=2, label='random'),
                 Line2D([0], [0], marker='None', color=col_swaths, linewidth=2, label='geographical distance'),
                 Line2D([0], [0], marker='None', color=col_real, label='skill-based')]
ax2.legend(handles=legend_colors, loc='lower right', borderaxespad=0.1)

#plt.show()
plt.savefig('compare_metrics.png')
