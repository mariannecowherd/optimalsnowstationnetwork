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
col = [col_random, col_swaths, col_real]

# load data
testcase = 'new'
corrmap = xr.open_mfdataset(f'corrmap_*_{testcase}.nc') # TODO all models
corrmap = corrmap.mean(dim=('lat','lon')).mrso

# start figure
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# plot individual models
frac = corrmap.frac_observed
for metric, ax in zip(corrmap.metric, (ax1,ax2,ax3,ax4)):
    for modelname in corrmap.model:
        for s, strategy in enumerate(corrmap.strategy):
            ax.plot(frac, corrmap.sel(metric=metric, model=modelname, strategy=strategy),
                    c=col[s], alpha=a, linewidth=0.5)

# plot multi model mean
meancorr = corrmap.mean(dim='model')
meancorr = meancorr.transpose('metric','strategy','frac_observed')
print(meancorr)

ax1.plot(frac, meancorr[0,0,:].values, c=col[0])
ax1.plot(frac, meancorr[0,1,:].values, c=col[1])
ax1.plot(frac, meancorr[0,2,:].values, c=col[2])
ax2.plot(frac, meancorr[1,0,:].values, c=col[0])
ax2.plot(frac, meancorr[1,1,:].values, c=col[1])
ax2.plot(frac, meancorr[1,2,:].values, c=col[2])
ax3.plot(frac, meancorr[2,0,:].values, c=col[0])
ax3.plot(frac, meancorr[2,1,:].values, c=col[1])
ax3.plot(frac, meancorr[2,2,:].values, c=col[2])
ax4.plot(frac, meancorr[3,0,:].values, c=col[0])
ax4.plot(frac, meancorr[3,1,:].values, c=col[1])
ax4.plot(frac, meancorr[3,2,:].values, c=col[2])

ax1.vlines(frac[0], ymin=0.1, ymax=1.1, colors='grey')
ax2.vlines(frac[0], ymin=0.1, ymax=1.1, colors='grey')
ax3.vlines(frac[0], ymin=0.1, ymax=1.1, colors='grey')
ax4.vlines(frac[0], ymin=0, ymax=6, colors='grey')
ax1.set_title('(d) Absolute values')
ax2.set_title('(b) Mean seasonal cycle')
ax3.set_title('(a) Anomalies')
ax4.set_title('(c) Trend')
ax1.set_ylabel('pearson correlation')
ax2.set_ylabel('pearson correlation')
ax3.set_ylabel('pearson correlation')
ax4.set_ylabel('MAE')
ax3.set_xlabel('percentage observed points')
ax4.set_xlabel('percentage observed points')
ax1.text(0.16, 0.98, 'current ISMN')
ax1.grid(alpha=a)
ax2.grid(alpha=a)
ax3.grid(alpha=a)
ax4.grid(alpha=a)

legend_colors = [Line2D([0], [0], marker='None', color=col_random, linewidth=2, label='random'),
                 Line2D([0], [0], marker='None', color=col_swaths, linewidth=2, label='geographical distance'),
                 Line2D([0], [0], marker='None', color=col_real, label='skill-based')]
ax2.legend(handles=legend_colors, loc='lower right', borderaxespad=0.)

plt.show()
#plt.savefig('compare_metrics.png')
