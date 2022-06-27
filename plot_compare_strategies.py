import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]

#modelnames = ['IPSL-CM6A-LR','HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR']
modelnames = ['HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR','IPSL-CM6A-LR',
              'ACCESS-ESM1-5','BCC-CSM2-MR','CESM2','CMCC-ESM2',
              'CNRM-ESM2-1','CanESM5','E3SM-1-1','FGOALS-g3',
              'GFDL-ESM4','GISS-E2-1-H','INM-CM4-8','UKESM1-0-LL'] 
modelnames = ['HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR','IPSL-CM6A-LR', # three models have weird results TODO include again
              'BCC-CSM2-MR','CESM2','CMCC-ESM2',
              'CNRM-ESM2-1','E3SM-1-1',
              'GFDL-ESM4','GISS-E2-1-H','INM-CM4-8'] 
metrics = ['r2','seasonality','corr','trend']
strategies = ['random','interp','systematic']
col = [col_random, col_swaths, col_real]

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
a = 0.2
#plt.close # DEBUG


frac_harmonised = np.arange(0,1.04,0.01)
meancorr = xr.DataArray(np.full((4,3,len(modelnames),len(frac_harmonised)), np.nan), 
            coords={'metric':metrics, 'strategy': strategies, 
                    'model':modelnames,'frac': frac_harmonised})

for metric, ax in zip(metrics, (ax1,ax2,ax3,ax4)):
    corr_list = []
    for modelname in modelnames:
        for s, strategy in enumerate(strategies):

            try:
                with open(f"corr_{strategy}_{modelname}_{metric}_new.pkl", "rb") as f:
                    corr = pickle.load(f)

                with open(f"nobs_{strategy}_{modelname}_{metric}_new.pkl", "rb") as f:
                    nobs = pickle.load(f)
            except FileNotFoundError:
                continue
            else:
                print(metric, modelname, strategy)

            # convert number of stations to percentages
            total_no_of_stations = nobs[-1]
            nobs = np.array(nobs)
            frac = nobs / total_no_of_stations

            # plot
            ax.plot(frac, corr, c=col[s], alpha=a, linewidth=0.5)

            # calculate mean: since different number of steps,
            # first need to interpolate
            corr = xr.DataArray(corr, coords={'frac': frac}).interp(frac=frac_harmonised).values
            meancorr.loc[metric, strategy, modelname,:] = corr

meancorr = meancorr.mean(dim='model')
ax1.plot(frac_harmonised, meancorr[0,0,:].values, c=col[0])
ax1.plot(frac_harmonised, meancorr[0,1,:].values, c=col[1])
ax1.plot(frac_harmonised, meancorr[0,2,:].values, c=col[2])
ax2.plot(frac_harmonised, meancorr[1,0,:].values, c=col[0])
ax2.plot(frac_harmonised, meancorr[1,1,:].values, c=col[1])
ax2.plot(frac_harmonised, meancorr[1,2,:].values, c=col[2])
ax3.plot(frac_harmonised, meancorr[2,0,:].values, c=col[0])
ax3.plot(frac_harmonised, meancorr[2,1,:].values, c=col[1])
ax3.plot(frac_harmonised, meancorr[2,2,:].values, c=col[2])
ax4.plot(frac_harmonised, meancorr[3,0,:].values, c=col[0])
ax4.plot(frac_harmonised, meancorr[3,1,:].values, c=col[1])
ax4.plot(frac_harmonised, meancorr[3,2,:].values, c=col[2])

ax1.vlines(frac[0], ymin=0.1, ymax=1.1, colors='grey') # TODO correct?
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
#ax1.set_xlabel('percentage observed points')
#ax2.set_xlabel('percentage observed points')
ax3.set_xlabel('percentage observed points')
ax4.set_xlabel('percentage observed points')
#ax1.set_ylim([0.4,1])
#ax2.set_ylim([0.4,1])
#ax3.set_ylim([0,5e-37])
ax1.text(0.16, 0.98, 'current ISMN')
#ax1.set_ylim([0,1.1])
ax1.grid(alpha=a)
ax2.grid(alpha=a)
ax3.grid(alpha=a)
ax4.grid(alpha=a)

legend_colors = [Line2D([0], [0], marker='None', color=col_random, linewidth=2, label='Random'),
                 Line2D([0], [0], marker='None', color=col_swaths, linewidth=2, label='geographical distance'),
                 Line2D([0], [0], marker='None', color=col_real, label='systematic')]
ax2.legend(handles=legend_colors, loc='lower right', borderaxespad=0.)

#plt.show()
plt.savefig('compare_metrics.png')
