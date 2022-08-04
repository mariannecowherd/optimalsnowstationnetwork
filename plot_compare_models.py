import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]

modelnames = ['HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR','IPSL-CM6A-LR',
              'ACCESS-ESM1-5','BCC-CSM2-MR','CESM2','CMCC-ESM2',
              'CNRM-ESM2-1','CanESM5','E3SM-1-1','FGOALS-g3',
              'GFDL-ESM4','GISS-E2-1-H','INM-CM4-8','UKESM1-0-LL'] 
metrics = ['r2','seasonality','corr','trend']
strategies = ['random','interp','systematic']
col = [col_random, col_swaths, col_real]
testcase = 'new'

a = 0.5

frac_harmonised = np.arange(0,1.04,0.01)
meancorr = xr.DataArray(np.full((4,3,len(modelnames),len(frac_harmonised)), np.nan), 
            coords={'metric':metrics, 'strategy': strategies, 
                    'model':modelnames,'frac': frac_harmonised})

largefilepath = '/net/so4/landclim/bverena/large_files/'
landmask = xr.open_dataarray(f'{largefilepath}opscaling/landmask_cmip6-ng.nc').squeeze()
added = xr.full_like(landmask.astype(float), np.nan)

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()

for modelname in modelnames:
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245, projection=proj)
    ax6 = fig.add_subplot(246, projection=proj)
    ax7 = fig.add_subplot(247, projection=proj)
    ax8 = fig.add_subplot(248, projection=proj)

    landmask = xr.open_dataarray(f'{largefilepath}opscaling/landmask_{modelname}.nc')
    obsmask = xr.open_dataarray(f'{largefilepath}opscaling/obsmask_{modelname}.nc')

    for metric, axes in zip(metrics, ((ax1,ax5),(ax2,ax6),(ax3,ax7),(ax4,ax8))):
        for s, strategy in enumerate(strategies):

            try:
                with open(f"corr_{strategy}_{modelname}_{metric}_{testcase}.pkl", "rb") as f:
                    corr = pickle.load(f)

                with open(f"nobs_{strategy}_{modelname}_{metric}_{testcase}.pkl", "rb") as f:
                    nobs = pickle.load(f)

                with open(f"lats_systematic_{modelname}_{metric}_{testcase}.pkl", "rb") as f:
                    latlist = pickle.load(f)

                with open(f"lons_systematic_{modelname}_{metric}_{testcase}.pkl", "rb") as f:
                    lonlist = pickle.load(f)

            except FileNotFoundError as e:
                print(e)
                continue
            else:
                print(metric, modelname, strategy)

            # convert number of stations to percentages
            total_no_of_stations = nobs[-1]
            nobs = np.array(nobs)
            frac = nobs / total_no_of_stations

            # plot
            ax_a, ax_b = axes
            ax_a.plot(frac, corr, c=col[s], alpha=a, linewidth=0.5)

            # map of gridpoint added
            if strategy == 'systematic':
                for l, (lats, lons) in enumerate(zip(latlist, lonlist)):
                    for lat, lon in zip(lats, lons):
                        added.loc[lat,lon] = l

                # mark first iteration, land, deserts
                landmask = landmask.astype(float)
                landmask = landmask.where(added != 0, 2)
                landmask.plot(ax=ax_b, cmap='Reds', add_colorbar=False, transform=transf)
                ax_b.coastlines()
                ax_b.set_title('')

            # calculate mean: since different number of steps,
            # first need to interpolate
            corr = xr.DataArray(corr, coords={'frac': frac}).interp(frac=frac_harmonised).values
            meancorr.loc[metric, strategy, modelname,:] = corr
    fig.suptitle(modelname)
    plt.show()

meancorr = meancorr.mean(dim='model')
