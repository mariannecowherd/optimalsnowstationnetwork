import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pickle

largefilepath = '/net/so4/landclim/bverena/large_files/'
landmask = xr.open_dataarray(f'{largefilepath}opscaling/landmask_cmip6-ng.nc').squeeze()

modelnames = ['HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR','IPSL-CM6A-LR',
              'ACCESS-ESM1-5','BCC-CSM2-MR','CESM2','CMCC-ESM2',
              'CNRM-ESM2-1','CanESM5','E3SM-1-1','FGOALS-g3',
              'GFDL-ESM4','GISS-E2-1-H','INM-CM4-8','UKESM1-0-LL'] 
method = 'systematic'
testcase = 'new'
metrics = ['r2','seasonality','corr','trend']

added = xr.full_like(landmask.astype(float), np.nan)
added = added.expand_dims(models=modelnames).copy()
for metric in metrics:
    for modelname in modelnames:
        with open(f"lats_systematic_{modelname}_{metric}_{testcase}.pkl", "rb") as f:
            latlist = pickle.load(f)

        with open(f"lons_systematic_{modelname}_{metric}_{testcase}.pkl", "rb") as f:
            lonlist = pickle.load(f)

        for l, (lats, lons) in enumerate(zip(latlist, lonlist)):
            for lat, lon in zip(lats, lons):
                added.loc[modelname,lat,lon] = l

    added.std(dim='models').plot(cmap='jet')
    plt.show()

    added.mean(dim='models').plot(cmap='jet')
    plt.show()
