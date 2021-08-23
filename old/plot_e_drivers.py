import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

def rename_vars(data):
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({varname: f'{modelname}_{ensemblename}'})
    #data = data.drop_dims('bnds')
    return data

rw_in = '/net/atmos/data/cmip6-ng/rsus/mon/g025/'
precip = '/net/atmos/data/cmip6-ng/pr/mon/g025/'
et = '/net/atmos/data/cmip6-ng/hfls/mon/g025/'
ssp = 5
rcp = '85'
varname = 'rsus'
rsus = xr.open_mfdataset(f'{rw_in}{varname}*_ssp{ssp}{rcp}_r1i1p1f1_*.nc', # shoyer says better use instead of glob
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars,  # rename vars to model name
                         join='left', # timestamp problem
                         compat='override')
varname = 'hfls'
hfls = xr.open_mfdataset(f'{et}{varname}*_ssp{ssp}{rcp}_r1i1p1f1_*.nc', # shoyer says better use instead of glob
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars,  # rename vars to model name
                         join='left', # timestamp problem
                         compat='override')
varname = 'pr'
precip = xr.open_mfdataset(f'{precip}{varname}*_ssp{ssp}{rcp}_r1i1p1f1_*.nc', # shoyer says better use instead of glob
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars,  # rename vars to model name
                         join='left', # timestamp problem
                         compat='override')

# to array, change coords
rsus = rsus.to_array()
rsus.coords['lon'] = (rsus.coords['lon'] + 180) % 360 - 180
rsus = rsus.sortby('lon')

hfls = hfls.to_array()
hfls.coords['lon'] = (hfls.coords['lon'] + 180) % 360 - 180
hfls = hfls.sortby('lon')

precip = precip.to_array()
precip.coords['lon'] = (precip.coords['lon'] + 180) % 360 - 180
precip = precip.sortby('lon')

# calculate corr
decades = np.arange(2015,2090,10)
#r_corr = xr.corr(rsus,hfls, dim='time')
for decade in decades:
    print(decade)
    timeslice = slice(str(decade),str(decade+10))
    p_corr = xr.corr(precip.sel(time=timeslice),hfls.sel(time=timeslice), dim='time')
    p_corr.mean(dim='variable').plot()
    plt.show()
