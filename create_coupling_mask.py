"""
mask of land-atmosphere coupling (as defined p(P,E) positive) and agpop region

"""

import cftime
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def rename_vars(data):
    varname = list(data.keys())[0]
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({varname: f'{modelname}'})
    try:
        data = data.drop_vars('file_qf')
    except ValueError:
        pass
    try:
        data = data.drop_dims('height')
    except ValueError:
        pass
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap):
        data['time'] = data.indexes['time'].to_datetimeindex()
    if isinstance(data.time[0].item(), cftime._cftime.Datetime360Day):
        data['time'] = data.indexes['time'].to_datetimeindex()
    return data

def open_from_list(filelist):
    data = xr.open_mfdataset(filenames, 
                             combine='nested', # timestamp problem
                             concat_dim=None,  # timestamp problem
                             compat='override', # DANGER
                             preprocess=rename_vars)  # rename vars to model name
    data = data.to_array(dim='model').rename(varname)
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')
    return data

cmip6_path = f'/net/atmos/data/cmip6-ng/mrso/mon/g025/'
filepaths = glob.glob(f'{cmip6_path}mrso_mon_*_ssp370_*r1i1*_g025.nc')
modelnames = [filepath.split('_')[2] for filepath in filepaths]
modelnames.remove('MCM-UA-1-0') # does not have pr
modelnames.remove('GISS-E2-2-G') # does not have et
modelnames = np.unique(modelnames) # double entries

varname = 'tas'
filenames = []
for modelname in modelnames:
    cmip6_path = f'/net/atmos/data/cmip6-ng/{varname}/mon/g025/'
    filenames.append(glob.glob(cmip6_path + f'{varname}_mon_{modelname}_ssp370_r1i1*_g025.nc')[0])
tas = open_from_list(filenames)

varname = 'evspsbl'
filenames = []
for modelname in modelnames:
    cmip6_path = f'/net/atmos/data/cmip6-ng/{varname}/mon/g025/'
    filenames.append(glob.glob(cmip6_path + f'{varname}_mon_{modelname}_ssp370_r1i1*_g025.nc')[0])
et = open_from_list(filenames)

# get landmask
upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'
landmask = xr.open_dataarray(f'{upscalepath}landmask.nc')

# cut out 2014 to 2050
et = et.sel(time=slice('2015','2050'))
tas = tas.sel(time=slice('2015','2050'))

# calc corr and plot
corr = xr.corr(et, tas, dim='time')
corr = corr.mean(dim='model')
corr = corr.where(landmask)
#for modelname in modelnames:
#    corr.sel(model=modelname).plot()
#    plt.show()
corr.plot()
plt.show()

# popcrop
pop = xr.open_dataarray(f'{upscalepath}population_density_regridded.nc')
crop = xr.open_dataarray(f'{upscalepath}cropland_regridded.nc')

import IPython; IPython.embed()
