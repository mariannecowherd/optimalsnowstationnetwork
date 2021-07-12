import xarray as xr

def rename_vars(data):
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({'tas': f'{modelname}_{ensemblename}'})
    data = data.drop_dims('bnds')
    return data

cmip6path = '/net/atmos/data/cmip6-ng/tas/ann/g025/'
ssp = 5
rcp = '85'
cmip = xr.open_mfdataset(f'{cmip6path}tas*_ssp{ssp}{rcp}_*.nc', # shoyer says better use instead of glob
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars,  # rename vars to model name
                         join='left', # timestamp problem
                         compat='override')
cmip = cmip.to_array()
cmip.coords['lon'] = (cmip.coords['lon'] + 180) % 360 - 180
cmip = cmip.sortby('lon')
mmm = cmip.mean(dim='variable').load()
import IPython; IPython.embed()
