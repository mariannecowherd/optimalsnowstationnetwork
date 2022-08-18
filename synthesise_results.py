niter = xr.open_mfdataset(f'niter_*.nc')

# calc rank percentages from iter
niter = niter / niter.max(dim=("lat", "lon")) # removed 1 - ...

# delete points that are desert in any model
meaniter = niter.mean(dim='model')
meaniter = meaniter.where(~np.isnan(niter).any(dim='model'))
meaniter.to_netcdf('meaniter.nc')
