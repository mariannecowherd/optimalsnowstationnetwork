import xarray as xr

koeppen = xr.open_rasterio(f'{largefilepath}Beck_KG_V1_present_0p5.tif')
koeppen = koeppen.rename({'x':'lon','y':'lat'}).squeeze()
koeppen_simple = xr.full_like(koeppen, np.nan)
k_reduced = [0,1,2,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,13]
kdict = dict(zip(range(31),k_reduced))
for lat in koeppen.lat:
    for lon in koeppen.lon:
        koeppen_simple.loc[lat,lon] = kdict[koeppen.loc[lat,lon].item()]
    print(lat.item())
koeppen_simple.to_netcdf(f'{largefilepath}opscaling/koeppen_simple.nc')
