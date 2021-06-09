"""
plot koeppen climate classification and classification per station
"""
import xarray as xr

largefilepath = '/net/so4/landclim/bverena/large_files/'
filename = f'{largefilepath}Beck_KG_V1_present_0p5.tif'
koeppen = xr.open_rasterio(filename)
import IPython; IPython.embed()
