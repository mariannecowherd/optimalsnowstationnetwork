import numpy as np
import xarray as xr

def calc_area(data):
    """
    Calc area in Mio km**2 per gridpoint from vector of latitudes and longitudes.
    
    data: xarray.DataArray with coords lat, lon

    returns:
    xarray with dim lats, lons, with area in Mio km**2 per grid point
    """
    
    lats, lons = data.lat.values, data.lon.values

    res = np.abs(np.diff(lats)[0]) # has to be resolution of "regions" for correct grid area calc
    grid = xr.Dataset({'lat': (['lat'], lats),
                       'lon': (['lon'], lons)})
    shape = (len(grid.lat),len(grid.lon))
    earth_radius = 6371*1000 # in m
    weights = np.cos(np.deg2rad(grid.lat))
    area = earth_radius**2 * np.deg2rad(res) * weights * np.deg2rad(res)
    area = np.repeat(area.values, shape[1]).reshape(shape)
    grid['area'] = (['lat','lon'], area)
    grid = grid.to_array() / (1000*1000) # to km**2
    grid = grid / (1000*1000) # to Mio km**2

    return grid
