# taken from
# https://github.com/MESMER-group/mesmer/blob/master/mesmer/io/load_constant_files.py

import numpy as np
import pyproj

def calc_geodist_exact(lon, lat):
    """exact great circle distance based on WSG 84
    Parameters
    ----------
    lon : array-like
        1D array of longitudes
    lat : array-like
        1D array of latitudes
    Returns
    -------
    geodist : np.array
        2D array of great circle distances.
    """

    # ensure correct shape
    lon, lat = np.asarray(lon), np.asarray(lat)
    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError("lon and lat need to be 1D arrays of the same shape")

    geod = pyproj.Geod(ellps="WGS84")

    n_points = len(lon)

    geodist = np.zeros([n_points, n_points])

    # calculate only the upper right half of the triangle
    for i in range(n_points):

        # need to duplicate gridpoint (required by geod.inv)
        lt = np.tile(lat[i], n_points - (i + 1))
        ln = np.tile(lon[i], n_points - (i + 1))

        geodist[i, i + 1 :] = geod.inv(ln, lt, lon[i + 1 :], lat[i + 1 :])[2]

    # convert m to km
    geodist /= 1000
    # fill the lower left half of the triangle (in-place)
    geodist += np.transpose(geodist)
    return geodist


