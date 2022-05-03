"""
trying GP for upscaling, but needs a proper covariance function (= kernel). I will use Lea's dampened distance covariance function from MESMER
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import regionmask
from geographiclib.geodesic import Geodesic

# open example file for lat lon values and landmask
modelname = 'MPI-ESM1-2-HR'
experimentname = 'historical'
ensemblename = 'r1i1p1f1'
cmipfile = f'/net/atmos/data/cmip6-ng/mrso/ann/g025/mrso_ann_{modelname}_{experimentname}_{ensemblename}_g025.nc'
data = xr.open_dataset(cmipfile)

# get landmask
n_greenland = regionmask.defined_regions.natural_earth.countries_110.map_keys('Greenland')
n_antarctica = regionmask.defined_regions.natural_earth.countries_110.map_keys('Antarctica')
mask = regionmask.defined_regions.natural_earth.countries_110.mask(data)
landmask = (mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask))

# get 1d mesgrid arrays of lat and lon of all landpoints
lat, lon = data.lat, data.lon
lon, lat = np.meshgrid(lon, lat)
#lat, lon = lat.flatten(), lon.flatten()
lat, lon = lat[landmask], lon[landmask]

# from MESMER
def calc_geodist_exact(lon, lat):

    g = Geodesic(6378.137, 1 / 298.257223563)

    n_points = len(lon)

    geodist = np.zeros([n_points, n_points])

    # calculate only the upper half of the triangle
    for i in range(n_points):
        lt, ln = lat[i], lon[i]
        for j in range(i + 1, n_points):
            geodist[i, j] = g.Inverse(lt, ln, lat[j], lon[j], Geodesic.DISTANCE)["s12"]

        #if i % 200 == 0:
        print("done with gp", i)

    # fill the lower half of the triangle (in-place)
    geodist += np.transpose(geodist)

    return geodist

def gaspari_cohn(r):
    """
    smooth, exponentially decaying Gaspari-Cohn correlation function
    Parameters
    ----------
    r : np.ndarray
        d/L with d = geographical distance in km, L = localisation radius in km

    Returns
    -------
    y : np.ndarray
        Gaspari-Cohn correlation function value for given r

    Notes
    -----
    - Smooth exponentially decaying correlation function which mimics a Gaussian
      distribution but vanishes at r=2, i.e., 2 x the localisation radius (L)
    - based on Gaspari-Cohn 1999, QJR (as taken from Carrassi et al 2018, Wiley
      Interdiscip. Rev. Clim. Change)
    """
    r = np.abs(r)
    shape = r.shape
    # flatten the array
    r = r.ravel()

    y = np.zeros(r.shape)

    # subset the data
    sel = (r >= 0) & (r < 1)
    r_s = r[sel]
    y[sel] = (
        1 - 5 / 3 * r_s ** 2 + 5 / 8 * r_s ** 3 + 1 / 2 * r_s ** 4 - 1 / 4 * r_s ** 5
    )

    sel = (r >= 1) & (r < 2)
    r_s = r[sel]

    y[sel] = (
        4
        - 5 * r_s
        + 5 / 3 * r_s ** 2
        + 5 / 8 * r_s ** 3
        - 1 / 2 * r_s ** 4
        + 1 / 12 * r_s ** 5
        - 2 / (3 * r_s)
    )

    return y.reshape(shape)

geodist = calc_geodist_exact(lon, lat)
phi_gc = gaspari_cohn(geodist  / 1500)
