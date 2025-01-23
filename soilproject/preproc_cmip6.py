"""
Copyright 2023 ETH Zurich, author: Verena Bessenbacher
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file opens soil moisture, precipiptation and temperature files from each
CMIP6 model individually, selects the relevant land grid points included in the
study (see create_coupling_mask.py). Then it creates a table that stores for
each of these landpoints the recorded soil moisture as well as the
precipitation and temperature from the current and the last 12 months and saves
it as a netcdf.
"""

import glob
import cftime
import numpy as np
import xarray as xr
import xesmf as xe
import regionmask
import warnings

# set filepaths
upscalepath = "/net/so4/landclim/bverena/large_files/opscaling/"

# suppress warnings for non-standard calendars
warnings.filterwarnings(
    "ignore",
    message="Converting a CFTimeIndex with dates "
    + "from a non-standard calendar, '360_day', to a pandas.DatetimeIndex, "
    + "which uses dates from the standard calendar.  This may lead to subtle "
    + "errors in operations that depend on the length of time between dates.",
)
warnings.filterwarnings(
    "ignore",
    message="Converting a CFTimeIndex with dates "
    + "from a non-standard calendar, 'noleap', to a pandas.DatetimeIndex, "
    + "which uses dates from the standard calendar.  This may lead to subtle "
    "errors in operations that depend on the length of time between dates.",
)


# functions for reading CMIP6 files
def preprocess_cmip(data):

    # get varname, modelname, ensemblename and rename variable
    varname = list(data.keys())[0]
    _, _, modelname, _, ensemblename, _ = data.encoding["source"].split("_")
    data = data.rename({varname: f"{modelname}_{ensemblename}"})

    # drop variables file_qf and height, if avail
    try:
        data = data.drop_vars("file_qf")
    except ValueError:
        pass
    try:
        data = data.drop_dims("height")
    except ValueError:
        pass

    # convert non-standard calendars to standard calendar
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap):
        data["time"] = data.indexes["time"].to_datetimeindex()
    if isinstance(data.time[0].item(), cftime._cftime.Datetime360Day):
        data["time"] = data.indexes["time"].to_datetimeindex()

    return data


def open_cmip_suite(modelname, varname):
    cmip6path = f"/net/atmos/data/cmip6-ng/{varname}/mon/g025/"
    possible_filenames = (cmip6path
                         + f"{varname}_mon_{modelname}_ssp370_r1i1*_g025.nc")

    # sometimes 2 ensemble members are chose. select first one
    filename = glob.glob(possible_filenames)[0]

    # open file
    data = xr.open_mfdataset(
        filename,
        combine="nested",  # timestamp problem
        concat_dim=None,  # timestamp problem
        preprocess=preprocess_cmip,
    )
    data = data.to_array(dim="model").rename(varname)

    # convert lon from -180 180 to 0 360
    data.coords["lon"] = (data.coords["lon"] + 180) % 360 - 180
    data = data.sortby("lon")

    return data


# get permafrost mask like https://doi.org/10.5194/tc-14-3155-2020
isfrost = xr.open_dataarray(f'{upscalepath}isfrost.nc')

# get names of all models that can be included
cmip6path = "/net/atmos/data/cmip6-ng/mrso/mon/g025/"
filepaths = glob.glob(f"{cmip6path}mrso_mon_*_ssp370_*r1i1*_g025.nc")
modelnames = [filepath.split("_")[2] for filepath in filepaths]
modelnames.remove("MCM-UA-1-0")  # does not have pr
modelnames = np.unique(modelnames)  # double entries

for modelname in modelnames:
    print(modelname)

    # load data
    mrso = open_cmip_suite(modelname, "mrso")
    tas = open_cmip_suite(modelname, "tas")
    pr = open_cmip_suite(modelname, "pr")

    # cut out 2014 to 2050
    mrso = mrso.sel(time=slice("2015", "2050"))
    tas = tas.sel(time=slice("2015", "2050"))
    pr = pr.sel(time=slice("2015", "2050"))

    # cut out Greenland and Antarctica and ocean for landmask
    n_greenland = (
        regionmask.defined_regions.natural_earth_v5_0_0.countries_110.map_keys(
            "Greenland"
        )
    )
    n_antarctica = (
        regionmask.defined_regions.natural_earth_v5_0_0.countries_110.map_keys(
            "Antarctica"
        )
    )
    mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(mrso)
    mrso = mrso.where(
        (mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask))
    )
    tas = tas.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))
    pr = pr.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))

    # cut out deserts
    largefilepath = "/net/so4/landclim/bverena/large_files/"
    koeppen = (
        xr.open_dataset(f"{largefilepath}opscaling/koeppen_simple.nc")
        .to_array()
        .squeeze()
    )
    koeppen = koeppen.drop("variable")
    regridder = xe.Regridder(koeppen, mrso, "bilinear")
    koeppen = regridder(koeppen)
    isdesert = koeppen == 4
    mrso = mrso.where(~isdesert)
    tas = tas.where(~isdesert)
    pr = pr.where(~isdesert)

    # mask out land ice (as suggested by MH)
    # land ice mask sftgif not avail in cmip6-ng.
    # alternative: mask out gridpoints that are constant in time
    # (and see if that's enough for removing the boreal part)
    # permafrost = xr.open_dataset(f'{largefilepath}opscaling/ESACCI-PERMAFROST-L4-PFR-MODISLST_CRYOGRID-AREA4_PP-2015-fv03.0.nc') # PROBLEM: very weird non-latlon coordinate system
    mrso = mrso.where(~isfrost)
    tas = tas.where(~isfrost)
    pr = pr.where(~isfrost)

    # mask out not-agpop-smcoupl region
    agpop_smcoup_mask = xr.open_dataarray(
        f"{largefilepath}opscaling/smcoup_agpop_mask.nc"
    )
    mrso = mrso.where(agpop_smcoup_mask)
    tas = tas.where(agpop_smcoup_mask)
    pr = pr.where(agpop_smcoup_mask)

    # create lagged features
    tas_1month = tas.copy(deep=True).shift(time=1, fill_value=0).rename("tas_1m")
    tas_2month = tas.copy(deep=True).shift(time=2, fill_value=0).rename("tas_2m")
    tas_3month = tas.copy(deep=True).shift(time=3, fill_value=0).rename("tas_3m")
    tas_4month = tas.copy(deep=True).shift(time=4, fill_value=0).rename("tas_4m")
    tas_5month = tas.copy(deep=True).shift(time=5, fill_value=0).rename("tas_5m")
    tas_6month = tas.copy(deep=True).shift(time=6, fill_value=0).rename("tas_6m")
    tas_7month = tas.copy(deep=True).shift(time=7, fill_value=0).rename("tas_7m")
    tas_8month = tas.copy(deep=True).shift(time=8, fill_value=0).rename("tas_8m")
    tas_9month = tas.copy(deep=True).shift(time=9, fill_value=0).rename("tas_9m")
    tas_10month = tas.copy(deep=True).shift(time=10, fill_value=0).rename("tas_10m")
    tas_11month = tas.copy(deep=True).shift(time=11, fill_value=0).rename("tas_11m")
    tas_12month = tas.copy(deep=True).shift(time=12, fill_value=0).rename("tas_12m")

    pr_1month = pr.copy(deep=True).shift(time=1, fill_value=0).rename("pr_1m")
    pr_2month = pr.copy(deep=True).shift(time=2, fill_value=0).rename("pr_2m")
    pr_3month = pr.copy(deep=True).shift(time=3, fill_value=0).rename("pr_3m")
    pr_4month = pr.copy(deep=True).shift(time=4, fill_value=0).rename("pr_4m")
    pr_5month = pr.copy(deep=True).shift(time=5, fill_value=0).rename("pr_5m")
    pr_6month = pr.copy(deep=True).shift(time=6, fill_value=0).rename("pr_6m")
    pr_7month = pr.copy(deep=True).shift(time=7, fill_value=0).rename("pr_7m")
    pr_8month = pr.copy(deep=True).shift(time=8, fill_value=0).rename("pr_8m")
    pr_9month = pr.copy(deep=True).shift(time=9, fill_value=0).rename("pr_9m")
    pr_10month = pr.copy(deep=True).shift(time=10, fill_value=0).rename("pr_10m")
    pr_11month = pr.copy(deep=True).shift(time=11, fill_value=0).rename("pr_11m")
    pr_12month = pr.copy(deep=True).shift(time=12, fill_value=0).rename("pr_12m")

    # merge predictors into one dataset
    pred = xr.merge(
        [
            tas,
            tas_1month,
            tas_2month,
            tas_3month,
            tas_4month,
            tas_5month,
            tas_6month,
            tas_7month,
            tas_8month,
            tas_9month,
            tas_10month,
            tas_11month,
            tas_12month,
            pr,
            pr_1month,
            pr_2month,
            pr_3month,
            pr_4month,
            pr_5month,
            pr_6month,
            pr_7month,
            pr_8month,
            pr_9month,
            pr_10month,
            pr_11month,
            pr_12month,
        ]
    )

    # save
    mrso = mrso.drop_vars("band")
    mrso = mrso.to_dataset(name="mrso")
    mrso.to_netcdf(f"{upscalepath}mrso_{modelname}.nc")
    pred.to_netcdf(f"{upscalepath}pred_{modelname}.nc")

# create landmask of valid (not ice or desert) land mask
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
    mrso.lon, mrso.lat
)
landmask = landmask.where((mask != n_greenland) & (mask != n_antarctica))
landmask = landmask.where(~isdesert, np.nan)
landmask = landmask.where(~isfrost, np.nan)
landmask = landmask.where(agpop_smcoup_mask, np.nan)
landmask = ~landmask.astype(bool)
landmask = landmask.to_dataset(name="landmask").drop_vars(["band", "height"])
landmask.to_netcdf(f"{upscalepath}landmask.nc")
