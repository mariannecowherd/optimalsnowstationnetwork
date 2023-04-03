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

This file takes the preprocessed ISMN stations (see preproc_ismn.py) and CMIP6
data (see preproc_cmip6.py) and runs the algorithm described in the paper to
add virtual stations according to the strategies=methods (random, systematic,
interp), metrics (corr, trend) and for each CMIP6 model.
"""

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
import argparse
from calc_geodist import calc_geodist_exact as calc_geodist
import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", 
    level=logging.DEBUG
)

# define options
parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", dest="method", type=str)  # systematic, random, interp
parser.add_argument("--metric", "-p", dest="metric", type=str)  # corr, trend
parser.add_argument("--model", "-c", dest="model", type=str)  # eg MIROC6
args = parser.parse_args()

method = args.method
metric = args.metric
modelname = args.model

# define constants
largefilepath = "/net/so4/landclim/bverena/large_files/"
upscalepath = "/net/so4/landclim/bverena/large_files/opscaling/"
logging.info(f"method {method} metric {metric} modelname {modelname}...")

# read CMIP6 files
logging.info("read cmip ...")
mrso = xr.open_dataset(f"{upscalepath}mrso_{modelname}.nc")["mrso"].load().squeeze()
pred = xr.open_dataset(f"{upscalepath}pred_{modelname}.nc").load().squeeze()
landmask = xr.open_dataset(f"{upscalepath}landmask.nc").to_array().squeeze()
landmask = landmask.drop_vars("variable")

# calc standardised anomalies
# tree-based methods do not need standardisation see https://datascience.stack
# exchange.com/questions/5277/do-you-have-to-normalize-data-when-building-decis
# ion-trees-using-r
# therefore these are only used for metric computation
mrso_mean = mrso.groupby("time.month").mean()
mrso_std = mrso.groupby("time.month").std()

pred_mean = pred.groupby("time.month").mean()
pred_std = pred.groupby("time.month").std()

logging.info(f"metric {metric} data shape {mrso.shape}")

# read station data
logging.info("read station data ...")
stations = xr.open_dataset(f"{largefilepath}df_gaps.nc")["mrso"]

# select the stations still active
inactive_networks = [
    "HOBE",
    "PBO_H20",
    "IMA_CAN1",
    "SWEX_POLAND",
    "CAMPANIA",
    "HiWATER_EHWSN",
    "METEROBS",
    "UDC_SMOS",
    "KHOREZM",
    "ICN",
    "AACES",
    "HSC_CEOLMACHEON",
    "MONGOLIA",
    "RUSWET-AGRO",
    "CHINA",
    "IOWA",
    "RUSWET-VALDAI",
    "RUSWET-GRASS",
]
stations = stations.where(~stations.network.isin(inactive_networks), drop=True)

# create iter objects
latlist = stations.lat_cmip.values.tolist()  # only needed for first iteration
lonlist = stations.lon_cmip.values.tolist()
niter = xr.full_like(landmask.astype(float), np.nan)
n = 100
i = 0
testcase = "_smmask2"
corrmaps = []

logging.info("start loop ...")
while True:
    # create boolean obsmask and unobsmask
    obsmask = xr.full_like(landmask, False)
    unobsmask = landmask.copy(deep=True)
    for lat, lon in zip(latlist, lonlist):
        unobsmask.loc[lat, lon] = False
        if landmask.loc[
            lat, lon
        ].item():  # obs gridpoint if station contained and on CMIP land
            obsmask.loc[lat, lon] = True

    if i == 0: 

        # save obsmask before adding first new virtual stations
        obsmask.to_netcdf(
            f"{upscalepath}obsmask.nc"
        )  # is the same for all models, metrics, strategies
        

    # divide into obs and unobs gridpoints
    obslat, obslon = np.where(obsmask)
    obslat, obslon = xr.DataArray(obslat, dims="landpoints"), xr.DataArray(
        obslon, dims="landpoints"
    )

    mrso_obs = mrso.isel(lat=obslat, lon=obslon)
    pred_obs = pred.isel(lat=obslat, lon=obslon)
    mrso_mean_obs = mrso_mean.isel(lat=obslat, lon=obslon)
    mrso_std_obs = mrso_std.isel(lat=obslat, lon=obslon)

    unobslat, unobslon = np.where(unobsmask)
    unobslat, unobslon = xr.DataArray(unobslat, dims="landpoints"), xr.DataArray(
        unobslon, dims="landpoints"
    )

    mrso_unobs = mrso.isel(lat=unobslat, lon=unobslon)
    pred_unobs = pred.isel(lat=unobslat, lon=unobslon)
    mrso_mean_unobs = mrso_mean.isel(lat=unobslat, lon=unobslon)
    mrso_std_unobs = mrso_std.isel(lat=unobslat, lon=unobslon)

    month_mask = xr.open_dataarray(f"{upscalepath}monthmask.nc")
    mask_obs = month_mask.isel(lat=obslat, lon=obslon)
    mask_unobs = month_mask.isel(lat=unobslat, lon=unobslon)

    logging.info(
        f"{mrso_obs.shape[1]} gridpoints observed, {mrso_unobs.shape[1]} gridpoints unobserved"
    )

    if i == 0: 

        # get list of lat, lon for each existing station
        latlist = mrso_obs.lat.values.tolist()
        lonlist = mrso_obs.lon.values.tolist()

    # stack landpoints and time
    y_train = mrso_obs.stack(datapoints=("landpoints", "time"))
    y_test = mrso_unobs.stack(datapoints=("landpoints", "time"))

    X_train = pred_obs.stack(datapoints=("landpoints", "time")).to_array().T
    X_test = pred_unobs.stack(datapoints=("landpoints", "time")).to_array().T

    if y_test.size == 0: # check if unobserved points are still avail
        logging.info("all points are observed. stop process...")
        break

    if mrso_unobs.shape[1] < n:
        n = mrso_unobs.shape[1]  # rest of points
        logging.info(f"last iteration with {n} points ...")

    # train and predict with random forest
    kwargs = {
        "n_estimators": 100,
        "min_samples_leaf": 1,  # those are all default values anyways
        "max_features": "sqrt",
        "max_samples": None,
        "bootstrap": True,
        "warm_start": False,
        "n_jobs": 40,  # set to number of trees
        "verbose": 0,
    }

    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train, y_train)

    y_predict = xr.full_like(y_test, np.nan)
    y_predict[:] = rf.predict(X_test)

    y_train_predict = xr.full_like(y_train, np.nan)
    y_train_predict[:] = rf.predict(X_train)

    # unstack
    y_predict = y_predict.unstack("datapoints").load()
    y_test = y_test.unstack("datapoints").load()
    y_train = y_train.unstack("datapoints").load()
    y_train_predict = y_train_predict.unstack("datapoints").load()
    y_latlon = y_test.copy(deep=True)

    # calculate upscale performance
    corrmap = xr.full_like(landmask.astype(float), np.nan)
    if metric == "corr":
        # calculate anomalies
        y_test = y_test.groupby("time.month") - mrso_mean_unobs
        y_predict = y_predict.groupby("time.month") - mrso_mean_unobs
        y_train = y_train.groupby("time.month") - mrso_mean_obs
        y_train_predict = y_train_predict.groupby("time.month") - mrso_mean_obs

        # select 3 driest consecutive months
        for year in np.unique(y_test.coords["time.year"]):
            y_train.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ] = y_train.loc[dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))].where(
                mask_obs.T.values
            )
            y_train_predict.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ] = y_train_predict.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ].where(
                mask_obs.T.values
            )
            y_test.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ] = y_test.loc[dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))].where(
                mask_unobs.T.values
            )
            y_predict.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ] = y_predict.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ].where(
                mask_unobs.T.values
            )

        # resample yearly
        y_test = y_test.resample(time="1y").mean()
        y_predict = y_predict.resample(time="1y").mean()
        y_train = y_train.resample(time="1y").mean()
        y_train_predict = y_train_predict.resample(time="1y").mean()

        # calc corr
        corr = xr.corr(y_test, y_predict, dim="time")
        corr_train = xr.corr(y_train, y_train_predict, dim="time")
    elif metric == "seasonality":
        y_test = y_test.groupby("time.month").mean().rename(month="time")
        y_predict = y_predict.groupby("time.month").mean().rename(month="time")
        y_train = y_train.groupby("time.month").mean().rename(month="time")
        y_train_predict = (
            y_train_predict.groupby("time.month").mean().rename(month="time")
        )

        corr = xr.corr(y_test, y_predict, dim="time")
        corr_train = xr.corr(y_train, y_train_predict, dim="time")
    elif (
        metric == "trend"
    ):  # until like in Cook 2020: anomalies with reference to baseline period (1851-1880)
        # select 3 driest consecutive months
        for year in np.unique(y_test.coords["time.year"]):
            y_train.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ] = y_train.loc[dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))].where(
                mask_obs.T.values
            )
            y_train_predict.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ] = y_train_predict.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ].where(
                mask_obs.T.values
            )
            y_test.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ] = y_test.loc[dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))].where(
                mask_unobs.T.values
            )
            y_predict.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ] = y_predict.loc[
                dict(time=slice(f"{year}-01-01", f"{year+1}-01-01"))
            ].where(
                mask_unobs.T.values
            )

        # resample yearly
        y_test = y_test.resample(time="1y").mean()
        y_predict = y_predict.resample(time="1y").mean()
        y_train = y_train.resample(time="1y").mean()
        y_train_predict = y_train_predict.resample(time="1y").mean()

        # calculate trends
        ms_to_year = 365 * 24 * 3600 * 10**9
        test_trends = (
            y_test.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)
            * ms_to_year
        )
        predict_trends = (
            y_predict.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)
            * ms_to_year
        )
        train_trends = (
            y_train.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)
            * ms_to_year
        )
        train_predict_trends = (
            y_train_predict.polyfit(dim="time", deg=1).polyfit_coefficients.sel(
                degree=1
            )
            * ms_to_year
        )
        corr = -np.abs(test_trends - predict_trends)  # corr is neg abs diff
        corr_train = -np.abs(train_trends - train_predict_trends)
    else:
        raise AttributeError("metric not known")
    corrmap[unobslat, unobslon] = corr
    corrmap[obslat, obslon] = corr_train

    # find x gridpoints with minimum performance
    if method == "systematic":
        landpts = np.argsort(corr)[:n]
    elif method == "random":
        landpts = np.random.choice(np.arange(corr.size), size=n, replace=False)
    elif method == "interp":
        landlat = mrso_obs.lat.values.tolist() + mrso_unobs.lat.values.tolist()
        landlon = mrso_obs.lon.values.tolist() + mrso_unobs.lon.values.tolist()
        nobs = mrso_obs.shape[-1]
        dist = calc_geodist(landlon, landlat)
        np.fill_diagonal(dist, np.nan)  # inplace
        dist = dist[nobs:, :nobs]  # dist of obs to unobs
        mindist = np.nanmin(dist, axis=1)
        mindist[np.isnan(mindist)] = 0  # nans are weird in argsort
        landpts = np.argsort(mindist)[-n:]
    else:
        raise AttributeError("method not known")

    # add new gridpoints to list of observed gridpoints
    lats = (
        y_latlon.where(y_latlon.landpoints.isin(landpts), drop=True)
        .coords["lat"]
        .values[:, 0]
    )
    lons = (
        y_latlon.where(y_latlon.landpoints.isin(landpts), drop=True)
        .coords["lon"]
        .values[:, 0]
    )
    latlist = latlist + lats.tolist()
    lonlist = lonlist + lons.tolist()

    # calc mean corr for log and turn sign corrmap
    if metric == "trend":
        mean_corr = -corrmap.mean().item()
        corrmap = -corrmap
    else:
        mean_corr = corrmap.mean().item()
    logging.info(f"iteration {i} obs landpoints {len(latlist)} mean metric {mean_corr}")

    # save results
    frac_obs = mrso_obs.shape[1] / (mrso_obs.shape[1] + mrso_unobs.shape[1])
    corrmap = corrmap.assign_coords(frac_observed=frac_obs)
    # assign scalar coords bug: https://stackoverflow.com/questions/58858083/how-to-create-scalar-dimension-in-xarray
    corrmap = corrmap.expand_dims({"model": 1, "metric": 1, "strategy": 1}).copy()
    corrmap = corrmap.assign_coords(
        {
            "model": np.atleast_1d(modelname),
            "metric": np.atleast_1d(metric),
            "strategy": np.atleast_1d(method),
        }
    )
    corrmaps.append(corrmap)
    for lat, lon in zip(lats, lons):
        niter.loc[lat, lon] = i

    # set iter
    i += 1

niter = niter.expand_dims({"model": 1, "metric": 1, "strategy": 1}).copy()
niter = niter.assign_coords(
    {
        "model": np.atleast_1d(modelname),
        "metric": np.atleast_1d(metric),
        "strategy": np.atleast_1d(method),
    }
)

corrmaps = xr.concat(corrmaps, dim="frac_observed")
corrmaps = corrmaps.to_dataset(name="mrso")
corrmaps.to_netcdf(f"corrmap_{method}_{modelname}_{metric}{testcase}.nc")
niter = niter.to_dataset(name="mrso")
niter.to_netcdf(f"niter_{method}_{modelname}_{metric}{testcase}.nc")
