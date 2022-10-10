# Optimising soil moisture station networks for future climates

This is the code used for producing the results in "Optimising soil moisture station networks for future climates" (submitted to GRL). It can be used to be applied to other station networks. 

This repository is very much beta and will improve once the accompanying study is published.

The workflow is as follows:

1) extract stations as netcdf file from your local copy of the ISMN data with <preproc_ismn.py>
2) extract soil moisture, temperature and precipitation from your local copy of the CMIP6(ng) data with <preproc_cmip_land.py>
3) run the up-scaling framework as described in the paper with <upscale_cmip.py> for each CMIP6 model and each performance metric separately.
4) (optionally) plot the results with the plot scripts provided.
