# Optimising soil moisture station networks for future climates

This is the code used for producing the results in "Optimising soil moisture station networks for future climates" (GRL https://doi.org/10.1029/2022GL101667, [1] ). It can be adapted and used to be applied to other station networks, other metrics or other setups.

The workflow is as follows:

1) extract stations as netcdf file from your local copy of the ISMN data with <preproc_ismn.py>
2) extract soil moisture, temperature and precipitation from your local copy of the CMIP6(ng) data with <preproc_cmip6.py>
3) run the up-scaling framework as described in the paper with <upscale_cmip.py> for each CMIP6 model and each performance metric separately.
4) (optionally) plot the results with the plot scripts provided.

Resources:
  [1] Bessenbacher, V., Gudmundsson, L. and Seneviratne, S. I. (2023): Optimizing soil moisture station networks for future climates. Geophysical Research Letters, 50, e2022GL101667. https://doi. org/10.1029/2022GL101667
