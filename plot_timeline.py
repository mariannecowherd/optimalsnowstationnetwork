import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import xesmf as xe

largefilepath = '/net/so4/landclim/bverena/large_files/'
stations = pd.read_csv(f'{largefilepath}station_info_grid.csv')
case = 'latlontime'

# get number of stations per year
start_year = []
end_year = []
for s in range(stations.shape[0]):
    start_year.append(int(stations.start[s][:4]))
    end_year.append(int(stations.end[s][:4]))

start_year = np.array(start_year)
end_year = np.array(end_year)

no_stations = []
for year in np.arange(1950,2021):
    no_stations.append(((year >= start_year) & (year <= end_year)).sum())

no_years = []
for start, end in zip(stations.start, stations.end):
    no_years.append(pd.date_range(start,end,freq='y').shape[0])

# plot timeline of number of stations and prediction
#pred = f'{largefilepath}RFpred_{case}.nc'
#orig = f'{largefilepath}ERA5_{case}.nc'
#unc = f'{largefilepath}UncPred_{case}.nc'
#pred = xr.open_dataarray(pred)
#orig = xr.open_dataarray(orig)
#unc = xr.open_dataarray(unc)

# regrid to koeppen grid
#regridder = xe.Regridder(unc, koeppen, 'bilinear', reuse_weights=False)
#unc = regridder(unc)
#pred = regridder(pred)
#orig = regridder(orig)

# aggregate on time per koeppen area
#import IPython; IPython.embed()
## TODO continue here
#for i in range(13):
#    unc.where(koeppen == i).mean(dim=('lat','lon')).plot()
#    np.sqrt(((pred - orig)**2).mean(dim=('lat','lon'))).plot()

#unc_t = np.zeros(len(no_stations))
#pre_t = np.zeros(len(no_stations))
#unc_t[29:65] = unc.values
#pre_t[29:65] = pred.values

reduced_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','ET','EF']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
ax.plot(no_stations)
#ax.plot(unc_t*1000)
#ax.plot(pre_t*1000)
ax.set_xticks(np.arange(0,80,10))
ax.set_xticklabels(np.arange(1950,2030,10))
ax.set_xlabel('year')
ax.set_ylabel('number of stations')
plt.show()
