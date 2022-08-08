import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import regionmask
import numpy as np
import xesmf as xe

# TODO
# size of the marker is size of the region

colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]

# load files
largefilepath = '/net/so4/landclim/bverena/large_files/'
orig = xr.open_mfdataset(f'orig_*_corr.nc', combine='nested', concat_dim='model').mrso # DEBUG TODO ALL FILES
double = xr.open_mfdataset(f'double_*_corr.nc', combine='nested', concat_dim='model').mrso
meaniter = xr.open_dataarray('meaniter.nc')
meaniter = meaniter.sel(metric='_corr') < 0.25
obsmask = xr.open_dataarray(f'{largefilepath}opscaling/obsmask.nc') # DEBUG later all models
orig = orig.mean(dim='model')
double = double.mean(dim='model')

# regions
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))
koeppen = xr.open_dataarray(f'{largefilepath}opscaling/koeppen_simple.nc')
countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(orig.lon, orig.lat)
countries = countries.where(~np.isnan(landmask))
pop = xr.open_dataarray(f'{largefilepath}opscaling/population_density_regridded.nc')
crop = xr.open_dataarray(f'{largefilepath}opscaling/cropland_regridded.nc')
agpop = ((crop > 50) | (pop > 30))
agpop = agpop.drop(['band','spatial_ref','raster']).squeeze()

# area per grid point
res = np.abs(np.diff(regions.lat)[0]) # has to be resolution of "regions" for correct grid area calc
grid = xr.Dataset({'lat': (['lat'], regions.lat.data),
                   'lon': (['lon'], regions.lon.data)})
shape = (len(grid.lat),len(grid.lon))
earth_radius = 6371*1000 # in m
weights = np.cos(np.deg2rad(grid.lat))
area = earth_radius**2 * np.deg2rad(res) * weights * np.deg2rad(res)
area = np.repeat(area.values, shape[1]).reshape(shape)
grid['area'] = (['lat','lon'], area)
grid = grid.to_array() / (1000*1000) # to km**2
grid = grid / (1000*1000) # to Mio km**2

# area per grid point KOEPPEN
res = np.abs(np.diff(koeppen.lat)[0]) # has to be resolution of "regions" for correct grid area calc
grid_k = xr.Dataset({'lat': (['lat'], koeppen.lat.data),
                   'lon': (['lon'], koeppen.lon.data)})
shape = (len(grid_k.lat),len(grid_k.lon))
earth_radius = 6371*1000 # in m
weights = np.cos(np.deg2rad(grid_k.lat))
area = earth_radius**2 * np.deg2rad(res) * weights * np.deg2rad(res)
area = np.repeat(area.values, shape[1]).reshape(shape)
grid_k['area'] = (['lat','lon'], area)
grid_k = grid_k.to_array() / (1000*1000) # to km**2
grid_k = grid_k / (1000*1000) # to Mio km**2

# regrid for koeppen
regridder = xe.Regridder(orig, koeppen, 'bilinear', reuse_weights=False)
orig_highres = regridder(orig)
double_highres = regridder(double)
obsmask_highres = regridder(obsmask)
meaniter_highres = regridder(meaniter)

# station density current and future
den_ar6_current = obsmask.groupby(regions).sum() / grid.groupby(regions).sum()
den_ar6_future = (obsmask | meaniter).groupby(regions).sum() / grid.groupby(regions).sum()
den_ar6 = (den_ar6_future / den_ar6_current).squeeze()

den_countries_current = obsmask.groupby(countries).sum() / grid.groupby(countries).sum()
den_countries_future = (obsmask | meaniter).groupby(countries).sum() / grid.groupby(countries).sum()
den_countries = (den_countries_future / den_countries_current).squeeze()

den_koeppen_current = obsmask_highres.groupby(koeppen).sum() / grid_k.groupby(koeppen).sum()
den_koeppen_future = (obsmask_highres | meaniter_highres).groupby(koeppen).sum() / grid_k.groupby(koeppen).sum()
den_koeppen = (den_koeppen_future / den_koeppen_current).squeeze()

# get region names
countries_names = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.names
region_names = regionmask.defined_regions.ar6.land.names
koeppen_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','EF','ET']

# group by region
orig_ar6 = orig.groupby(regions).mean()
orig_koeppen = orig_highres.groupby(koeppen).mean()
orig_countries = orig.groupby(countries).mean()
orig_agpop = orig.groupby(agpop).mean()

double_ar6 = double.groupby(regions).mean()
double_koeppen = double_highres.groupby(koeppen).mean()
double_countries = double.groupby(countries).mean()
double_agpop = double.groupby(agpop).mean()

# drop desert and ice regions from koeppen
orig_koeppen = orig_koeppen.drop_sel(koeppen_class=[0,4,12,13])
double_koeppen = double_koeppen.drop_sel(koeppen_class=[0,4,12,13])
den_koeppen_current = den_koeppen_current.drop_sel(koeppen_class=[0,4,12,13])
den_koeppen_future = den_koeppen_future.drop_sel(koeppen_class=[0,4,12,13])
koeppen_names = ['Af','Am','Aw','BS','Cs','Cw','Cf','Ds','Dw','Df']

# drop deserts and ice regions from ar6
#less than 10% covered; checked with
#np.array(region_names)[((~np.isnan(double)).groupby(regions).sum().values / xr.full_like(double,1).groupby(regions).sum().values) < 0.1]
ar6_exclude_desertice = [0,20,36,40,44,45]
orig_ar6 = orig_ar6.drop_sel(mask=ar6_exclude_desertice)
double_ar6 = double_ar6.drop_sel(mask=ar6_exclude_desertice)
den_ar6_current = den_ar6_current.drop_sel(mask=ar6_exclude_desertice)
den_ar6_future = den_ar6_future.drop_sel(mask=ar6_exclude_desertice)

# drop ar6 regions where no stations were added for cleaning up plot
ar6_exclude = den_ar6_future.mask[den_ar6_future.squeeze() == 0]
ar6_include = den_ar6_future.mask[den_ar6_future.squeeze() != 0] 
orig_ar6 = orig_ar6.drop_sel(mask=ar6_exclude)
double_ar6 = double_ar6.drop_sel(mask=ar6_exclude)
den_ar6_current = den_ar6_current.drop_sel(mask=ar6_exclude)
den_ar6_future = den_ar6_future.drop_sel(mask=ar6_exclude)

idxs = np.arange(46)
idxs = [idx for idx in idxs if idx not in ar6_exclude_desertice]
idxs = [idx for idx in idxs if idx not in ar6_exclude]
region_names = np.array(region_names)[idxs]

# create legend
a = 0.5
legend_elements = [Line2D([0], [0], marker='o', color='w', 
                   label='current station density', markerfacecolor=col_real,
                   markeredgecolor='black', alpha=a, markersize=10),
                   Line2D([0], [0], marker='o', color='w', 
                   label='double station density', markerfacecolor=col_real,
                   markeredgecolor=col_real, alpha=1, markersize=10)]


# plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.grid()
ax2.grid()


for x1,y1,x2,y2 in zip(den_koeppen_current,orig_koeppen,den_koeppen_future,double_koeppen):
    ax1.plot([x1, x2], [y1, y2], c=col_real, alpha=a)
for label,x,y in zip(koeppen_names, den_koeppen_future, double_koeppen):
    ax1.text(x=x,y=y,s=label)
ax1.scatter(den_koeppen_current,orig_koeppen, c=col_real, edgecolor='black', alpha=a)
ax1.scatter(den_koeppen_future,double_koeppen, c=col_real)


for x1,y1,x2,y2 in zip(den_ar6_current,orig_ar6,den_ar6_future,double_ar6):
    ax2.plot([x1, x2], [y1, y2], c=col_random, alpha=a)
for label,x,y in zip(region_names, den_ar6_future, double_ar6):
    ax2.text(x=x,y=y,s=label)
ax2.scatter(den_ar6_current,orig_ar6, c=col_random, edgecolor='black', alpha=a)
ax2.scatter(den_ar6_future,double_ar6, c=col_random)


ax1.set_xlabel('stations per Mio $km^2$')
ax2.set_xlabel('stations per Mio $km^2$')

ax1.set_ylabel('pearson correlation')
ax2.set_ylabel('pearson correlation')
ax1.set_ylim([0.15,0.9])
ax2.set_ylim([0.15,0.9])

ax1. legend(handles=legend_elements)

plt.savefig('doubling_scatter.pdf')
