import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import regionmask
import numpy as np
import xesmf as xe

# load files
largefilepath = '/net/so4/landclim/bverena/large_files/'
resmap = xr.open_mfdataset(f'resmap_A*_corr.nc', combine='nested').mrso

# regions
koeppen = xr.open_dataarray(f'{largefilepath}koeppen_simple.nc')
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(resmap.lon, resmap.lat)
countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(resmap.lon, resmap.lat)
countries = countries.where(~np.isnan(landmask))
pop = xr.open_dataarray(f'{largefilepath}opscaling/population_density_regridded.nc')
crop = xr.open_dataarray(f'{largefilepath}opscaling/cropland_regridded.nc')
agpop = ((crop > 50) | (pop > 30))
agpop = agpop.drop(['band','spatial_ref','raster']).squeeze()

# regrid res
regridder = xe.Regridder(resmap, koeppen, 'bilinear', reuse_weights=False)
resmap_highres = regridder(resmap)

# get region names
region_names = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.names
koeppen_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','EF','ET']
#koeppen_names = ['Af','Am','Aw','BS','Cs','Cw','Cf','Ds','Dw','Df']

# group by koeppen
res_koeppen = resmap_highres.groupby(koeppen).mean()
res_countries = resmap.groupby(countries).mean()
res_agpop = resmap.groupby(agpop).mean()

# drop desert and ice regions
res_koeppen = res_koeppen.drop_sel(group=[0,4,12,13])

# sort after correlation increase
res_koeppen = res_koeppen.sortby(res_koeppen, ascending=False)
res_countries = res_countries.sortby(res_countries, ascending=False)
countries_sorted = [region_names[int(country)] for country in res_countries.mask.values.tolist()]
koeppen_sorted = [koeppen_names[group] for group in res_koeppen.group.values]

# only 15 countries with largest increase
res_countries = res_countries[14:30] # first few are nan countries
countries_sorted = countries_sorted[14:30]

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223, projection=proj)
ax1.bar(np.arange(len(res_koeppen)), res_koeppen)
ax2.bar(np.arange(len(res_countries)), res_countries)
resmap.plot(ax=ax3, transform=transf, cmap='Greens')
ax1.set_xticks(np.arange(len(res_koeppen)))
ax2.set_xticks(np.arange(len(res_countries)))
ax1.set_xticklabels(koeppen_sorted, rotation=90)
ax2.set_xticklabels(countries_sorted, rotation=90)
ax1.set_xlabel('koeppen climate')
ax2.set_xlabel('countries')
ax3.set_title('pearson correlation improvement after doubling of stations')
plt.show()
