# -*- coding: utf-8 -*-
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import regionmask
import numpy as np
import xesmf as xe
from calc_worldarea import calc_area

# colors
colors = np.array([[81,73,171],[124,156,172],[236,197,140],[85,31,50],[189,65,70],[243,220,124]])
colors = colors/255.
col_random = colors[4,:]
col_swaths = colors[2,:]
col_real = colors[0,:]

# get region names
region_names = regionmask.defined_regions.ar6.land.abbrevs
koeppen_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','EF','ET']
koeppen_names = ['Af','Am','Aw','BS','Cs','Cw','Cf','Ds','Dw','Df']

# load files
largefilepath = '/net/so4/landclim/bverena/large_files/'
testcase = 'smmask2'
niter = xr.open_mfdataset(f'niter_systematic_*_{testcase}.nc', coords='minimal').mrso
corrmaps = xr.open_mfdataset(f'corrmap_systematic_*_{testcase}.nc', coords='minimal').mrso
obsmask = xr.open_dataarray(f'{largefilepath}opscaling/obsmask.nc').squeeze()

# calc rank percentages from iter
niter = niter / niter.max(dim=("lat", "lon")) # removed 1 - ...

# extract current and double corr
min_frac = min(corrmaps.frac_observed)
double_frac = min_frac*2
orig = corrmaps.sel(frac_observed=min_frac, method='nearest')
double = corrmaps.sel(frac_observed=double_frac, method='nearest')
#corr_increase = np.abs(double - orig)

# calculate multi model mean
orig = orig.mean(dim='model')
double = double.mean(dim='model')
#corr_increase = corr_increase.mean(dim='model').squeeze().load()
meaniter = niter.mean(dim='model').squeeze()

# drop unnecessary dimensions and coords
#meaniter = meaniter.drop(['strategy','metric']).squeeze()
#orig = orig.drop(['frac_observed','strategy','metric']).squeeze()
#double = double.drop(['frac_observed','strategy','metric']).squeeze()
meaniter = meaniter.drop(['strategy']).squeeze()
orig = orig.drop(['frac_observed','strategy']).squeeze()
double = double.drop(['frac_observed','strategy']).squeeze()

# select threshold
meaniter = meaniter < double_frac

# regions
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))
koeppen = xr.open_dataarray(f'{largefilepath}opscaling/koeppen_simple.nc')
koeppen = koeppen.drop(['band'])

# regrid for koeppen
regridder = xe.Regridder(koeppen, orig, 'bilinear', reuse_weights=False)
koeppen = regridder(koeppen)

# area per grid point
grid = calc_area(regions).drop('variable').squeeze()

# station density current and future
den_ar6_current = obsmask.groupby(regions).sum() / grid.groupby(regions).sum()
den_ar6_future = (obsmask | meaniter).groupby(regions).sum() / grid.groupby(regions).sum()

den_koeppen_current = obsmask.groupby(koeppen).sum() / grid.groupby(koeppen).sum()
den_koeppen_future = (obsmask | meaniter).groupby(koeppen).sum() / grid.groupby(koeppen).sum()

# pearson corr current and future 
orig_ar6 = orig.groupby(regions).mean()
orig_koeppen = orig.groupby(koeppen).mean()

double_ar6 = double.groupby(regions).mean()
double_koeppen = double.groupby(koeppen).mean()

# drop desert and ice regions from koeppen
orig_koeppen = orig_koeppen.drop_sel(group=[0,4,12,13])
double_koeppen = double_koeppen.drop_sel(group=[0,4,12,13])
den_koeppen_current = den_koeppen_current.drop_sel(group=[0,4,12,13])
den_koeppen_future = den_koeppen_future.drop_sel(group=[0,4,12,13])

# drop deserts and ice regions from ar6
#less than 10% covered; checked with
#np.array(region_names)[((~np.isnan(double)).groupby(regions).sum().values / xr.full_like(double,1).groupby(regions).sum().values) < 0.1]
ar6_exclude_desertice = [0,20,36,40,44,45]
orig_ar6 = orig_ar6.drop_sel(mask=ar6_exclude_desertice)
double_ar6 = double_ar6.drop_sel(mask=ar6_exclude_desertice)
den_ar6_current = den_ar6_current.drop_sel(mask=ar6_exclude_desertice)
den_ar6_future = den_ar6_future.drop_sel(mask=ar6_exclude_desertice)

# drop regions from ar6 that have less than 20% coverage with current mask
smcoupmask = xr.open_dataarray(f'{largefilepath}opscaling/landmask.nc')
tmp = np.where((((smcoupmask | obsmask)*grid).groupby(regions).sum() / grid.groupby(regions).sum() < 0.1).values)[0].tolist()
ar6_exclude_lowcov = list(set(tmp) - set(ar6_exclude_desertice))
orig_ar6 = orig_ar6.drop_sel(mask=ar6_exclude_lowcov)
double_ar6 = double_ar6.drop_sel(mask=ar6_exclude_lowcov)
den_ar6_current = den_ar6_current.drop_sel(mask=ar6_exclude_lowcov)
den_ar6_future = den_ar6_future.drop_sel(mask=ar6_exclude_lowcov)

# remove greenland # fringe station that isn't really on greenland
den_ar6_current[0] = np.nan 
den_ar6_future[:,0] = np.nan 
orig_ar6[:,0] = np.nan
double_ar6[:,0] = np.nan

# calc change 
corr_increase = np.abs(orig_ar6 - double_ar6)
corr_increase = np.round(corr_increase, 3)
den_increase = den_ar6_future - den_ar6_current
den_increase = np.round(den_increase, 3)

# create world map
#density_c = xr.full_like(regions, 0)
#for region, d in zip(range(int(regions.max().item())), den_ar6_current):
#    density_c = density_c.where(regions != region, d) # unit stations per bio square km

doubleden = xr.full_like(meaniter, 0)
for region in den_increase.mask:
    doubleden = doubleden.where(regions != region, den_increase.sel(mask=region)) # unit stations per bio square km

doubling = xr.full_like(meaniter, 0)
for region in corr_increase.mask:
    doubling = doubling.where(regions != region, corr_increase.sel(mask=region)) # unit stations per bio square km

# set no change to nan
doubleden = doubleden.where(doubleden != 0, np.nan)
doubling = doubling.where(doubling != 0, np.nan)
import IPython; IPython.embed()

# set ocean negative number
doubling = doubling.where(~np.isnan(landmask), -10)
doubleden = doubleden.where(~np.isnan(landmask), -10)

# plot constants
fs = 24
plt.rcParams.update({'font.size': fs})
cmap = plt.get_cmap('Greens').copy()
bad_color = 'lightgrey'
cmap.set_under('aliceblue')
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(25,20))
ax1 = fig.add_subplot(321, projection=proj)
ax2 = fig.add_subplot(322, projection=proj)
ax3 = fig.add_subplot(323, projection=proj)
ax4 = fig.add_subplot(324, projection=proj)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)
#ax7 = fig.add_subplot(427)
#ax8 = fig.add_subplot(428)
plt.rcParams.update({'font.size': fs})

# maps plot
doubleden[0,:,:].plot(ax=ax1, add_colorbar=False, cmap=cmap, 
                                  levels=np.arange(0,3.5,0.5), transform=transf, 
                                  vmin=0, vmax=3)
im1 = doubleden[1,:,:].plot(ax=ax2, add_colorbar=False, cmap=cmap, 
                                   levels=np.arange(0,3.5,0.5), transform=transf, 
                                   vmin=0, vmax=3)
im2 = doubling[0,:,:].plot(ax=ax3, add_colorbar=False, cmap=cmap,
                                    levels=np.arange(0,0.35,0.05), 
                                    transform=transf, vmin=0, vmax=0.3)
im3 = doubling[1,:,:].plot(ax=ax4, add_colorbar=False, cmap=cmap, 
                                    levels=np.arange(0,1.2,0.2), 
                                    transform=transf, vmin=0, vmax=1.0) # DEBUG

regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
                                         ax=ax1, add_label=False, projection=transf)
regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
                                         ax=ax2, add_label=False, projection=transf)
regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
                                         ax=ax3, add_label=False, projection=transf)
regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
                                         ax=ax4, add_label=False, projection=transf)

left = 0.16
bottom = 0.37
cbar_ax = fig.add_axes([left, 0.67, 0.7, 0.02]) # left bottom width height
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=fs)
cbar.set_label('additional stations per million $km^2$', fontsize=fs)

cbar_ax = fig.add_axes([left, bottom, 0.27, 0.02]) # left bottom width height 
cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=fs)
cbar.set_label('correlation increase', fontsize=fs)

cbar_ax = fig.add_axes([0.58, bottom, 0.27, 0.02]) # left bottom width height 
cbar = fig.colorbar(im3, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=fs)
cbar.set_label('MAE decrease [$kg\;m^{-2}$]', fontsize=fs)

#ax1.text(-0.3, 0.5,'change in station density',transform=ax1.transAxes, va='center', fontsize=fs)
#ax3.text(-0.3, 0.5,'change in performance metric',transform=ax3.transAxes, va='center', fontsize=fs)
ax1.set_title('(a) Interannual variability: change in station density', fontsize=fs)
ax2.set_title('(b) Trend: change in station density', fontsize=fs)
ax3.set_title('(c) Interannual variability: change in correlation', fontsize=fs)
ax4.set_title('(d) Trend: change in MAE', fontsize=fs)

#ax1.set_title('', fontsize=fs) 
#ax2.set_title('', fontsize=fs) 
#ax3.set_title('', fontsize=fs) 
#ax4.set_title('', fontsize=fs) 

ax1.set_facecolor(bad_color)
ax2.set_facecolor(bad_color)
ax3.set_facecolor(bad_color)
ax4.set_facecolor(bad_color)

ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax4.coastlines()

# scatter plot
a = 0.5
legend_elements = [Line2D([0], [0], marker='o', color='w', 
                   label='current station number', markerfacecolor=col_random,
                   markeredgecolor='black', alpha=a, markersize=15),
                   Line2D([0], [0], marker='o', color='w', 
                   label='globally doubled station number', markerfacecolor=col_random,
                   markeredgecolor=col_random, alpha=1, markersize=15)]

ax5.grid(0.2)
ax6.grid(0.2)
#ax7.grid(0.2)
#ax8.grid(0.2)

#ax7.text(-0.35, 0.5,'AR6 regions',transform=ax7.transAxes, va='center', fontsize=fs)
#ax5.text(-0.35, 0.5,'Koppen-Geiger \nclimates',transform=ax5.transAxes,fontsize=fs)
s = 250

#'Af','Am','Aw','BS','Cs','Cw','Cf','Ds','Dw','Df'
y_offset = [0.02,-0.02,0,0,0,0,0,0,-0.01,0]
for x1,y1,x2,y2 in zip(den_koeppen_current,orig_koeppen[0,:],den_koeppen_future[0,:],double_koeppen[0,:]):
    ax5.plot([x1, x2], [y1, y2], c=col_random, alpha=a, linewidth=2)
for n, (label,x,y) in enumerate(zip(koeppen_names, den_koeppen_future[0,:], double_koeppen[0,:])):
    ax5.text(x=x+0.10,y=y+y_offset[n],s=label)
ax5.scatter(den_koeppen_current,orig_koeppen[0,:], c=col_random, edgecolor='black', alpha=a, s=s)
ax5.scatter(den_koeppen_future[0,:],double_koeppen[0,:], c=col_random, s=s)

#'Af','Am','Aw','BS','Cs','Cw','Cf','Ds','Dw','Df'
y_offset = [0,0,0,0,-0.15,0,0,0,0,0]
for x1,y1,x2,y2 in zip(den_koeppen_current,orig_koeppen[1,:],den_koeppen_future[1,:],double_koeppen[1,:]):
    ax6.plot([x1, x2], [y1, y2], c=col_random, alpha=a, linewidth=2)
for n, (label,x,y) in enumerate(zip(koeppen_names, den_koeppen_future[1,:], double_koeppen[1,:])):
    ax6.text(x=x+0.10,y=y+y_offset[n],s=label)
ax6.scatter(den_koeppen_current,orig_koeppen[1,:], c=col_random, edgecolor='black', alpha=a, s=s)
ax6.scatter(den_koeppen_future[1,:],double_koeppen[1,:], c=col_random, s=s)

#for x1,y1,x2,y2 in zip(den_ar6_current,orig_ar6[0,:],den_ar6_future[0,:],double_ar6[0,:]):
#    ax7.plot([x1, x2], [y1, y2], c=col_random, alpha=a)
#for label,x,y in zip(region_names, den_ar6_future[0,:], double_ar6[0,:]):
#    ax7.text(x=x+0.08,y=y,s=label)
#ax7.scatter(den_ar6_current,orig_ar6[0,:], c=col_random, edgecolor='black', alpha=a)
#ax7.scatter(den_ar6_future[0,:],double_ar6[0,:], c=col_random)
#
#for x1,y1,x2,y2 in zip(den_ar6_current,orig_ar6[1,:],den_ar6_future[1,:],double_ar6[1,:]):
#    ax8.plot([x1, x2], [y1, y2], c=col_random, alpha=a)
#for label,x,y in zip(region_names, den_ar6_future[1,:], double_ar6[1,:]):
#    ax8.text(x=x+0.08,y=y,s=label)
#ax8.scatter(den_ar6_current,orig_ar6[1,:], c=col_random, edgecolor='black', alpha=a)
#ax8.scatter(den_ar6_future[1,:],double_ar6[1,:], c=col_random)

#ax7.set_xlabel('stations per Mio $km^2$', fontsize=fs)
#ax8.set_xlabel('stations per Mio $km^2$', fontsize=fs)
ax5.set_xlabel('stations per Mio $km^2$', fontsize=fs)
ax6.set_xlabel('stations per Mio $km^2$', fontsize=fs)
ax5.set_ylabel('Pearson correlation', fontsize=fs)
#ax7.set_ylabel('pearson correlation', fontsize=fs)
ax6.set_ylabel('MAE [$kg\;m^{-2}$]', fontsize=fs)
#ax8.set_ylabel('MAE', fontsize=fs)

ax5.set_ylim([0.3,0.8])
#ax7.set_ylim([0.15,0.9])
ax6.set_ylim([0.5,3])
#ax8.set_ylim([0.5,3.75])

ax5.set_xlim([-0.2,6.5])
ax6.set_xlim([-0.2,6.5])
#ax7.set_xlim([-0.2,12])
#ax8.set_xlim([-0.2,15])

ax5.set_title('(e) Interannual variability: \nchange per Koppen-Geiger climate', fontsize=fs)
ax6.set_title('(f) Long-term trend: \nchange per Koppen-Geiger climate', fontsize=fs)
#ax7.set_title('g)')
#ax8.set_title('h)')

ax5.legend(handles=legend_elements, loc='lower right')
plt.subplots_adjust(hspace=0.75)
plt.rcParams.update({'font.size': fs})
plt.savefig('doubling.pdf')
