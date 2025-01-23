import xarray as xr
import matplotlib.pyplot as plt

mod = xr.open_dataset('y_test.nc').mrso
obs = xr.open_dataset('y_predict.nc').mrso

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

mod = mod.sel(landpoints=0)
obs = obs.sel(landpoints=0)

# absolute values
mod.plot(ax=ax1)
obs.plot(ax=ax1)

# mean seasonal cycle
meanmod = mod.groupby('time.month').mean()
meanobs = obs.groupby('time.month').mean()
meanmod.plot(ax=ax2)
meanobs.plot(ax=ax2)

# anomalies
anommod= mod.groupby('time.month') - meanmod
anomobs= obs.groupby('time.month') - meanobs
anommod.plot(ax=ax3)
anomobs.plot(ax=ax3)

# trend
ms_to_year = 365*24*3600*10**9
trendmod = mod.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year
trendobs = obs.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year

ax4.scatter(0,trendmod.values, color='blue')
ax4.scatter(1,trendobs.values, color='orange')
ax4.set_ylim([-4,0])
ax4.set_xlim([-0.5,1.5])

ax1.set_title('(a) Absolute values')
ax2.set_title('(b) Mean seasonal cycle')
ax3.set_title('(c) Anomalies')
ax4.set_title('(d) Trend')

plt.show()
