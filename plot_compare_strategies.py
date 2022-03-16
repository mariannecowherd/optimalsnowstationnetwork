import pickle
import numpy as np
import matplotlib.pyplot as plt


modelname = 'IPSL-CM6A-LR'
with open(f"corr_random_{modelname}.pkl", "rb") as f:
    corr_random = pickle.load(f)

with open(f"corr_systematic_{modelname}.pkl", "rb") as f:
    corr_sys = pickle.load(f)

with open(f"corr_interp_{modelname}.pkl", "rb") as f:
    corr_interp = pickle.load(f)

with open(f"nobs_random_{modelname}.pkl", "rb") as f:
    nobs_random = pickle.load(f)

with open(f"nobs_systematic_{modelname}.pkl", "rb") as f:
    nobs_sys = pickle.load(f)

with open(f"nobs_interp_{modelname}.pkl", "rb") as f:
    nobs_interp = pickle.load(f)

# convert number of stations to percentages
total_no_of_stations = nobs_random[-1]
nobs_random = np.array(nobs_random)
nobs_sys = np.array(nobs_sys)
nobs_interp = np.array(nobs_interp)
frac_random = nobs_random / total_no_of_stations
frac_sys = nobs_sys / total_no_of_stations
frac_interp = nobs_interp / total_no_of_stations

# plot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set_title(modelname)
ax.plot(frac_random, corr_random, label='random station placing')
ax.plot(frac_interp, corr_interp, label='interp station placing')
ax.plot(frac_sys, corr_sys, label='systematic station placing')
ax.set_ylabel('pearson correlation')
ax.set_xlabel('percentage observed points')
ax.vlines(frac_random[0], ymin=0.4, ymax=1, colors='grey')
ax.text(0.16, 0.98, 'current ISMN')
ax.legend(loc='lower right')
plt.show()
