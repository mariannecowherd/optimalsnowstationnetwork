"""
function to calculate fisher information, source: https://stackoverflow.com/questions/48695308/fisher-information-calculation-extended

    @author: Verena Bessenbacher
    @date: 21 01 2021
"""

import xarray as xr
import matplotlib.pyplot as plt

largefilepath = '/net/so4/landclim/bverena/large_files/'
data = xr.open_dataarray(largefilepath + 'real_small/features_init_idebug_True.nc')

y = data.loc[:,'swvl1']
notyvars = data.coords['variable'].values.tolist()
notyvars.remove('swvl1')
X = data.loc[:,notyvars]
 
y_small = y[:100].values
X_small = X[:100,:].values

# calc fisher info
from scipy.sparse import diags
V = diags(y_small * (1 - y_small)).toarray()
I = X_small.T.dot(V).dot(X_small)
plt.imshow(I)
plt.show()
