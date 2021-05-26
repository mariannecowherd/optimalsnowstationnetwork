"""
function to calculate fisher information, source: https://stackoverflow.com/questions/48695308/fisher-information-calculation-extended

    @author: Verena Bessenbacher
    @date: 21 01 2021
"""

from namelist import largefilepath
import xarray as xr
import matplotlib.pyplot as plt

data = xr.open_dataarray(largefilepath + 'features_init_real_None_smart_idebug_True.nc').T

y = data.loc[:,'swvl1']
notyvars = data.coords['variable'].values.tolist()
notyvars.remove('swvl1')
X = data.loc[:,notyvars]
 
y_small = y[::100000].values
X_small = X[::100000,:].values

# calc fisher info
from scipy.sparse import diags
V = diags(y_small * (1 - y_small)).toarray()
I = X_small.T.dot(V).dot(X_small)
plt.imshow(I)
plt.show()
