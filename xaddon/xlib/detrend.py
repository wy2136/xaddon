#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Tue Aug 13 13:13:23 EDT 2019

import xarray as xr, numpy as np, pandas as pd
from scipy.signal import detrend as _detrend

def detrend(da, dim=None, **kwargs):
    '''xarray-wrapped version of scipy.signal.detrend.'''
    if dim is None:
        assert da.ndim == 1, 'only 1-D array is allowed when dim is None; specify dim explicitly for higher-dimension array'
        axis = 0
    else:
        axis = da.dims.index(dim)
    kwargs['axis'] = axis

    return xr.apply_ufunc(_detrend, da,
        kwargs=kwargs,
        dask='allowed')
        
