#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Tue Aug 13 12:10:34 EDT 2019

import xarray as xr, numpy as np, pandas as pd
from scipy.stats import linregress as _linregress

def linregress(da_y, da_x, dim=None):
    '''xarray-wrapped function of scipy.stats.linregress.
    Note the order of the input arguments x, y is reversed to the original scipy function.'''
    if dim is None:
        dim = [d for d in da_y.dims if d in da_x.dims][0]

    slope, intercept, r, p, stderr = xr.apply_ufunc(_linregress, da_x, da_y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask='allowed')

    slope.attrs['long_name'] = 'slope of the linear regression'
    intercept.attrs['long_name'] = 'intercept of the linear regression'
    r.attrs['long_name'] = 'correlation coefficient'
    p.attrs['long_name'] = 'p-value'
    stderr.attrs['long_name'] = 'standard error of the estimated gradient'

    return xr.Dataset(dict(slope=slope, intercept=intercept, 
        r=r, p=p, stderr=stderr))
