#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Tue Aug 13 12:10:34 EDT 2019

import xarray as xr, numpy as np, pandas as pd
from scipy.stats import linregress as _linregress
from scipy.stats import t as stu

def linregress(da_y, da_x, dim=None, ess_on=False):
    '''xarray-wrapped function of scipy.stats.linregress.
    Note the order of the input arguments x, y is reversed to the original scipy function. 
    ess_on: effective sample size is on.'''
    if dim is None:
        dim = [d for d in da_y.dims if d in da_x.dims][0]

    slope, intercept, r, p, stderr = xr.apply_ufunc(_linregress, da_x, da_y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask='allowed')
    """
    rg = xr.apply_ufunc(_linregress, da_x, da_y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask='allowed')
    slope, intercept, r, p, stderr, intercept_stderr = rg.slope, rg.intercept, rg.r, rg.p, rg.stderr, rg.intercept_stderr
    """
    predicted = da_x * slope + intercept

    slope.attrs['long_name'] = 'slope of the linear regression'
    intercept.attrs['long_name'] = 'intercept of the linear regression'
    r.attrs['long_name'] = 'correlation coefficient'
    p.attrs['long_name'] = 'p-value'
    stderr.attrs['long_name'] = 'standard error of the estimated slope'
    #intercept_stderr.attrs['long_name'] = 'standard error of the estimated intercept'
    predicted.attrs['long_name'] = 'predicted values by the linear regression model'

    ds = xr.Dataset(dict(slope=slope, intercept=intercept, 
        r=r, p=p, stderr=stderr, predicted=predicted))

    if ess_on:
        N = da_y[dim].size
        #lag-1 correlation of da_x and da_y
        _slope, _intercept, r1x, _p, _stderr = xr.apply_ufunc(_linregress, 
            da_x.isel({dim: slice(0, -1)}).drop(dim), da_x.isel({dim: slice(1, None)}).drop(dim),
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[], [], [], [], []],
            vectorize=True,
            dask='allowed')
        _slope, _intercept, r1y, _p, _stderr = xr.apply_ufunc(_linregress, 
            da_y.isel({dim: slice(0, -1)}).drop(dim), da_y.isel({dim: slice(1, None)}).drop(dim),
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[], [], [], [], []],
            vectorize=True,
            dask='allowed')
        Nef = N*(1 - r1x*r1y)/(1 + r1x*r1y)
        Nef = Nef.astype('int32')
        Nef.attrs['long_name'] = 'effective sample size'
        dof = Nef - 2
        tvalue = r * np.sqrt(dof/(1 - r**2))
        pe = 2*stu.cdf(-np.abs(tvalue), dof)
        pe = xr.DataArray(pe, dims=p.dims, coords=p.coords)
        pe.attrs['long_name'] = 'p-value taking into account effective sample size'

        ds['Nef'] = Nef
        ds['pe'] = pe

    return ds 
