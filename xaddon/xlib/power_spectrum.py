#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Wed Aug  7 12:53:27 EDT 2019

import xarray as xr
from numpy.fft import rfft, rfftfreq

def power_spectrum(da, dim=None):
    '''calculate the power spectrum of the given DataArray'''
    if not isinstance(da, xr.DataArray):
        da = xr.DataArray(da)

    if dim is None:
        dim = da.dims[-1]

    # xarray-version rfft
    da_f = xr.apply_ufunc(rfft, da,
        input_core_dims=[[dim]],
        output_core_dims=[['freq']],
        kwargs={'axis': -1} )

    # power spectrum
    ps = da_f.real**2 + da_f.imag**2
    ps.name = 'power'

    # frequency coordinate
    ps['freq'] = xr.DataArray( rfftfreq(da[dim].size),
        dims='freq', attrs={'units': 'units: sample freq'})

    return ps
