#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Thu Aug  8 11:50:34 EDT 2019
import xarray as xr
from scipy.signal import find_peaks as _find_peaks

def find_peaks(da, **kwargs):
    '''xarray-version scipy.signal.find_peaks'''
    assert len(da.dims) == 1, 'input DataArray must be 1-D'

    dim = da.dims[0]
    if len(da.coords)==0: # no coords
        da[dim] = range(da.size)

    # use scipy function to find the indices of peaks
    idices_of_peaks, peaks_info = _find_peaks(da, **kwargs)
    
    # select and sort peaks
    peaks = da.isel({dim: idices_of_peaks})
    peaks = peaks.sortby(peaks, ascending=False)

    # wrap into a dataset
    ds = xr.Dataset({'peaks': peaks})
    for key, value in peaks_info.items():
        ds[key] = xr.DataArray(value, dims=dim, coords=[ds[dim]])

    return ds
