#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Thu Aug  8 11:18:09 EDT 2019

import xarray as xr

from .api import detrend as _detrend
from .api import find_peaks as _find_peaks
from .api import linregress as _linregress
from .api import power_spectrum as _power_spectrum

@xr.register_dataarray_accessor('go')
class AddonAccessor(object):
    def __init__(self, da):
        self._obj = da
    
    def detrend(self, *args, **kwargs):
        '''see xaddon.api.detrend'''
        return _detrend(self._obj, *args, **kwargs)

    def find_peaks(self, *args, **kwargs):
        '''see xaddon.api.find_peaks'''
        return _find_peaks(self._obj, *args, **kwargs)

    def power_spectrum(self, *args, **kwargs):
        '''see xaddon.api.power_spectrum'''
        return _power_spectrum(self._obj, *args, **kwargs)

@xr.register_dataarray_accessor('linregress')
class LinearRegressAccessor(object):
    def __init__(self, da):
        self._obj = da

    def on(self, *args, **kwargs):
        '''see xaddon.api.linregress'''
        return _linregress(self._obj, *args, **kwargs)
