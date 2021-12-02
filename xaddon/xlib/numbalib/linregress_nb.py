#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Tue Jun 15 22:10:57 EDT 2021
if __name__ == '__main__':
    from misc.timer import Timer
    tt = Timer(f'start {__file__}')
#import sys, os.path, os, glob
import xarray as xr, numpy as np, pandas as pd
from numba import njit,guvectorize
from scipy.stats import t as stu
#import matplotlib.pyplot as plt
#more imports
#
if __name__ == '__main__':
    tt.check('end import')
#
#start from here
@njit
def mycorr(x, y):
    xm = np.mean(x)
    ym = np.mean(y)
    xxm = np.mean(x*x)
    yym = np.mean(y*y)
    xym = np.mean(x*y)
    return (xym - xm*ym)/np.sqrt( (xxm - xm*xm)*(yym - ym*ym) )
    
@guvectorize(["f4[:],f4[:], b1[:],f4[:],f4[:],f4[:],i4[:],f4[:],f4[:],f4[:],f4[:]"], 
    "(n),(n),()->(),(),(),(),(),(),(),(n)", target='parallel')
def linregress_core(x, y, ess_on, slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr):
    """calculate parameters associated with linear relationship between x and y.
    see https://en.wikipedia.org/wiki/Simple_linear_regression"""
    xm = np.mean(x)
    ym = np.mean(y)
    xxm = np.mean(x*x)
    yym = np.mean(y*y)
    xym = np.mean(x*y)
    s2x = xxm - xm*xm
    s2y = yym - ym*ym
    sxy = xym - xm*ym
    #slope
    slope[0] = sxy/s2x
    #intercept
    intercept[0] = ym - slope[0]*xm
    e = np.zeros(x.shape)
    for ii in range(x.size):
        e[ii] = y[ii] - intercept[0] - slope[0]*x[ii]
    eem = np.mean(e*e)
    #correlation
    r[0] = slope[0]*np.sqrt(s2x/s2y)
    N = x.size
    #effective sample size considered
    if ess_on[0]:
        s = 1.0
        for tao in range(1, N//2+1):
            #lag-1 correlation coefficient; here we use np.corrcoef
            #r1x = np.corrcoef(x[:-tao], x[tao:])[0,1]
            #r1y = np.corrcoef(y[:-tao], y[tao:])[0,1]
            #use mycorr is faster than np.corrcoef
            r1x = mycorr(x[:-tao], x[tao:])
            r1y = mycorr(y[:-tao], y[tao:])
            if r1x < 0 or r1y <0:
                break
            s = s + 2*(1 - tao/N)*r1x*r1y
        Ne = int(N/s)
        dof[0] = Ne - 2#degree of freedom in t-test
    else:
        dof[0] = N - 2
    #tvalue
    tvalue[0] = r[0]*np.sqrt( dof[0]/(1 - r[0]*r[0]) )
    #standard error of slope
    slope_stderr[0] = np.sqrt( eem/s2x/dof[0] )
    #standard error of intercept
    intercept_stderr[0] = slope_stderr[0] * np.sqrt( xxm )
    #standard error of the prediction
    for ii in range(x.size):
        predict_stderr[ii] = np.sqrt( eem/dof[0]*( 1.0 + (x[ii] - xm)**2/s2x ) )

    #return slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr

def linregress(x, y, ess_on=False, alpha=0.05):
    """wrap around linregress_core and use scipy.stats.t.cdf to calculate pvalue and t_alpha"""
    x_ = x.astype('float32')
    y_ = y.astype('float32')
    slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr = linregress_core(x_, y_, ess_on)
    pvalue = 2*stu.cdf(-np.abs(tvalue), dof)
    t_alpha = stu.ppf(1-alpha/2, dof)
    
    return slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr, pvalue, t_alpha

def xlinregress(da_y, da_x, dim=None, ess_on=False, alpha=0.05):
    """xarray-version linregress (accept xr.DataArray as input)"""
    if dim is None:
        dim = [d for d in da_y.dims if d in da_x.dims][0]
    slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr, pvalue, t_alpha = xr.apply_ufunc(
        linregress,
        da_x, da_y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], [], [], [], [dim], [], []],
        dask='allowed', kwargs={'ess_on': ess_on, 'alpha': alpha})
    ds = xr.Dataset(dict(
        slope=slope, intercept=intercept, r=r, 
        dof=dof, tvalue=tvalue, 
        slope_stderr=slope_stderr, intercetp_stderr=intercept_stderr, predict_stderr=predict_stderr,
        pvalue=pvalue, t_alpha=t_alpha))
    
    return ds

if __name__ == '__main__':
    from wyconfig import * #my plot settings
    from xaddon.xlib.linregress import linregress as linregress_da
    import xarray as xr
    from xdata import ds
    x = ds.nino34.values
    y = ds.iod.values
    
    slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr, pvalue, t_alpha = linregress(x, y)
    print(f'{slope = }; {intercept = }; {r = }; {dof = }; {tvalue = }; {slope_stderr = }; {intercept_stderr = }; {pvalue = }; {t_alpha = }')

    #print(linregress_da(xr.DataArray(y), xr.DataArray(x)))
    print('\n scipy.stats.linregress result')
    print(linregress_da(ds.iod, ds.nino34))
    print('\n xlinregress result with ess_on=False')
    print(xlinregress(ds.iod, ds.nino34, ess_on=False))
    print('\n xlinregress result with ess_on=True')
    print(xlinregress(ds.iod, ds.nino34, ess_on=True))

    da_y = ds.sst.pipe(lambda x: x.groupby('time.month') - x.groupby('time.month').mean('time') )
    da_x = ds.nino34
    tt.check('start xlinregress')
    ds_x = xlinregress(da_y, da_x)
    tt.check('start linregress_da')
    ds_da = linregress_da(da_y, da_x)
    tt.check('end linregress_da')

    plt.scatter(x, y, color='none', edgecolor='C0')
    x_ = np.sort(x)
    iis = np.argsort(x)
    predict = x_*slope + intercept
    spread = predict_stderr[iis]*t_alpha
    plt.fill_between(x_, predict-spread, predict + spread, alpha=0.5, color='C0')
    plt.plot(x_, predict, color='C0', ls='--')

    tt.check(f'**Done**')
    plt.show()
