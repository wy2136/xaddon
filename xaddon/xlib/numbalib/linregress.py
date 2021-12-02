#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Tue Jun 15 22:10:57 EDT 2021
if __name__ == '__main__':
    from misc.timer import Timer
    tt = Timer(f'start {__file__}')
#import sys, os.path, os, glob
#import xarray as xr, numpy as np, pandas as pd
import numpy as np
from numba import guvectorize
from scipy.stats import t as stu
#import matplotlib.pyplot as plt
#more imports
#
if __name__ == '__main__':
    tt.check('end import')
#
#start from here
def linregress_core(x, y, ess_on=False):
    """calculate parameters associated with linear relationship between x and y.
    see https://en.wikipedia.org/wiki/Simple_linear_regression"""
    r = np.corrcoef(x,y)[0,1]
    xm = np.mean(x)
    ym = np.mean(y)
    xxm = np.mean(x*x)
    yym = np.mean(y*y)
    xym = np.mean(x*y)
    s2x = xxm - xm*xm
    s2y = yym - ym*ym
    sxy = xym - xm*ym
    slope = sxy/s2x
    intercept = ym - slope*xm
    e = y - intercept - slope*x
    eem = np.mean(e*e)
    r = slope*np.sqrt(s2x/s2y)
    N = x.size
    if ess_on: #effective sample size used
        s = 1.0
        for tao in range(1, N//2+1):
            #lag-1 correlation coefficient; here we use np.corrcoef
            r1x = np.corrcoef(x[:-tao], x[tao:])[0,1]
            r1y = np.corrcoef(y[:-tao], y[tao:])[0,1]
            if r1x < 0 or r1y <0:
                break
            s = s + 2*(1 - tao/N)*r1x*r1y
        Ne = int(N/s)
        dof = Ne - 2
    else:
        dof = N - 2
    tvalue = r*np.sqrt( dof/(1 - r*r) )
    slope_stderr = np.sqrt( eem/s2x/dof )
    intercept_stderr = slope_stderr * np.sqrt( xxm )
    predict_stderr = np.sqrt( eem/dof*( 1.0 + (x - xm)**2/s2x ) )

    return slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr

def linregress(x, y, alpha=0.05, ess_on=False):
    slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr = linregress_core(x, y, ess_on)
    pvalue = 2*stu.cdf(-np.abs(tvalue), dof)
    t_alpha = stu.ppf(1-alpha/2, dof)
    
    return slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr, pvalue, t_alpha


     
 
if __name__ == '__main__':
    from wyconfig import * #my plot settings
    from xaddon.xlib.linregress import linregress as linregress_da
    import xarray as xr
    """
    np.random.seed(0)
    x = np.random.randn(100)
    y = np.random.randn(100)
    """
    from xdata import ds
    x = ds.nino34.values
    y = ds.iod.values
    #x = x[0::12]
    #y = y[0::12]
    #y = np.sin(np.linspace(0, np.pi*2, 100))
    print(f'{x = }')
    print(f'{y = }')
    
    slope, intercept, r, dof, tvalue, slope_stderr, intercept_stderr, predict_stderr, pvalue, t_alpha = linregress(x, y, ess_on=True)
    print(f'{slope = }; {intercept = }; {r = }; {dof = }; {tvalue = }; {slope_stderr = }; {intercept_stderr = }; {pvalue = }; {t_alpha = }')

    print(linregress_da(xr.DataArray(y), xr.DataArray(x)))

    plt.scatter(x, y)
    x_ = np.sort(x)
    iis = np.argsort(x)
    predict = x_*slope + intercept
    spread = predict_stderr[iis]*t_alpha
    plt.fill_between(x_, predict-spread, predict + spread, alpha=0.5, color='C0')
    plt.plot(x_, predict, color='C0')
    """
    plt.scatter(x, x*slope + intercept)
    plt.scatter(x, x*slope + intercept + predict_stderr*t_alpha)
    plt.scatter(x, x*slope + intercept - predict_stderr*t_alpha)
    """
    plt.show()


    tt.check(f'**Done**')
