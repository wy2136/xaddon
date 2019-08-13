# xaddon
A collection of lightly xarray-wrapped functions that are often used in atmospheric, ocean and climate science.

## example 

        import xarray as xr
        import numpy as np
        import xaddon

        np.random.seed(0)
        da = xr.DataArray(np.random.randn(100), dims='day', name='Ta', attrs={'units': 'degC'})

        da.go.power_spectrum() \
            .plot()
        da.go.power_spectrum() \
            .go.find_peaks().peaks \
            .plot(marker='x', linestyle='none')

![power_spectrum_peaks](example/power_spectrum_peaks.png)

## notebook examples
* [linear regression](https://nbviewer.jupyter.org/github/wy2136/xaddon/blob/master/example/xaddon_example_linear_regression.ipynb)
* [power spectrum and peaks](https://nbviewer.jupyter.org/github/wy2136/xaddon/blob/master/example/xaddon_example.ipynb)
