# python 3 compatiblity


import numpy as np
import xarray as xr


# this automatically converts dask arrays to numpy
# use with caution
def regrid_vertical(qdarr, trdarr, trlevs, dim):
    """Regrid a DataArray ``q``, co-located with tracer ``trarr``
    to a new vertical grid with levels defined by ``trlevs`` along
    the specified dimension."""

    dims = list(qdarr.dims)
    ax = dims.index(dim)
    qtr = _regrid_vertical(qdarr.values, trdarr.values, trlevs, axis=ax)
    # make new coordinates
    trcoord = 0.5 * (trlevs[1:] + trlevs[:-1])
    trcoord_name = trdarr.name + "_coord"
    coords = {}
    dims[ax] = trcoord_name
    for d in dims:
        try:
            coords[d] = qdarr.coords[d]
        except KeyError:
            pass
    coords[trcoord_name] = trcoord
    return xr.DataArray(qtr, coords, dims)


# numpy version
def _regrid_vertical(q, tr, trlevs, axis=0, extra_mask=None):
    """Regrid a variable q into tracer coordinates defined by trlevs
    along a specified axis."""
    assert q.shape == tr.shape
    Nbins = len(trlevs) - 1
    # make sure the vertical axis is axis 0
    if axis != 0:
        q = q.swapaxes(0, axis)
        tr = tr.swapaxes(0, axis)
    # reshape to flatten other axes
    shape_orig = q.shape
    Npts = np.prod(q.shape[1:])
    Nr = q.shape[0]
    q = q.reshape((Nr, Npts))
    tr = tr.reshape((Nr, Npts))
    # mask any values of outside the allowed range
    # tr_m = np.ma.masked_greater_equal(
    #            np.ma.masked_less(tr, trlevs.min()), trlevs.max())
    # if self.extra_mask is not None:
    #    tr_m.mask += self.extra_mask
    # mask = tr_m.mask

    # get indices of bins for whole array
    idx = np.digitize(tr.ravel(), trlevs) - 1
    idx.shape = tr.shape

    # if there were values below the bottom bin, idx will have values less than 0
    idx[idx < 0] = 0
    idx[idx >= Nbins] = Nbins - 1

    # now do the bin counting for each point
    qtr = np.zeros((Nbins, Npts))
    for n in range(Npts):
        if Nr == 1:
            # can use a simple index array
            qtr[idx[:, n], n] = q[:, n]
        else:
            qtr[:, n] = np.bincount(idx[:, n], weights=q[:, n], minlength=Nbins)[:Nbins]
    qtr = qtr.reshape((Nbins,) + shape_orig[1:])
    if axis != 0:
        qtr = qtr.swapaxes(0, axis)
    return qtr
