# make some functions for taking divergence
from dask import array as da
import xray
import numpy as np

class MITgcmDataset(object):

    def __init__(self, ds):
        self.ds = ds

    def _get_coords_from_dims(self, dims, replace=None):
        dims = list(dims)
        if replace:
            for k in replace:
                dims[dims.index(k)] = replace[k]
        return {dim: self.ds[dim] for dim in dims}, dims
    
    def extend_zl_to_zp1(self, darr, fill_value=0.):
        coords, dims = self._get_coords_from_dims(darr.dims)
        zdim = dims.index('Zl')
        # shape of the new array to concat at the bottom
        shape = list(darr.shape)
        shape[zdim] = 1
        # replace Zl with the bottom level
        coords['Zl'] = np.atleast_1d(self.ds['Zp1'][-1].data)
        # an array of zeros at the bottom
        # need different behavior for numpy vs dask
        if darr.chunks:
            chunks = list(darr.data.chunks)
            chunks[zdim] = (1,)
            zarr = fill_value * da.ones(shape, dtype=darr.dtype, chunks=chunks)
        else:
            zarr = np.zeros(shape, darr.dtype)
        zeros = xray.DataArray(zarr, coords, dims)
        newdarr = xray.concat([darr, zeros], dim='Zl').rename({'Zl':'Zp1'})
        if newdarr.chunks:
            return newdarr.chunk({'Zp1': len(newdarr.Zp1)})
        else:
            return newdarr

    def diff_zp1_to_z(self, darr):
        a_up = darr.isel(Zp1=slice(None,-1))
        a_dn = darr.isel(Zp1=slice(1,None))
        a_diff = a_up.data - a_dn.data
        print a_diff.shape
        # dimensions and coords of new array
        coords, dims = self._get_coords_from_dims(darr.dims, replace={'Zp1':'Z'})
        return xray.DataArray(a_diff, coords, dims,
                              name=darr.name+'_diff_zp1_to_z')
    
    def diff_zl_to_z(self, darr):
        darr_zp1 = self.extend_zl_to_zp1(darr)
        darr_diff = self.diff_zp1_to_z(darr_zp1)
        return darr_diff.rename(darr.name + '_diff_zl_to_z')
    
    # doesn't actually need parent ds
    # this could go in xray
    def roll(self, darr, n, dim):
        """Clone of numpy.roll for xray DataArrays."""
        left = darr.isel(**{dim:slice(None,-n)})
        right = darr.isel(**{dim:slice(-n,None)})
        return xray.concat([right, left], dim=dim)
    
    def diff_xp1_to_x(self, darr):
        """Difference DataArray ``darr`` in the x direction.
        Assumes that ``darr`` is located at the xp1 point."""
        left = darr
        right = self.roll(darr, -1, 'Xp1')
        if darr.chunks:
            right = right.chunk(darr.chunks)
        diff = right.data - left.data
        coords, dims = self._get_coords_from_dims(darr.dims, replace={'Xp1':'X'})
        return xray.DataArray(diff, coords, dims
                              ).rename(darr.name + '_diff_xp1_to_x')
    
    def diff_yp1_to_y(self, darr):
        """Difference DataArray ``darr`` in the y direction.
        Assumes that ``darr`` is located at the yp1 point."""
        left = darr
        right = self.roll(darr, -1, 'Yp1')
        if darr.chunks:
            right = right.chunk(darr.chunks)
        diff = right.data - left.data
        coords, dims = self._get_coords_from_dims(darr.dims, replace={'Yp1':'Y'})
        return xray.DataArray(diff, coords, dims
                              ).rename(darr.name + '_diff_yp1_to_y')