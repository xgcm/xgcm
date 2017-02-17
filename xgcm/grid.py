from __future__ import print_function
from future.utils import iteritems
from collections import OrderedDict
import pytest
import xarray as xr
import numpy as np

from . import comodo


class Grid:
    """An object that knows how to interpolate and take derivatives."""

    def __init__(self, ds, check_dims=True):
        """Create a new Grid object from an input dataset

        PARAMETERS
        ----------
        ds : xarray.Dataset
            Contains the relevant grid information
        check_dims : bool, optional
            Whether to check the compatibility of input data dimensions before
            performing grid operations.
        """
        self._ds = ds
        self._check_dims = check_dims


        self._axes = OrderedDict()
        for ax in ['X', 'Y']:
            # figure out what the grid dimensions are
            coord_names = comodo.get_axis_coords(ds, ax)
            ncoords = len(coord_names)
            if ncoords == 0:
                # didn't find anything for this axis
                pass
            else:
                if ncoords != 2:
                    raise ValueError('Must have two different %s coordinates. '
                                     'Instead got %s' % (ax, repr(coord_names)))
                axis_data = OrderedDict()
                for name in coord_names:
                    coord = ds[name]
                    axis_shift = coord.attrs.get('c_grid_axis_shift')
                    if (axis_shift is None) or (axis_shift == 0):
                        # we found the center coordinate
                        axis_data['c'] = name
                        axis_data['c_coord'] = coord
                    elif (axis_shift==0.5) or (axis_shift==-0.5):
                        axis_data['g'] = name
                        axis_data['g_coord'] = coord
                        # TODO: clearly document the sign convention
                        axis_data['shift'] = 1 if axis_shift==0.5 else -1
                    else:
                        raise ValueError('Invalid c_grid_axis_shift (%g) for '
                                         'coord %s' % (axis_shift, name))
                self._axes[ax] = axis_data


    def __repr__(self):
        summary = ['<xgcm.Grid>']

        for ax, info in iteritems(self._axes):
            axis_info = ('%s-axis: %s (cell center), %s (cell face, shift %g)' %
                         (ax, info['c'], info['g'], info['shift']))
            summary.append(axis_info)
        return '\n'.join(summary)

    def interp_c_to_g(self, da, axis):
        """Interpolate dataarray from c grid to u grid.

        Parameters
        ----------
        da : xarray.dataarray
            Original data on the t grid
        axis : {'X', 'Y'}
            Dimension along which to interpolate

        Returns
        -------
        da_i : xarray.dataarray
            Interpolated data on the u grid
        """

        ax = self._axes[axis]
        da_shift = self.shift(da, ax['c'], -ax['shift'])
        # TODO: generalize to higher order interpolation
        data_interp = 0.5*(da.data + da_shift.data)
        # wrap in a new DataArray
        da_i = da.copy()
        da_i.data = data_interp
        da_i = _replace_dim(da_i, ax['c'], ax['g_coord'])
        return da_i

    def shift(self, da, dim, shift):
        """Shift the values of da along the specified dimension.
        """
        # TODO: generalize rolling function, allow custom shifts, handle
        # boundary conditions, etc.
        return da.roll(**{dim: shift})

def _replace_dim(da, olddim, newdim, drop=True):
    """Replace a dimension with a new dimension

    PARAMETERS
    ----------
    da : xarray.DataArray
    olddim : str
        name of the dimension to replace
    newdim : xarray.DataArray
        dimension to replace it with
    drop : bool, optional
        whether to drop other coords. This is a good idea, because the other
        coords are probably not valid in the new dimension

    RETURNS
    -------
    da_new : xarray.DataArray
    """

    da_new = da.rename({olddim: newdim.name})
    # note that alignment along a dimension is skipped when you are overriding
    # the relevant coordinate values
    da_new .coords[newdim.name] = newdim
    da_new  = da_new.reset_coords(drop=drop)
    return da_new
