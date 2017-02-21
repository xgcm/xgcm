from __future__ import print_function
from future.utils import iteritems
from collections import OrderedDict
import pytest
import xarray as xr
import numpy as np

from . import comodo


class Grid:
    """An object that knows how to interpolate and take derivatives."""

    def __init__(self, ds, check_dims=True, x_periodic=True, y_periodic=True,
                 z_periodic=False):
        """Create a new Grid object from an input dataset.

        PARAMETERS
        ----------
        ds : xarray.Dataset
            Contains the relevant grid information. Coordinate attributes
            should conform to Comodo conventions [1]_.
        check_dims : bool, optional
            Whether to check the compatibility of input data dimensions before
            performing grid operations.
        x_periodic : bool, optional
            Whether the domain is periodic in the X direction.
        y_periodic : bool, optional
            Whether the domain is periodic in the Y direction.
        y_periodic : bool, optional
            Whether the domain is periodic in the Z direction.

        REFERENCES
        ----------
        .. [1] Comodo Conventions http://pycomodo.forge.imag.fr/norm.html
        """
        self._ds = ds
        self._check_dims = check_dims
        self._periodic = {'X': x_periodic, 'Y': y_periodic, 'Z': z_periodic}

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
                        # we found the face coordinate
                        axis_data['g'] = name
                        axis_data['g_coord'] = coord
                        # TODO: clearly document the sign convention
                        axis_data['shift'] = 1 if axis_shift==0.5 else -1
                    else:
                        raise ValueError('Invalid c_grid_axis_shift (%g) for '
                                         'coord %s' % (axis_shift, name))
                self._axes[ax] = axis_data

        # check grid size consistency
        # we can deal with two cases:
        #  * the c dim and g dim are the same size
        #  * the g dim is one element longer than the c dim
        # define a slice used to subset
        for ax, info in iteritems(self._axes):
            clen = len(info['c_coord'])
            glen = len(info['g_coord'])
            if clen==glen:
                # all good
                self._axes[ax]['pad'] = 0
            elif clen==(glen - 1):
                self._axes[ax]['pad'] = 1
            else:
                raise ValueError("Incompatible c and g dimension lengths on "
                                 "axis %s (%g, %g)" % (ax, clen, glen))



    def __repr__(self):
        summary = ['<xgcm.Grid>']

        for ax, info in iteritems(self._axes):
            axis_info = ('%s-axis:     %s: %g (cell center), %s: %g '
                         '(cell face, shift %g)' %
                         (ax, info['c'], len(info['c_coord']),
                          info['g'], len(info['g_coord']), info['shift']))
            if info['pad']:
                axis_info += ' padded,'
            if self._periodic[ax]:
                axis_info += ' periodic'
            else:
                axis_info += ' non-periodic'
            summary.append(axis_info)
        return '\n'.join(summary)


    def interp(self, da, axis):
        """Interpolate neighboring points to the intermediate grid point.

        PARAMETERS
        ----------
        da : xarray.DataArray
            The data to interpolate
        axis: {'X', 'Y'}
            The name of the axis along which to interpolate

        RETURNS
        -------
        da_i : xarray.DataArray
            The interpolated data
        """

        def interp_function(data_left, data_right, shift):
            # linear, centered interpolation
            # TODO: generalize to higher order interpolation
            return 0.5*(data_left + data_right)
        return self._neighbor_binary_func(da, axis, interp_function)


    def diff(self, da, axis):
        """Difference neighboring points to the intermediate grid point.

        PARAMETERS
        ----------
        da : xarray.DataArray
            The data to difference
        axis: {'X', 'Y'}
            The name of the axis along which to difference

        RETURNS
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        def interp_function(data_left, data_right, shift):
            # linear, centered interpolation
            # TODO: generalize to higher order interpolation
            return shift*(data_right - data_left)
        return self._neighbor_binary_func(da, axis, interp_function)


    def _neighbor_binary_func(self, da, axis, f):
        """Apply a function to neighboring points.

        PARAMETERS
        ----------
        da : xarray.DataArray
            The data to difference
        axis: {'X', 'Y'}
            The name of the axis along which to difference
        f : function
            With signature f(da_left, da_right, shift)

        RETURNS
        -------
        da_i : xarray.DataArray
            The differenced data
        """
        # figure out of it's a c or g variable
        ax = self._axes[axis]
        is_cgrid = ax['c'] in da.dims
        is_ggrid = ax['g'] in da.dims

        if is_cgrid:
            ax_name = ax['c']
            new_coord = ax['g_coord']
            shift = -ax['shift']

        elif is_ggrid:
            ax_name = ax['g']
            new_coord = ax['c_coord']
            shift = ax['shift']
        else:
            raise ValueError("Couldn't find an %s axis dimension in da" % axis)

        # shift data appropriately
        # if the grid is not periodic, we will discard the invalid points later
        da_shift = self.shift(da, ax_name, shift)

        data_new = f(da_shift.data, da.data, shift)

        # wrap in a new DataArray
        da_i = da.copy()
        da_i.data = data_new

        # we might need to truncate or pad the data
        if is_ggrid:
            if ax['pad']:
                # truncate
                if ax['shift']==1:
                    da_i = da_i.isel(**{ax_name: slice(1,None)})
                elif ax['shift']==-1:
                    da_i = da_i.isel(**{ax_name: slice(0,-1)})
            else:
                # deal with non-periodic case
                pass
        elif is_cgrid:
            # here the behavior depends on whether the data is periodic
            if ax['pad'] and self._periodic[axis]:
                raise NotImplementedError("Don't know how to pad periodic "
                                          "dims.")
            elif ax['pad'] and not self._periodic[axis]:
                # need to snip data from both sides
                new_coord = new_coord[1:-1]
                # and coordinate from one side
                if ax['shift']==1:
                    da_i = da_i.isel(**{ax_name: slice(0,-1)})
                elif ax['shift']==-1:
                    da_i = da_i.isel(**{ax_name: slice(1,None)})

        da_i = _replace_dim(da_i, ax_name, new_coord)
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
