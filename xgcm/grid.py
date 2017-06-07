from __future__ import print_function
from future.utils import iteritems
from collections import OrderedDict
import pytest
import xarray as xr
import numpy as np

from . import comodo
from .duck_array_ops import concatenate

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


class Axis:
    """An object that knows how to interpolate and take derivatives.
    There are four types of variable positions we can have::

         Centered
         |------o-------|------o-------|------o-------|------o-------|
               [0]            [1]            [2]            [3]

         Left face
         |------o-------|------o-------|------o-------|------o-------|
        [0]            [1]            [2]            [3]

         Right face
         |------o-------|------o-------|------o-------|------o-------|
                       [0]            [1]            [2]            [3]

         Face
         |------o-------|------o-------|------o-------|------o-------|
        [0]            [1]            [2]            [3]            [4]

    The first three have the same length and are thus distinguished by
    the `c_grid_axis_shift` attribute.
    """

    def __init__(self, ds, axis_name, periodic=True):
        """Create a new Axis object from an input dataset.

        PARAMETERS
        ----------
        ds : xarray.Dataset
            Contains the relevant grid information. Coordinate attributes
            should conform to Comodo conventions [1]_.
        axis_name : str
            The name of the axis (should match axis attribute)
        periodic : bool, optional
            Whether the domain is periodic along this axis
        """
        self._ds = ds
        self._name = axis_name
        self._periodic = periodic

        # figure out what the grid dimensions are
        coord_names = comodo.get_axis_coords(ds, axis_name)
        ncoords = len(coord_names)
        if ncoords == 0:
            # didn't find anything for this axis
            raise ValueError("Couldn't find any coordinates for axis %s"
                             % axis_name)

        # now figure out what type of coordinates these are:
        # center, left, right, or face
        coords = {name: ds[name] for name in coord_names}
        axis_shift = {name: coord.attrs.get('c_grid_axis_shift')
                      for name, coord in coords.items()}
        coord_len = {name: len(coord) for name, coord in coords.items()}

        # look for the center coord, which is required
        # this list will potential contain "center" and "face" points
        coords_without_axis_shift = {name: coord_len[name]
                                     for name, shift in axis_shift.items()
                                     if not shift}
        if len(coords_without_axis_shift) == 0:
            raise ValueError("Couldn't find a center coordinate for axis %s"
                             % axis_name)
        center_coord_name = min(coords_without_axis_shift,
                                key=coords_without_axis_shift.get)
        # knowing the length of the center coord is key to decoding the other
        # coords
        axis_len = coord_len[center_coord_name]

        # now we can start filling in the information about the different coords
        axis_coords = OrderedDict()
        axis_coords['center'] = coords[center_coord_name]

        # now check the other coords
        coord_names.remove(center_coord_name)
        for name in coord_names:
            shift = axis_shift[name]
            clen = coord_len[name]
            # face coordinate is characterized by the following property
            if clen == axis_len + 1:
                # we neglect the shift attribute completely here, since it is
                # irrelevant
                axis_coords['face'] = coords[name]
            elif shift == -0.5:
                if clen == axis_len:
                    axis_coords['left'] = coords[name]
                else:
                    raise ValueError("Left coordinate %s has incompatible "
                                     "length %g (axis_len=%g)"
                                     % (name, clen, axis_len))
            elif shift == 0.5:
                if clen == axis_len:
                    axis_coords['right'] = coords[name]
                else:
                    raise ValueError("Right coordinate %s has incompatible "
                                     "length %g (axis_len=%g)"
                                     % (name, clen, axis_len))
            else:
                raise ValueError("Coordinate %s has invalid c_grid_axis_shift "
                                 "attribute `%g`" % (name, shift))

        self.coords = axis_coords

    def __repr__(self):
        is_periodic = 'periodic' if self._periodic else 'not periodic'
        summary = ["<xgcm.Axis '%s' %s>" % (self._name, is_periodic)]
        summary.append('Axis Coodinates:')
        for name, coord in iteritems(self.coords):
            coord_info = ('  * %-8s %s (%g)' % (name, coord.name, len(coord)))
            summary.append(coord_info)
        return '\n'.join(summary)


    def interp(self, da, to=None):
        """Interpolate neighboring points to the intermediate grid point along
        this axis.

        PARAMETERS
        ----------
        da : xarray.DataArray
            The data to interpolate
        to : {'face', 'left', 'right', 'face'}, optional
            The grid position to which to interpolate. If not specified,
            defaults will be inferred.

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


    def diff(self, da, to=None):
        """Difference neighboring points to the intermediate grid point.

        PARAMETERS
        ----------
        da : xarray.DataArray
            The data to difference
        to : {'face', 'left', 'right', 'face'}, optional
            The grid position to which to interpolate. If not specified,
            defaults will be inferred.

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


    def _neighbor_binary_func(self, da, f, to):
        """Apply a function to neighboring points.

        PARAMETERS
        ----------
        da : xarray.DataArray
            The data to difference
        f : function
            With signature f(da_left, da_right, shift)
        to : {'face', 'left', 'right', 'face'}
            The grid position to which to interpolate. If not specified,
            defaults will be inferred.

        RETURNS
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        # get the two neighboring sets of raw data
        data_left, data_right = self._get_neighbor_data_pairs(da, to)
        # apply the function
        data_new = f(data_left, data_right)

        # wrap in a new xarray wrapper
        da_new = self._wrap_and_replace_coords(da, data_new, to)

        return da_new


    def _get_neighbor_data_pairs(self, da, position_to, boundary_cond=None):
        """Returns da_left, da_right."""
        position_from, dim = self._get_axis_coord(da)
        # different cases for different relationships between coordinates
        if position_from == position_to:
            raise ValueError("Can't get neighbor pairs for the same position.")

        transition = (position_from, position_to)

        if transition == ('face', 'center'):
            # doesn't matter if domain is periodic or not
            left = da.isel(**{dim: slice(None, -1)}).data
            right = da.isel(**{dim: slice(1, None)}).data
        elif (self._periodic and (transition == ('face', 'left') or
                                  (transition == ('right', 'face')))):
            left = da.roll(**{dim: 1}).data
            right = da.data
        elif (self._periodic and (transition == ('face', 'right') or
                                  (transition == ('left', 'face')))):
            left = da.data
            right = da.roll(**{dim: -1}).data
        else:
            is_periodic = 'periodic' if self._periodic else 'non-periodic'
            raise NotImplementedError(' to '.join(transition) +
                                      ' (%s) transition not yet supported.'
                                      % is_periodic)

    def _wrap_and_replace_coords(self, da, data_new, position_to):
        position_from, old_dim = self._get_axis_coord(da)
        new_coord = self.coords[position_to]
        new_dim = new_coord.name

        orig_dims = da.dims

        coords = OrderedDict()
        dims = []
        for d in orig_dims:
            if d == old_dim:
                dims.append[new_dim]
                coords[new_dim] = new_coord
            else:
                dims.append[d]
                coords[d] = da.coords[d]

        return xr.DataArray(data_new, dims=dims, coords=coords)


    def _get_axis_coord(self, da):
        """Return the position and name of the axis coordiante in a DataArray.
        """
        for position, coord in iteritems(self.coords):
            # TODO: should we have more careful checking of alignment here?
            if coord.name in da.dims:
                return position, coord.name


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
    da_new.coords[newdim.name] = newdim
    da_new  = da_new.reset_coords(drop=drop)
    return da_new
