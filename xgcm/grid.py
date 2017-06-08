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

    def __init__(self, ds, check_dims=True, periodic=True):
        """Create a new Grid object from an input dataset.

        PARAMETERS
        ----------
        ds : xarray.Dataset
            Contains the relevant grid information. Coordinate attributes
            should conform to Comodo conventions [1]_.
        check_dims : bool, optional
            Whether to check the compatibility of input data dimensions before
            performing grid operations.
        periodic : {True, False, list}
            Whether the grid is periodic (i.e. "wrap-around"). If a list is
            specified (e.g. `['X', 'Y']`), the axis names in the list will be
            be periodic and any other axes founds will be assumed non-periodic.

        REFERENCES
        ----------
        .. [1] Comodo Conventions http://pycomodo.forge.imag.fr/norm.html
        """
        self._ds = ds
        self._check_dims = check_dims

        all_axes = comodo.get_all_axes(ds)

        self.axes = OrderedDict()
        for axis_name in all_axes:
            try:
                is_periodic = axis_name in periodic
            except TypeError:
                is_periodic = periodic
            self.axes[axis_name] = Axis(ds, axis_name, is_periodic)


    def __repr__(self):
        summary = ['<xgcm.Grid>']
        for name, axis in iteritems(self.axes):
            is_periodic = 'periodic' if axis._periodic else 'not periodic'
            summary.append('%s Axis (%s):' % (name, is_periodic))
            summary += axis._coord_desc()
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

    def __init__(self, ds, axis_name, periodic=True, default_shifts={}):
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
        default_shifts : dict, optional
            Default mapping from and to grid positions (e.g. {'XC': 'XG'}).
            Will be inferred if not specified.
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

        # set default position shifts
        fallback_shifts = {'center': ('left', 'right', 'face'),
                           'left': ('center',), 'right': ('center',),
                           'face': ('center',)}
        self._default_shifts = {}
        for pos in axis_coords:
            # use user-specified value if present
            if pos in default_shifts:
                self._default_shifts[pos] = default_shifts[pos]
            else:
                for possible_shift in fallback_shifts[pos]:
                    if possible_shift in axis_coords:
                        self._default_shifts[pos] = possible_shift
                        break

    def __repr__(self):
        is_periodic = 'periodic' if self._periodic else 'not periodic'
        summary = ["<xgcm.Axis '%s' %s>" % (self._name, is_periodic)]
        summary.append('Axis Coodinates:')
        summary += self._coord_desc()
        return '\n'.join(summary)

    def _coord_desc(self):
        summary = []
        for name, coord in iteritems(self.coords):
            coord_info = ('  * %-8s %s (%g)' % (name, coord.name, len(coord)))
            if name in self._default_shifts:
                coord_info += ' --> %s' % self._default_shifts[name]
            summary.append(coord_info)
        return summary


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

        def interp_function(data_left, data_right):
            # linear, centered interpolation
            # TODO: generalize to higher order interpolation
            return 0.5*(data_left + data_right)
        return self._neighbor_binary_func(da, interp_function, to)


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

        def diff_function(data_left, data_right):
            return data_right - data_left
        return self._neighbor_binary_func(da, diff_function, to)


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
        """Returns data_left, data_right."""
        position_from, dim = self._get_axis_coord(da)
        # different cases for different relationships between coordinates
        if position_from == position_to:
            raise ValueError("Can't get neighbor pairs for the same position.")

        transition = (position_from, position_to)

        if transition == ('face', 'center'):
            # doesn't matter if domain is periodic or not
            left = da.isel(**{dim: slice(None, -1)}).data
            right = da.isel(**{dim: slice(1, None)}).data
        elif (self._periodic and (transition == ('center', 'left') or
                                  (transition == ('right', 'center')))):
            left = da.roll(**{dim: 1}).data
            right = da.data
        elif (self._periodic and (transition == ('center', 'right') or
                                  (transition == ('left', 'center')))):
            left = da.data
            right = da.roll(**{dim: -1}).data
        else:
            is_periodic = 'periodic' if self._periodic else 'non-periodic'
            raise NotImplementedError(' to '.join(transition) +
                                      ' (%s) transition not yet supported.'
                                      % is_periodic)

        return left, right

    def _wrap_and_replace_coords(self, da, data_new, position_to):
        """Take the base coords from da, the data from data_new, and return
        a new DataArray with a coordinate on position_to."""
        position_from, old_dim = self._get_axis_coord(da)
        try:
            new_coord = self.coords[position_to]
        except KeyError:
            raise KeyError("Position '%s' was not found in axis.coords."
                           % position_to)
        new_dim = new_coord.name

        orig_dims = da.dims

        coords = OrderedDict()
        dims = []
        for d in orig_dims:
            if d == old_dim:
                dims.append(new_dim)
                coords[new_dim] = new_coord
            else:
                dims.append(d)
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
