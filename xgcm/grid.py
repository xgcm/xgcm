from __future__ import print_function
from __future__ import absolute_import
from future.utils import iteritems
from collections import OrderedDict
import docrep
import xarray as xr
import numpy as np

from . import comodo
from .duck_array_ops import _pad_array

docstrings = docrep.DocstringProcessor(doc_key='My doc string')



class Axis:
    """
    An object that represents a group of coodinates that all lie along the same
    physical dimension but at different positions with respect to a grid cell.
    There are four possible possition::

         Center
         |------o-------|------o-------|------o-------|------o-------|
               [0]            [1]            [2]            [3]

         Left
         |------o-------|------o-------|------o-------|------o-------|
        [0]            [1]            [2]            [3]

         Right
         |------o-------|------o-------|------o-------|------o-------|
                       [0]            [1]            [2]            [3]

         Inner
         |------o-------|------o-------|------o-------|------o-------|
                       [0]            [1]            [2]

         Outer
         |------o-------|------o-------|------o-------|------o-------|
        [0]            [1]            [2]            [3]            [4]

    The `center` position is the only one without the `c_grid_axis_shift`
    attribute, which must be present for the other four. However, the actual
    value of `c_grid_axis_shift` is ignored for `inner` and `outer`, which are
    differentiated by their length.
    """

    def __init__(self, ds, axis_name, periodic=True, default_shifts={}):
        """
        Create a new Axis object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant grid information. Coordinate attributes
            should conform to Comodo conventions [1]_.
        axis_name : str
            The name of the axis (should match axis attribute)
        periodic : bool, optional
            Whether the domain is periodic along this axis
        default_shifts : dict, optional
            Default mapping from and to grid positions
            (e.g. `{'center': 'left'}`). Will be inferred if not specified.


        REFERENCES
        ----------
        .. [1] Comodo Conventions http://pycomodo.forge.imag.fr/norm.html
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
        # center, left, right, or outer
        coords = {name: ds[name] for name in coord_names}

        # some tortured logic for dealing with malforme c_grid_axis_shift
        # attributes such as produced by old versions of xmitgcm.
        # This should be a float (either -0.5 or 0.5)
        # this function returns that, or True of the attribute is set to
        # anything at all
        def _maybe_fix_type(attr):
            if attr is not None:
                try:
                    return float(attr)
                except TypeError:
                    return True

        axis_shift = {name:
                          _maybe_fix_type(coord.attrs.get('c_grid_axis_shift'))
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
        elif len(coords_without_axis_shift) > 1:
            raise ValueError("Found two coordinates without "
                             "`c_grid_axis_shift` attribute for axis %s"
                             % axis_name)
        center_coord_name = list(coords_without_axis_shift)[0]
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
            if clen == axis_len + 1:
                axis_coords['outer'] = coords[name]
            elif clen == axis_len - 1:
                axis_coords['inner'] = coords[name]
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
                raise ValueError("Coordinate %s has invalid or missing "
                                 "`c_grid_axis_shift` attribute `%s`" %
                                 (name, repr(shift)))

        self.coords = axis_coords

        # set default position shifts
        fallback_shifts = {'center': ('left', 'right', 'outer', 'inner'),
                           'left': ('center',), 'right': ('center',),
                           'outer': ('center',), 'inner': ('center',)}
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



    @docstrings.get_sectionsf('neighbor_binary_func')
    @docstrings.dedent
    def _neighbor_binary_func(self, da, f, to, boundary=None, fill_value=0.0):
        """
        Apply a function to neighboring points.

        Parameters
        ----------
        da : xarray.DataArray
            The data on which to operate
        f : function
            With signature f(da_left, da_right, shift)
        to : {'center', 'left', 'right', 'inner', 'outer'}
            The direction in which to shift the array. If not specified,
            default will be used.
        boundary : {None, 'fill', 'extend'}
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.)

        fill_value : float, optional
             The value to use in the boundary condition with `boundary='fill'`.

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        position_from, dim = self._get_axis_coord(da)
        if to is None:
            to = self._default_shifts[position_from]

        # get the two neighboring sets of raw data
        data_left, data_right = self._get_neighbor_data_pairs(da, to,
                                      boundary=boundary, fill_value=fill_value)
        # apply the function
        data_new = f(data_left, data_right)

        # wrap in a new xarray wrapper
        da_new = self._wrap_and_replace_coords(da, data_new, to)

        return da_new

    docstrings.delete_params('neighbor_binary_func.parameters', 'f')


    def _get_neighbor_data_pairs(self, da, position_to, boundary=None,
                                 fill_value=0.0):
        """Returns data_left, data_right."""

        position_from, dim = self._get_axis_coord(da)

        valid_positions = ['outer', 'inner', 'left', 'right', 'center']
        if position_to not in valid_positions:
            raise ValueError("`%s` is not a valid axis position" % position_to)

        if self._periodic and boundary:
            raise ValueError("`boundary=%s` is not allowed with periodic "
                             "axis %s." % (boundary, self._name))

        if position_from == position_to:
            raise ValueError("Can't get neighbor pairs for the same position.")

        transition = (position_from, position_to)

        if ((transition == ('outer', 'center')) or
            (transition == ('center', 'inner'))):
            # doesn't matter if domain is periodic or not
            left = da.isel(**{dim: slice(None, -1)}).data
            right = da.isel(**{dim: slice(1, None)}).data
        elif ((transition == ('center', 'outer')) or
              (transition == ('inner', 'center'))):
            # pad both sides of the array
            left = _pad_array(da, dim, left=True,
                              boundary=boundary, fill_value=fill_value)
            right = _pad_array(da, dim, boundary=boundary,
                               fill_value=fill_value)
        # TODO: figure out if it matters whether we slice the original array
        # before or after padding
        elif (not self._periodic and ((transition == ('center', 'left')) or
                                       (transition == ('right', 'center')))):
            # pad only left
            left = _pad_array(da.isel(**{dim: slice(0,-1)}), dim, left=True,
                              boundary=boundary, fill_value=fill_value)
            right = da.data
        elif (not self._periodic and ((transition == ('center', 'right')) or
                                      (transition == ('left', 'center')))):
            # pad only left
            right = _pad_array(da.isel(**{dim: slice(1,None)}), dim, boundary=boundary,
                               fill_value=fill_value)
            left = da.data
        elif (self._periodic and ((transition == ('center', 'left')) or
                                  (transition == ('right', 'center')))):
            left = da.roll(**{dim: 1}).data
            right = da.data
        elif (self._periodic and ((transition == ('center', 'right')) or
                                  (transition == ('left', 'center')))):
            left = da.data
            right = da.roll(**{dim: -1}).data
        else:
            is_periodic = 'periodic' if self._periodic else 'non-periodic'
            raise NotImplementedError(' to '.join(transition) +
                                      ' (%s) transition not yet supported.'
                                      % is_periodic)

        return left, right


    @docstrings.dedent
    def interp(self, da, to=None, boundary=None, fill_value=0.0):
        """
        Interpolate neighboring points to the intermediate grid point along
        this axis.

        Parameters
        ----------
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_i : xarray.DataArray
            The interpolated data

        """

        def interp_function(data_left, data_right):
            # linear, centered interpolation
            # TODO: generalize to higher order interpolation
            return 0.5*(data_left + data_right)
        return self._neighbor_binary_func(da, interp_function, to,
                                          boundary, fill_value)


    @docstrings.dedent
    def diff(self, da, to=None, boundary=None, fill_value=0.0):
        """
        Difference neighboring points to the intermediate grid point.

        Parameters
        ----------
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        def diff_function(data_left, data_right):
            return data_right - data_left
        return self._neighbor_binary_func(da, diff_function, to,
                                          boundary, fill_value)




    def _wrap_and_replace_coords(self, da, data_new, position_to):
        """
        Take the base coords from da, the data from data_new, and return
        a new DataArray with a coordinate on position_to.
        """
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



class Grid:
    """
    An object with multiple :class:`xgcm.Axis` objects representing different
    independent axes.
    """

    def __init__(self, ds, check_dims=True, periodic=True, default_shifts={}):
        """
        Create a new Grid object from an input dataset.

        Parameters
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
        default_shifts : dict
            A dictionary of dictionaries specifying default grid position
            shifts (e.g. `{'X': {'center': 'left', 'left': 'center'}}`)

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
            if axis_name in default_shifts:
                axis_default_shifts = default_shifts[axis_name]
            else:
                axis_default_shifts = {}
            self.axes[axis_name] = Axis(ds, axis_name, is_periodic,
                                        default_shifts=axis_default_shifts)


    def __repr__(self):
        summary = ['<xgcm.Grid>']
        for name, axis in iteritems(self.axes):
            is_periodic = 'periodic' if axis._periodic else 'not periodic'
            summary.append('%s Axis (%s):' % (name, is_periodic))
            summary += axis._coord_desc()
        return '\n'.join(summary)

    @docstrings.dedent
    def interp(self, da, axis, **kwargs):
        """
        Interpolate neighboring points to the intermediate grid point along
        this axis.

        Parameters
        ----------
        axis : str
            Name of the axis on which ot act
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_i : xarray.DataArray
            The interpolated data
        """

        ax = self.axes[axis]
        return ax.interp(da, **kwargs)

    @docstrings.dedent
    def diff(self, da, axis, **kwargs):
        """
        Difference neighboring points to the intermediate grid point.

        Parameters
        ----------
        axis : str
            Name of the axis on which ot act
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        ax = self.axes[axis]
        return ax.diff(da, **kwargs)



_other_docstring_options="""
    * 'dirichlet'
       The value of the array at the boundary point is specified by
       `fill_value`.
    * 'neumann'
       The value of the array diff at the boundary point is
       specified[1]_ by `fill_value`.

        .. [1] https://en.wikipedia.org/wiki/Dirichlet_boundary_condition
        .. [2] https://en.wikipedia.org/wiki/Neumann_boundary_condition
"""
