import functools
import itertools
import operator
import warnings
from collections import OrderedDict

import numpy as np
import xarray as xr

from . import comodo
from .duck_array_ops import _apply_boundary_condition, _pad_array, concatenate
from .metrics import iterate_axis_combinations

try:
    import numba

    from .transform import conservative_interpolation, linear_interpolation
except ImportError:
    numba = None


def _maybe_promote_str_to_list(a):
    # TODO: improve this
    if isinstance(a, str):
        return [a]
    else:
        return a


_VALID_BOUNDARY = [None, "fill", "extend", "extrapolate"]


class Axis:
    """
    An object that represents a group of coordinates that all lie along the same
    physical dimension but at different positions with respect to a grid cell.
    There are four possible positions:

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

    def __init__(
        self,
        ds,
        axis_name,
        periodic=True,
        default_shifts={},
        coords=None,
        boundary=None,
        fill_value=None,
    ):
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
        coords : dict, optional
            Mapping of axis positions to coordinate names
            (e.g. `{'center': 'XC', 'left: 'XG'}`)
        boundary : str or dict, optional,
            boundary can either be one of {None, 'fill', 'extend', 'extrapolate'}

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition where
              the difference at the boundary will be zero.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            This sets the default value. It can be overriden by specifying the
            boundary kwarg when calling specific methods.
        fill_value : float, optional
            The value to use in the boundary condition when `boundary='fill'`.

        REFERENCES
        ----------
        .. [1] Comodo Conventions https://web.archive.org/web/20160417032300/http://pycomodo.forge.imag.fr/norm.html
        """

        self._ds = ds
        self.name = axis_name
        self._periodic = periodic
        if boundary not in _VALID_BOUNDARY:
            raise ValueError(
                f"Expected 'boundary' to be one of {_VALID_BOUNDARY}. Received {boundary!r} instead."
            )
        self.boundary = boundary
        if fill_value is not None and not isinstance(fill_value, (int, float)):
            raise ValueError("Expected 'fill_value' to be a number.")
        self.fill_value = fill_value if fill_value is not None else 0.0

        if coords:
            # use specified coords
            self.coords = {pos: name for pos, name in coords.items()}
        else:
            # fall back on comodo conventions
            self.coords = comodo.get_axis_positions_and_coords(ds, axis_name)

        # self.coords is a dictionary with the following structure
        #   key: position_name {'center' ,'left' ,'right', 'outer', 'inner'}
        #   value: name of the dimension

        # set default position shifts
        fallback_shifts = {
            "center": ("left", "right", "outer", "inner"),
            "left": ("center",),
            "right": ("center",),
            "outer": ("center",),
            "inner": ("center",),
        }
        self._default_shifts = {}
        for pos in self.coords:
            # use user-specified value if present
            if pos in default_shifts:
                self._default_shifts[pos] = default_shifts[pos]
            else:
                for possible_shift in fallback_shifts[pos]:
                    if possible_shift in self.coords:
                        self._default_shifts[pos] = possible_shift
                        break

        ########################################################################
        # DEVELOPER DOCUMENTATION
        #
        # The attributes below are my best attempt to represent grid topology
        # in a general way. The data structures are complicated, but I can't
        # think of any way to simplify them.
        #
        # self._facedim (str) is the name of a dimension (e.g. 'face') or None.
        # If it is None, that means that the grid topology is _simple_, i.e.
        # that this is not a cubed-sphere grid or similar. For example:
        #
        #     ds.dims == ('time', 'lat', 'lon')
        #
        # If _facedim is set to a dimension name, that means that shifting
        # grid positions requires exchanging data among multiple "faces"
        # (a.k.a. "tiles", "facets", etc.). For this to work, there must be a
        # dimension corresponding to the different faces. This is `_facedim`.
        # For example:
        #
        #     ds.dims == ('time', 'face', 'lat', 'lon')
        #
        # In this case, `self._facedim == 'face'`
        #
        # We initialize all of this to None and let the `Grid` class handle
        # setting these attributes for complex geometries.
        self._facedim = None
        #
        # `self._connections` is a dictionary. It contains information about the
        # connectivity among this axis and other axes.
        # It should have the structure
        #
        #     {facedim_index: ((left_facedim_index, left_axis, left_reverse),
        #                      (right_facedim_index, right_axis, right_reverse)}
        #
        # `facedim_index` : a value used to index the `self._facedim` dimension
        #   (If `self._facedim` is `None`, then there should be only one key in
        #   `facedim_index` and that key should be `None`.)
        # `left_facedim_index` : the facedim index of the neighbor to the left.
        #   (If `self._facedim` is `None`, this must also be `None`.)
        # `left_axis` : an `Axis` object for the values to the left of this axis
        # `left_reverse` : bool, whether the connection should be reversed. By
        #   default, the left side of this axis will be connected to the right
        #   side of the neighboring axis. `left_reverse` overrides this and
        #   instead connects to the left side of the neighboring axis
        self._connections = {None: (None, None)}

        # now we implement periodic coordinates by setting appropriate
        # connections
        if periodic:
            self._connections = {None: ((None, self, False), (None, self, False))}

    def __repr__(self):
        is_periodic = "periodic" if self._periodic else "not periodic"
        summary = [
            "<xgcm.Axis '%s' (%s, boundary=%r)>"
            % (self.name, is_periodic, self.boundary)
        ]
        summary.append("Axis Coordinates:")
        summary += self._coord_desc()
        return "\n".join(summary)

    def _coord_desc(self):
        summary = []
        for name, cname in self.coords.items():
            coord_info = "  * %-8s %s" % (name, cname)
            if name in self._default_shifts:
                coord_info += " --> %s" % self._default_shifts[name]
            summary.append(coord_info)
        return summary

    def _neighbor_binary_func(
        self,
        da,
        f,
        to,
        boundary=None,
        fill_value=None,
        boundary_discontinuity=None,
        vector_partner=None,
        keep_coords=False,
    ):
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
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
        fill_value : float, optional
            The value to use in the boundary condition with `boundary='fill'`.
        vector_partner : dict, optional
            A single key (string), value (DataArray)
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data
        """
        position_from, dim = self._get_position_name(da)
        if to is None:
            to = self._default_shifts[position_from]

        if boundary is None:
            boundary = self.boundary
        if fill_value is None:
            fill_value = self.fill_value
        data_new = self._neighbor_binary_func_raw(
            da,
            f,
            to,
            boundary=boundary,
            fill_value=fill_value,
            boundary_discontinuity=boundary_discontinuity,
            vector_partner=vector_partner,
        )
        # wrap in a new xarray wrapper
        da_new = self._wrap_and_replace_coords(da, data_new, to, keep_coords)

        return da_new

    def _neighbor_binary_func_raw(
        self,
        da,
        f,
        to,
        boundary=None,
        fill_value=0.0,
        boundary_discontinuity=None,
        vector_partner=None,
        position_check=True,
    ):

        # get the two neighboring sets of raw data
        data_left, data_right = self._get_neighbor_data_pairs(
            da,
            to,
            boundary=boundary,
            fill_value=fill_value,
            boundary_discontinuity=boundary_discontinuity,
            vector_partner=vector_partner,
            position_check=position_check,
        )

        # apply the function
        data_new = f(data_left, data_right)

        return data_new

    def _get_edge_data(
        self,
        da,
        is_left_edge=True,
        boundary=None,
        fill_value=0.0,
        ignore_connections=False,
        vector_partner=None,
        boundary_discontinuity=None,
    ):
        """Get the appropriate edge data given axis connectivity and / or
        boundary conditions.
        """

        position, this_dim = self._get_position_name(da)
        this_axis_num = da.get_axis_num(this_dim)

        def face_edge_data(fnum, face_axis, count=1):
            # get the edge data for a single face

            # There will not necessarily be connection data for every face
            # for every axis. If there is no connection data, fnum will not
            # be a key for self._connections.

            if fnum in self._connections:
                # it should always be a len 2 tuple
                face_connection = self._connections[fnum][0 if is_left_edge else 1]
            else:
                face_connection = None

            if (face_connection is None) or ignore_connections:
                # no connection: use specified boundary condition instead
                if self._facedim:
                    da_face = da.isel(**{self._facedim: slice(fnum, fnum + 1)})
                else:
                    da_face = da
                return _apply_boundary_condition(
                    da_face,
                    this_dim,
                    is_left_edge,
                    boundary=boundary,
                    fill_value=fill_value,
                )

            neighbor_fnum, neighbor_axis, reverse = face_connection

            # check for consistency
            if face_axis is None:
                assert neighbor_fnum is None

            # Build up a slice that selects the correct edge region for a
            # given face. We work directly with variables rather than
            # DataArrays in the hopes of greater efficiency, avoiding
            # indexing / alignment

            # Start with getting all the data
            edge_slice = [slice(None)] * da.ndim
            if face_axis is not None:
                # get the neighbor face
                edge_slice[face_axis] = slice(neighbor_fnum, neighbor_fnum + 1)

            data = da
            # vector_partner is a one-entry dictionary
            # - key is an axis identifier (e.g. 'X')
            # - value is a DataArray
            if vector_partner:
                vector_partner_axis_name = next(iter(vector_partner))
                if neighbor_axis.name == vector_partner_axis_name:
                    data = vector_partner[vector_partner_axis_name]
                    if reverse:
                        raise NotImplementedError(
                            "Don't know how to handle "
                            "vectors with reversed "
                            "connections."
                        )
            # TODO: there is still lots to figure out here regarding vectors.
            # What we have currently works fine for vectors oriented normal
            # to the axis (e.g. interp and diff u along x axis)
            # It does NOT work for vectors tangent to the axis
            # (e.g. interp and diff v along x axis)
            # That is a pretty hard problem to solve, because rotating these
            # faces also mixes up left vs right position. The solution will be
            # quite involved and will probably require the edge points to be
            # populated.
            # I don't even know how to detect the fail case, let alone solve it.

            neighbor_edge_dim = neighbor_axis.coords[position]
            neighbor_edge_axis_num = data.get_axis_num(neighbor_edge_dim)
            if is_left_edge and not reverse:
                neighbor_edge_slice = slice(-count, None)
            else:
                neighbor_edge_slice = slice(None, count)
            edge_slice[neighbor_edge_axis_num] = neighbor_edge_slice

            # the orthogonal dimension need to be reoriented if we are
            # connected to the other axis. Is this because of some deep
            # topological principle?
            if neighbor_axis is not self:
                ortho_axis = da.get_axis_num(self.coords[position])
                ortho_slice = slice(None, None, -1)
                edge_slice[ortho_axis] = ortho_slice

            edge = data.variable[tuple(edge_slice)].data

            # the axis of the edge on THIS face is not necessarily the same
            # as the axis on the OTHER face
            if neighbor_axis is not self:
                edge = edge.swapaxes(neighbor_edge_axis_num, this_axis_num)

            return edge

        if self._facedim:
            face_axis_num = da.get_axis_num(self._facedim)
            arrays = [
                face_edge_data(fnum, face_axis_num) for fnum in da[self._facedim].values
            ]
            edge_data = concatenate(arrays, face_axis_num)
        else:
            edge_data = face_edge_data(None, None)
        if self._periodic:
            if boundary_discontinuity:
                if is_left_edge:
                    edge_data = edge_data - boundary_discontinuity
                elif not is_left_edge:
                    edge_data = edge_data + boundary_discontinuity
        return edge_data

    def _extend_left(
        self,
        da,
        boundary=None,
        fill_value=0.0,
        ignore_connections=False,
        vector_partner=None,
        boundary_discontinuity=None,
    ):

        axis_num = self._get_axis_dim_num(da)
        kw = dict(
            is_left_edge=True,
            boundary=boundary,
            fill_value=fill_value,
            ignore_connections=ignore_connections,
            vector_partner=vector_partner,
            boundary_discontinuity=boundary_discontinuity,
        )
        edge_data = self._get_edge_data(da, **kw)
        return concatenate([edge_data, da.data], axis=axis_num)

    def _extend_right(
        self,
        da,
        boundary=None,
        fill_value=0.0,
        ignore_connections=False,
        vector_partner=None,
        boundary_discontinuity=None,
    ):
        axis_num = self._get_axis_dim_num(da)
        kw = dict(
            is_left_edge=False,
            boundary=boundary,
            fill_value=fill_value,
            ignore_connections=ignore_connections,
            vector_partner=vector_partner,
            boundary_discontinuity=boundary_discontinuity,
        )
        edge_data = self._get_edge_data(da, **kw)
        return concatenate([da.data, edge_data], axis=axis_num)

    def _get_neighbor_data_pairs(
        self,
        da,
        position_to,
        boundary=None,
        fill_value=0.0,
        ignore_connections=False,
        boundary_discontinuity=None,
        vector_partner=None,
        position_check=True,
    ):

        position_from, dim = self._get_position_name(da)
        axis_num = da.get_axis_num(dim)

        boundary_kwargs = dict(
            boundary=boundary,
            fill_value=fill_value,
            ignore_connections=ignore_connections,
            vector_partner=vector_partner,
            boundary_discontinuity=boundary_discontinuity,
        )

        valid_positions = ["outer", "inner", "left", "right", "center"]

        if position_to == position_from:
            raise ValueError("Can't get neighbor pairs for the same position.")

        if position_to not in valid_positions:
            raise ValueError(
                "`%s` is not a valid axis position name. Valid "
                "names are %r." % (position_to, valid_positions)
            )

        # This prevents the grid generation to work, I added an optional
        # kwarg that deactivates this check
        # (only set False from autogenerate/generate_grid_ds)

        if position_check:
            if position_to not in self.coords:
                raise ValueError(
                    "This axis doesn't contain a `%s` position" % position_to
                )

        transition = (position_from, position_to)

        if (transition == ("outer", "center")) or (transition == ("center", "inner")):
            # don't need any edge values
            left = da.isel(**{dim: slice(None, -1)}).data
            right = da.isel(**{dim: slice(1, None)}).data
        elif (transition == ("center", "outer")) or (transition == ("inner", "center")):
            # need both edge values
            left = self._extend_left(da, **boundary_kwargs)
            right = self._extend_right(da, **boundary_kwargs)
        elif transition == ("center", "left") or transition == ("right", "center"):
            # need to slice *after* getting edge because otherwise we could
            # mess up complicated connections (e.g. cubed-sphere)
            left = self._extend_left(da, **boundary_kwargs)
            # unfortunately left is not an xarray so we have to slice
            # it the long numpy way
            slc = axis_num * (slice(None),) + (slice(0, -1),)
            left = left[slc]
            right = da.data
        elif transition == ("center", "right") or transition == ("left", "center"):
            # need to slice *after* getting edge because otherwise we could
            # mess up complicated connections (e.g. cubed-sphere)
            right = self._extend_right(da, **boundary_kwargs)
            # unfortunately left is not an xarray so we have to slice
            # it the long numpy way
            slc = axis_num * (slice(None),) + (slice(1, None),)
            right = right[slc]
            left = da.data
        else:
            raise NotImplementedError(
                " to ".join(transition) + " transition not yet supported."
            )

        return left, right

    def interp(
        self,
        da,
        to=None,
        boundary=None,
        fill_value=None,
        boundary_discontinuity=None,
        vector_partner=None,
        keep_coords=False,
    ):
        """
        Interpolate neighboring points to the intermediate grid point along
        this axis.

        Parameters
        ----------
        da : xarray.DataArray
            The data on which to operate
        to : {'center', 'left', 'right', 'inner', 'outer'}
            The direction in which to shift the array. If not specified,
            default will be used.
        boundary : {None, 'fill', 'extend'}
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
        fill_value : float, optional
            The value to use in the boundary condition with `boundary='fill'`.
        vector_partner : dict, optional
            A single key (string), value (DataArray)
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.

        Returns
        -------
        da_i : xarray.DataArray
            The interpolated data

        """

        return self._neighbor_binary_func(
            da,
            raw_interp_function,
            to,
            boundary,
            fill_value,
            boundary_discontinuity,
            vector_partner,
            keep_coords=keep_coords,
        )

    def diff(
        self,
        da,
        to=None,
        boundary=None,
        fill_value=None,
        boundary_discontinuity=None,
        vector_partner=None,
        keep_coords=False,
    ):
        """
        Difference neighboring points to the intermediate grid point.

        Parameters
        ----------
        da : xarray.DataArray
            The data on which to operate
        to : {'center', 'left', 'right', 'inner', 'outer'}
            The direction in which to shift the array. If not specified,
            default will be used.
        boundary : {None, 'fill', 'extend'}
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
        fill_value : float, optional
            The value to use in the boundary condition with `boundary='fill'`.
        vector_partner : dict, optional
            A single key (string), value (DataArray)
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        return self._neighbor_binary_func(
            da,
            raw_diff_function,
            to,
            boundary,
            fill_value,
            boundary_discontinuity,
            vector_partner,
            keep_coords=keep_coords,
        )

    def cumsum(self, da, to=None, boundary=None, fill_value=0.0, keep_coords=False):
        """
        Cumulatively sum a DataArray, transforming to the intermediate axis
        position.

        Parameters
        ----------
        da : xarray.DataArray
            The data on which to operate
        to : {'center', 'left', 'right', 'inner', 'outer'}
            The direction in which to shift the array. If not specified,
            default will be used.
        boundary : {None, 'fill', 'extend'}
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
        fill_value : float, optional
            The value to use in the boundary condition with `boundary='fill'`.
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.

        Returns
        -------
        da_cum : xarray.DataArray
            The cumsummed data
        """

        pos, dim = self._get_position_name(da)

        if to is None:
            to = self._default_shifts[pos]

        # first use xarray's cumsum method
        da_cum = da.cumsum(dim=dim)

        boundary_kwargs = dict(boundary=boundary, fill_value=fill_value)

        # now pad / trim the data as necessary
        # here we enumerate all the valid possible shifts
        if (pos == "center" and to == "right") or (pos == "left" and to == "center"):
            # do nothing, this is the default for how cumsum works
            data = da_cum.data
        elif (pos == "center" and to == "left") or (pos == "right" and to == "center"):
            data = _pad_array(
                da_cum.isel(**{dim: slice(0, -1)}), dim, left=True, **boundary_kwargs
            )
        elif (pos == "center" and to == "inner") or (pos == "outer" and to == "center"):
            data = da_cum.isel(**{dim: slice(0, -1)}).data
        elif (pos == "center" and to == "outer") or (pos == "inner" and to == "center"):
            data = _pad_array(da_cum, dim, left=True, **boundary_kwargs)
        else:
            raise ValueError(
                "From `%s` to `%s` is not a valid position "
                "shift for cumsum operation." % (pos, to)
            )

        da_cum_newcoord = self._wrap_and_replace_coords(da, data, to, keep_coords)
        return da_cum_newcoord

    def min(
        self,
        da,
        to=None,
        boundary=None,
        fill_value=None,
        boundary_discontinuity=None,
        vector_partner=None,
        keep_coords=False,
    ):
        """
        Minimum of neighboring points on intermediate grid point.

        Parameters
        ----------
        da : xarray.DataArray
            The data on which to operate
        to : {'center', 'left', 'right', 'inner', 'outer'}
            The direction in which to shift the array. If not specified,
            default will be used.
        boundary : {None, 'fill', 'extend'}
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
        fill_value : float, optional
            The value to use in the boundary condition with `boundary='fill'`.
        vector_partner : dict, optional
            A single key (string), value (DataArray)
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        return self._neighbor_binary_func(
            da,
            raw_min_function,
            to,
            boundary,
            fill_value,
            boundary_discontinuity,
            vector_partner,
            keep_coords,
        )

    def max(
        self,
        da,
        to=None,
        boundary=None,
        fill_value=None,
        boundary_discontinuity=None,
        vector_partner=None,
        keep_coords=False,
    ):
        """
        Maximum of neighboring points on intermediate grid point.

        Parameters
        ----------
        da : xarray.DataArray
            The data on which to operate
        to : {'center', 'left', 'right', 'inner', 'outer'}
            The direction in which to shift the array. If not specified,
            default will be used.
        boundary : {None, 'fill', 'extend'}
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
        fill_value : float, optional
            The value to use in the boundary condition with `boundary='fill'`.
        vector_partner : dict, optional
            A single key (string), value (DataArray)
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        return self._neighbor_binary_func(
            da,
            raw_max_function,
            to,
            boundary,
            fill_value,
            boundary_discontinuity,
            vector_partner,
            keep_coords,
        )

    def transform(
        self,
        da,
        target,
        target_data=None,
        method="linear",
        mask_edges=True,
        bypass_checks=False,
        suffix="_transformed",
    ):
        """Convert an array of data to new 1D-coordinates.
        The method takes a multidimensional array of data `da` and
        transforms it onto another data_array `target_data` in the
        direction of the axis (for each 1-dimensional 'column').

        `target_data` can be e.g. the existing coordinate along an
        axis, like depth. xgcm automatically detects the appropriate
        coordinate and then transforms the data from the input
        positions to the desired positions defined in `target`. This
        is the default behavior. The method can also be used for more
        complex cases like transforming a dataarray into new
        coordinates that are defined by e.g. a tracer field like
        temperature, density, etc.

        Currently two methods are supported to carry out the
        transformation:

        - 'linear': Values are linear interpolated between 1D columns
          along `axis` of `da` and `target_data`. This method requires
          `target_data` to increase/decrease monotonically. `target`
          values are interpreted as new cell centers in this case. By
          default this method will return nan for values in `target` that
          are outside of the range of `target_data`, setting
          `mask_edges=False` results in the default np.interp behavior of
          repeated values.

        - 'conservative': Values are transformed while conserving the
          integral of `da` along each 1D column. This method can be used
          with non-monotonic values of `target_data`. Currently this will
          only work with extensive quantities (like heat, mass, transport)
          but not with intensive quantities (like temperature, density,
          velocity). N given `target` values are interpreted as cell-bounds
          and the returned array will have N-1 elements along the newly
          created coordinate, with coordinate values that are interpolated
          between `target` values.

        Parameters
        ----------
        da : xr.xr.DataArray
            Input data
        target : {np.array, xr.DataArray}
            Target points for transformation. Depending on the method is
            interpreted as cell center (method='linear') or cell bounds
            (method='conservative).
            Values correpond to `target_data` or the existing coordinate
            along the axis (if `target_data=None`). The name of the
            resulting new coordinate is determined by the input type.
            When passed as numpy array the resulting dimension is named
            according to `target_data`, if provided as xr.Dataarray
            naming is inferred from the `target` input.
        target_data : xr.DataArray, optional
            Data to transform onto (e.g. a tracer like density or temperature).
            Defaults to None, which infers the appropriate coordinate along
            `axis` (e.g. the depth).
        method : str, optional
            Method used to transform, by default "linear"
        mask_edges : bool, optional
            If activated, `target` values outside the range of `target_data`
            are masked with nan, by default True. Only applies to 'linear' method.
        bypass_checks : bool, optional
            Only applies for `method='linear'`.
            Option to bypass logic to flip data if monotonically decreasing along the axis.
            This will improve performance if True, but the user needs to ensure that values
            are increasing along the axis.
        suffix : str, optional
            Customizable suffix to the name of the output array. This will
            be added to the original name of `da`. Defaults to `_transformed`.

        Returns
        -------
        xr.DataArray
            The transformed data


        """
        # Theoretically we should be able to use a multidimensional `target`, which would need the additional information provided with `target_dim`.
        # But the feature is not tested yet, thus setting this to default value internally (resulting in error in `_parse_target`, when a multidim `target` is passed)
        target_dim = None

        # check optional numba dependency
        if numba is None:
            raise ImportError(
                "The transform functionality of xgcm requires numba. Install using `conda install numba`."
            )

        # raise error if axis is periodic
        if self._periodic:
            raise ValueError(
                "`transform` can only be used on axes that are non-periodic. Pass `periodic=False` to `xgcm.Grid`."
            )

        # raise error if the target values are not provided as xr.dataarray
        for var_name, variable, allowed_types in [
            ("da", da, [xr.DataArray]),
            ("target", target, [xr.DataArray, np.ndarray]),
            ("target_data", target_data, [xr.DataArray]),
        ]:
            if not (isinstance(variable, tuple(allowed_types)) or variable is None):
                raise ValueError(
                    f"`{var_name}` needs to be a {' or '.join([str(a) for a in allowed_types])}. Found {type(variable)}"
                )

        def _target_data_name_handling(target_data):
            """Handle target_data input without a name"""
            if target_data.name is None:
                warnings.warn(
                    "Input`target_data` has no name, but we need a name for the transformed dimension. The name `TRANSFORMED_DIMENSION` will be used. To avoid this warning, call `.rename` on `target_data` before calling `transform`."
                )
                target_data.name = "TRANSFORMED_DIMENSION"

        def _check_other_dims(target_da):
            # check if other dimensions (excluding ones associated with the transform axis) are the
            # same between `da` and `target_data`. If not provide instructions how to work around.

            da_other_dims = set(da.dims) - set(self.coords.values())
            target_da_other_dims = set(target_da.dims) - set(self.coords.values())
            if not target_da_other_dims.issubset(da_other_dims):
                raise ValueError(
                    f"Found additional dimensions [{target_da_other_dims-da_other_dims}]"
                    "in `target_data` not found in `da`. This could mean that the target "
                    "array is not on the same position along other axes."
                    " If the additional dimensions are associated witha staggered axis, "
                    "use grid.interp() to move values to other grid position. "
                    "If additional dimensions are not related to the grid (e.g. climate "
                    "model ensemble members or similar), use xr.broadcast() before using transform."
                )

        def _parse_target(target, target_dim, target_data_dim, target_data):
            """Parse target values into correct xarray naming and set default naming based on input data"""
            # if target_data is not provided, assume the target to be one of the staggered dataset dimensions.
            if target_data is None:
                target_data = self._ds[target_data_dim]

            # Infer target_dim from target
            if isinstance(target, xr.DataArray):
                if len(target.dims) == 1:
                    if target_dim is None:
                        target_dim = list(target.dims)[0]
                else:
                    if target_dim is None:
                        raise ValueError(
                            f"Cant infer `target_dim` from `target` since it has more than 1 dimension [{target.dims}]. This is currently not supported. `."
                        )
            else:
                # if the target is not provided as xr.Dataarray we take the name of the target_data as new dimension name
                _target_data_name_handling(target_data)
                target_dim = target_data.name
                target = xr.DataArray(
                    target, dims=[target_dim], coords={target_dim: target}
                )

            _check_other_dims(target_data)
            return target, target_dim, target_data

        _, dim = self._get_position_name(da)
        if method == "linear":
            target, target_dim, target_data = _parse_target(
                target, target_dim, dim, target_data
            )
            out = linear_interpolation(
                da,
                target_data,
                target,
                dim,
                dim,  # in this case the dimension of phi and theta are the same
                target_dim,
                mask_edges=mask_edges,
                bypass_checks=bypass_checks,
            )
        elif method == "conservative":
            # the conservative method requires `target_data` to be on the `outer` coordinate.
            # If that is not the case (a very common use case like transformation on any tracer),
            # we need to infer the boundary values (using the interp logic)
            # for this method we need the `outer` position. Error out if its not there.
            try:
                target_data_dim = self.coords["outer"]
            except KeyError:
                raise RuntimeError(
                    "In order to use the method `conservative` the grid object needs to have `outer` coordinates."
                )

            target, target_dim, target_data = _parse_target(
                target, target_dim, target_data_dim, target_data
            )

            # check on which coordinate `target_data` is, and interpolate if needed
            if target_data_dim not in target_data.dims:
                warnings.warn(
                    "The `target data` input is not located on the cell bounds. This method will continue with linear interpolation with repeated boundary values. For most accurate results provide values on cell bounds.",
                    UserWarning,
                )
                target_data = self.interp(target_data, boundary="extend")
                # This seems to end up with chunks along the axis dimension.
                # Rechunk to keep xr.apply_func from complaining.
                # TODO: This should be made obsolete, when the internals are refactored using numba
                target_data = target_data.chunk(
                    {self._get_position_name(target_data)[1]: -1}
                )

            out = conservative_interpolation(
                da,
                target_data,
                target,
                dim,
                target_data_dim,  # in this case the dimension of phi and theta are the same
                target_dim,
            )

        return out

    def _wrap_and_replace_coords(self, da, data_new, position_to, keep_coords=False):
        """
        Take the base coords from da, the data from data_new, and return
        a new DataArray with a coordinate on position_to.
        """

        if not keep_coords:
            warnings.warn(
                "The keep_coords keyword argument is being deprecated - in future it will be removed "
                "entirely, and the behaviour will always be that currently given by keep_coords=True.",
                category=DeprecationWarning,
            )

        position_from, old_dim = self._get_position_name(da)
        try:
            new_dim = self.coords[position_to]
        except KeyError:
            raise KeyError("Position '%s' was not found in axis.coords." % position_to)

        orig_dims = da.dims

        coords = OrderedDict()
        dims = []
        for d in orig_dims:
            if d == old_dim:
                dims.append(new_dim)
                # only add coordinate if it actually exists
                # otherwise this creates a new coordinate where before there
                # was none
                if new_dim in self._ds.coords:
                    coords[new_dim] = self._ds.coords[new_dim]
            else:
                dims.append(d)
                # only add coordinate if it actually exists...
                if d in da.coords:
                    coords[d] = da.coords[d]

        # add compatible coords
        if keep_coords:
            for c in da.coords:
                if c not in coords and set(da[c].dims).issubset(dims):
                    coords[c] = da[c]

        return xr.DataArray(data_new, dims=dims, coords=coords)

    def _get_position_name(self, da):
        """Return the position and name of the axis coordinate in a DataArray."""
        for position, coord_name in self.coords.items():
            # TODO: should we have more careful checking of alignment here?
            if coord_name in da.dims:
                return position, coord_name

        raise KeyError(
            "None of the DataArray's dims %s were found in axis "
            "coords." % repr(da.dims)
        )

    def _get_axis_dim_num(self, da):
        """Return the dimension number of the axis coordinate in a DataArray."""
        _, coord_name = self._get_position_name(da)
        return da.get_axis_num(coord_name)


class Grid:
    """
    An object with multiple :class:`xgcm.Axis` objects representing different
    independent axes.
    """

    def __init__(
        self,
        ds,
        check_dims=True,
        periodic=True,
        default_shifts={},
        face_connections=None,
        coords=None,
        metrics=None,
        boundary=None,
        fill_value=None,
    ):
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
            specified (e.g. ``['X', 'Y']``), the axis names in the list will be
            be periodic and any other axes founds will be assumed non-periodic.
        default_shifts : dict
            A dictionary of dictionaries specifying default grid position
            shifts (e.g. ``{'X': {'center': 'left', 'left': 'center'}}``)
        face_connections : dict
            Grid topology
        coords : dict, optional
            Specifies positions of dimension names along axes X, Y, Z, e.g
            ``{'X': {'center': 'XC', 'left: 'XG'}}``.
            Each key should be an axis name (e.g., `X`, `Y`, or `Z`) and map
            to a dictionary which maps positions (`center`, `left`, `right`,
            `outer`, `inner`) to dimension names in the dataset
            (in the example above, `XC` is at the `center` position and `XG`
            at the `left` position along the `X` axis).
            If the values are not present in ``ds`` or are not dimensions,
            an error will be raised.
        metrics : dict, optional
            Specification of grid metrics mapping axis names (X, Y, Z) to corresponding
            metric variable names in the dataset
            (e.g. {('X',):['dx_t'], ('X', 'Y'):['area_tracer', 'area_u']}
            for the cell distance in the x-direction ``dx_t`` and the
            horizontal cell areas ``area_tracer`` and ``area_u``, located at
            different grid positions).
        boundary : {None, 'fill', 'extend', 'extrapolate', dict}, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            Optionally a dict mapping axis name to seperate values for each axis
            can be passed.
        fill_value : {float, dict}, optional
            The value to use in boundary conditions with `boundary='fill'`.
            Optionally a dict mapping axis name to seperate values for each axis
            can be passed.
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.

        REFERENCES
        ----------
        .. [1] Comodo Conventions https://web.archive.org/web/20160417032300/http://pycomodo.forge.imag.fr/norm.html
        """
        self._ds = ds
        self._check_dims = check_dims

        # Deprecation Warnigns
        warnings.warn(
            "The `xgcm.Axis` class will be deprecated in the future. "
            "Please make sure to use the `xgcm.Grid` methods for your work instead.",
            category=DeprecationWarning,
        )
        # This will show up every time, but I think that is fine

        if boundary:
            warnings.warn(
                "The `boundary` argument will be renamed "
                "to `padding` to better reflect the process "
                "of array padding and avoid confusion with "
                "physical boundary conditions (e.g. ocean land boundary).",
                category=DeprecationWarning,
            )

        # Deprecation Warnigns
        if periodic:
            warnings.warn(
                "The `periodic` argument will be deprecated. "
                "To preserve previous behavior supply `boundary = 'periodic'.",
                category=DeprecationWarning,
            )

        if fill_value:
            warnings.warn(
                "The default fill_value will be changed to nan (from 0.0 previously) "
                "in future versions. Provide `fill_value=0.0` to preserve previous behavior.",
                category=DeprecationWarning,
            )

        extrapolate_warning = False
        if boundary == "extrapolate":
            extrapolate_warning = True
        if isinstance(boundary, dict):
            if any([k == "extrapolate" for k in boundary.keys()]):
                extrapolate_warning = True
        if extrapolate_warning:
            warnings.warn(
                "The `boundary='extrapolate'` option will no longer be supported in future releases.",
                category=DeprecationWarning,
            )

        if coords:
            all_axes = coords.keys()
        else:
            all_axes = comodo.get_all_axes(ds)
            coords = {}

        # check coords input validity
        for axis, positions in coords.items():
            for pos, dim in positions.items():
                if not (dim in ds.variables or dim in ds.dims):
                    raise ValueError(
                        f"Could not find dimension `{dim}` (for the `{pos}` position on axis `{axis}`) in input dataset."
                    )
                if dim not in ds.dims:
                    raise ValueError(
                        f"Input `{dim}` (for the `{pos}` position on axis `{axis}`) is not a dimension in the input datasets `ds`."
                    )

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

            if isinstance(boundary, dict):
                axis_boundary = boundary.get(axis_name, None)
            elif isinstance(boundary, str) or boundary is None:
                axis_boundary = boundary
            else:
                raise ValueError(
                    f"boundary={boundary} is invalid. Please specify a dictionary "
                    "mapping axis name to a boundary option; a string or None."
                )

            if isinstance(fill_value, dict):
                axis_fillvalue = fill_value.get(axis_name, None)
            elif isinstance(fill_value, (int, float)) or fill_value is None:
                axis_fillvalue = fill_value
            else:
                raise ValueError(
                    f"fill_value={fill_value} is invalid. Please specify a dictionary "
                    "mapping axis name to a boundary option; a number or None."
                )

            self.axes[axis_name] = Axis(
                ds,
                axis_name,
                is_periodic,
                default_shifts=axis_default_shifts,
                coords=coords.get(axis_name),
                boundary=axis_boundary,
                fill_value=axis_fillvalue,
            )

        if face_connections is not None:
            self._assign_face_connections(face_connections)

        self._metrics = {}

        if metrics is not None:
            for key, value in metrics.items():
                self.set_metrics(key, value)

    def _parse_axes_kwargs(self, kwargs):
        """Convvert kwarg input into dict for each available axis
        E.g. for a grid with 2 axes for the keyword argument `periodic`
        periodic = True --> periodic = {'X': True, 'Y':True}
        or if not all axes are provided, the other axes will be parsed as defaults (None)
        periodic = {'X':True} --> periodic={'X': True, 'Y':None}
        """
        parsed_kwargs = dict()
        if isinstance(kwargs, dict):
            parsed_kwargs = kwargs
        else:
            for axis in self.axes:
                parsed_kwargs[axis] = kwargs
        return parsed_kwargs

    def _assign_face_connections(self, fc):
        """Check a dictionary of face connections to make sure all the links are
        consistent.
        """

        if len(fc) > 1:
            raise ValueError(
                "Only one face dimension is supported for now. "
                "Instead found %r" % repr(fc.keys())
            )

        # we will populate this with the axes we find in face_connections
        axis_connections = {}

        facedim = list(fc.keys())[0]
        assert facedim in self._ds

        face_links = fc[facedim]
        for fidx, face_axis_links in face_links.items():
            for axis, axis_links in face_axis_links.items():
                # initialize the axis dict if necssary
                if axis not in axis_connections:
                    axis_connections[axis] = {}
                link_left, link_right = axis_links

                def check_neighbor(link, position):
                    if link is None:
                        return
                    idx, ax, rev = link
                    # need to swap position if the link is reversed
                    correct_position = int(not position) if rev else position
                    try:
                        neighbor_link = face_links[idx][ax][correct_position]
                    except (KeyError, IndexError):
                        raise KeyError(
                            "Couldn't find a face link for face %r"
                            "in axis %r at position %r" % (idx, ax, correct_position)
                        )
                    idx_n, ax_n, rev_n = neighbor_link
                    if ax not in self.axes:
                        raise KeyError("axis %r is not a valid axis" % ax)
                    if ax_n not in self.axes:
                        raise KeyError("axis %r is not a valid axis" % ax_n)
                    if idx not in self._ds[facedim].values:
                        raise IndexError(
                            "%r is not a valid index for face"
                            "dimension %r" % (idx, facedim)
                        )
                    if idx_n not in self._ds[facedim].values:
                        raise IndexError(
                            "%r is not a valid index for face"
                            "dimension %r" % (idx, facedim)
                        )
                    # check for consistent links from / to neighbor
                    if (idx_n != fidx) or (ax_n != axis) or (rev_n != rev):
                        raise ValueError(
                            "Face link mismatch: neighbor doesn't"
                            " correctly link back to this face. "
                            "face: %r, axis: %r, position: %r, "
                            "rev: %r, link: %r, neighbor_link: %r"
                            % (fidx, axis, position, rev, link, neighbor_link)
                        )
                    # convert the axis name to an acutal axis object
                    actual_axis = self.axes[ax]
                    return idx, actual_axis, rev

                left = check_neighbor(link_left, 1)
                right = check_neighbor(link_right, 0)
                axis_connections[axis][fidx] = (left, right)

        for axis, axis_links in axis_connections.items():
            self.axes[axis]._facedim = facedim
            self.axes[axis]._connections = axis_links

    def set_metrics(self, key, value, overwrite=False):
        metric_axes = frozenset(_maybe_promote_str_to_list(key))
        axes_not_found = [ma for ma in metric_axes if ma not in self.axes]
        if len(axes_not_found) > 0:
            raise KeyError(
                f"Metric axes {axes_not_found!r} not compatible with grid axes {tuple(self.axes)!r}"
            )

        metric_value = _maybe_promote_str_to_list(value)
        for metric_varname in metric_value:
            if metric_varname not in self._ds.variables:
                raise KeyError(
                    f"Metric variable {metric_varname} not found in dataset."
                )

        existing_metric_axes = set(self._metrics.keys())
        if metric_axes in existing_metric_axes:
            value_exist = self._metrics.get(metric_axes)
            # resetting coords avoids potential broadcasting / alignment issues
            value_new = self._ds[metric_varname].reset_coords(drop=True)
            did_overwrite = False
            # go through each existing value until data array with matching dimensions is selected
            for idx, ve in enumerate(value_exist):
                # double check if dimensions match
                if set(value_new.dims) == set(ve.dims):
                    if overwrite:
                        # replace existing data array with new data array input
                        self._metrics[metric_axes][idx] = value_new
                        did_overwrite = True
                    else:
                        raise ValueError(
                            f"Metric variable {ve.name} with dimensions {ve.dims} already assigned in metrics."
                            f" Overwrite {ve.name} with {metric_varname} by setting overwrite=True."
                        )
            # if no existing value matches new value dimension-wise, just append new value
            if not did_overwrite:
                self._metrics[metric_axes].append(value_new)
        else:
            # no existing metrics for metric_axes yet; initialize empty list
            self._metrics[metric_axes] = []
            for metric_varname in metric_value:
                metric_var = self._ds[metric_varname].reset_coords(drop=True)
                self._metrics[metric_axes].append(metric_var)

    def _get_dims_from_axis(self, da, axis):
        dim = []
        axis = _maybe_promote_str_to_list(axis)
        for ax in axis:
            if ax in self.axes:
                all_dim = self.axes[ax].coords.values()
                matching_dim = [di for di in all_dim if di in da.dims]
                if len(matching_dim) == 1:
                    dim.append(matching_dim[0])
                else:
                    raise ValueError(
                        f"Did not find single matching dimension {da.dims} from {da.name} corresponding to axis {ax}, got {matching_dim}."
                    )
            else:
                raise KeyError(f"Did not find axis {ax} from data array {da.name}")
        return dim

    def get_metric(self, array, axes):
        """
        Find the metric variable associated with a set of axes for a particular
        array.

        Parameters
        ----------
        array : xarray.DataArray
            The array for which we are looking for a metric. Only its dimensions are considered.
        axes : iterable
            A list of axes for which to find the metric.

        Returns
        -------
        metric : xarray.DataArray
            A metric which can broadcast against ``array``
        """

        metric_vars = None
        array_dims = set(array.dims)

        # Will raise a Value Error if array doesn't have a dimension corresponding to metric axes specified
        # See _get_dims_from_axis
        self._get_dims_from_axis(array, frozenset(axes))

        possible_metric_vars = set(tuple(k) for k in self._metrics.keys())
        possible_combos = set(itertools.permutations(tuple(axes)))
        overlap_metrics = possible_metric_vars.intersection(possible_combos)

        if len(overlap_metrics) > 0:
            # Condition 1: metric with matching axes and dimensions exist
            overlap_metrics = frozenset(*overlap_metrics)
            possible_metrics = self._metrics[overlap_metrics]
            for mv in possible_metrics:
                metric_dims = set(mv.dims)
                if metric_dims.issubset(array_dims):
                    metric_vars = mv
                    break
            if metric_vars is None:
                # Condition 2: interpolate metric with matching axis to desired dimensions
                warnings.warn(
                    f"Metric at {array.dims} being interpolated from metrics at dimensions {mv.dims}. Boundary value set to 'extend'."
                )
                metric_vars = self.interp_like(mv, array, "extend", None)
        else:
            for axis_combinations in iterate_axis_combinations(axes):
                try:
                    # will raise KeyError if the axis combination is not in metrics
                    possible_metric_vars = [
                        self._metrics[ac] for ac in axis_combinations
                    ]
                    for possible_combinations in itertools.product(
                        *possible_metric_vars
                    ):
                        metric_dims = set(
                            [d for mv in possible_combinations for d in mv.dims]
                        )
                        if metric_dims.issubset(array_dims):
                            # Condition 3: use provided metrics with matching dimensions to calculate for required metric
                            metric_vars = possible_combinations
                            break
                        else:
                            # Condition 4: metrics in the wrong position (must interpolate before multiplying)
                            possible_dims = [pc.dims for pc in possible_combinations]
                            warnings.warn(
                                f"Metric at {array.dims} being interpolated from metrics at dimensions {possible_dims}. Boundary value set to 'extend'."
                            )
                            metric_vars = tuple(
                                self.interp_like(pc, array, "extend", None)
                                for pc in possible_combinations
                            )
                    if metric_vars is not None:
                        # return the product of the metrics
                        metric_vars = functools.reduce(operator.mul, metric_vars, 1)
                        break
                except KeyError:
                    pass
        if metric_vars is None:
            raise KeyError(
                f"Unable to find any combinations of metrics for array dims {array_dims!r} and axes {axes!r}"
            )
        return metric_vars

    def interp_like(self, array, like, boundary=None, fill_value=None):
        """Compares positions between two data arrays and interpolates array to the position of like if necessary

        Parameters
        ----------
        array : DataArray
            DataArray to interpolate to the position of like
        like : DataArray
            DataArray with desired grid positions for source array
        boundary : str or dict, optional,
            boundary can either be one of {None, 'fill', 'extend', 'extrapolate'}
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition where
              the difference at the boundary will be zero.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            This sets the default value. It can be overriden by specifying the
            boundary kwarg when calling specific methods.
        fill_value : float, optional
            The value to use in the boundary condition when `boundary='fill'`.

        Returns
        -------
        array : DataArray
            Source data array with updated positions along axes matching with target array
        """

        interp_axes = []
        for axname, axis in self.axes.items():
            try:
                position_array, _ = axis._get_position_name(array)
                position_like, _ = axis._get_position_name(like)
            # This will raise a KeyError if you have multiple axes contained in self,
            # since the for-loop will go through all axes, but the method is applied for only 1 axis at a time
            # This is for cases where an axis is present in self that is not available for either array or like.
            # For the axis you are interested in interpolating, there should be data for it in grid, array, and like.
            except KeyError:
                continue
            if position_like != position_array:
                interp_axes.append(axname)

        array = self.interp(
            array,
            interp_axes,
            fill_value=fill_value,
            boundary=boundary,
        )
        return array

    def __repr__(self):
        summary = ["<xgcm.Grid>"]
        for name, axis in self.axes.items():
            is_periodic = "periodic" if axis._periodic else "not periodic"
            summary.append(
                "%s Axis (%s, boundary=%r):" % (name, is_periodic, axis.boundary)
            )
            summary += axis._coord_desc()
        return "\n".join(summary)

    def _grid_func(self, funcname, da, axis, **kwargs):
        """
        This function calls appropriate functions from `Axis` objects.
        It handles multiple axis input and weighting with metrics

        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            If not specified, default will be used.
            Optionally a dict with seperate values for each axis can be passed (see example)
        boundary : None or str or dict, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)

            Optionally a dict with separate values for each axis can be passed (see example)
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with seperate values for each axis can be passed (see example)
        vector_partner : dict, optional
            A single key (string), value (DataArray).
            Optionally a dict with seperate values for each axis can be passed (see example)
        metric_weighted : str or tuple of str or dict, optional
            Optionally use metrics to multiply/divide with appropriate metrics before/after the operation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed.
        """

        if (not isinstance(axis, list)) and (not isinstance(axis, tuple)):
            axis = [axis]
        # parse multi axis kwargs like e.g. `boundary`
        multi_kwargs = {k: self._parse_axes_kwargs(v) for k, v in kwargs.items()}

        out = da
        for axx in axis:
            kwargs = {k: v[axx] for k, v in multi_kwargs.items()}
            ax = self.axes[axx]
            kwargs.setdefault("boundary", ax.boundary)
            func = getattr(ax, funcname)
            metric_weighted = kwargs.pop("metric_weighted", False)

            if isinstance(metric_weighted, str):
                metric_weighted = (metric_weighted,)

            if metric_weighted:
                metric = self.get_metric(out, metric_weighted)
                out = out * metric

            out = func(out, **kwargs)

            if metric_weighted:
                metric_new = self.get_metric(out, metric_weighted)
                out = out / metric_new

        return out

    def interp(self, da, axis, **kwargs):
        """
        Interpolate neighboring points to the intermediate grid point along
        this axis.


        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            If not specified, default will be used.
            Optionally a dict with seperate values for each axis can be passed (see example)
        boundary : None or str or dict, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)

            Optionally a dict with separate values for each axis can be passed (see example)
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with seperate values for each axis can be passed (see example)
        vector_partner : dict, optional
            A single key (string), value (DataArray).
            Optionally a dict with seperate values for each axis can be passed (see example)
        metric_weighted : str or tuple of str or dict, optional
            Optionally use metrics to multiply/divide with appropriate metrics before/after the operation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed.

        Returns
        -------
        da_i : xarray.DataArray
            The interpolated data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if a global 2D dataset should be interpolated on both X and Y axis, but it is
        only periodic in the X axis, we can do this:

        >>> grid.interp(da, ["X", "Y"], periodic={"X": True, "Y": False})
        """
        return self._grid_func("interp", da, axis, **kwargs)

    def diff(self, da, axis, **kwargs):
        """
        Difference neighboring points to the intermediate grid point.

        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            If not specified, default will be used.
            Optionally a dict with seperate values for each axis can be passed (see example)
        boundary : None or str or dict, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)

            Optionally a dict with separate values for each axis can be passed (see example)
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with seperate values for each axis can be passed (see example)
        vector_partner : dict, optional
            A single key (string), value (DataArray).
            Optionally a dict with seperate values for each axis can be passed (see example)
        metric_weighted : str or tuple of str or dict, optional
            Optionally use metrics to multiply/divide with appropriate metrics before/after the operation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed.

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if a global 2D dataset should be differenced on both X and Y axis, but the fill
        value at the boundary should be differenc for each axis, we can do this:

        >>> grid.diff(da, ["X", "Y"], fill_value={"X": 0, "Y": 100})
        """
        return self._grid_func("diff", da, axis, **kwargs)

    def min(self, da, axis, **kwargs):
        """
        Minimum of neighboring points on the intermediate grid point.

                Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            If not specified, default will be used.
            Optionally a dict with seperate values for each axis can be passed (see example)
        boundary : None or str or dict, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)

            Optionally a dict with separate values for each axis can be passed (see example)
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with seperate values for each axis can be passed (see example)
        vector_partner : dict, optional
            A single key (string), value (DataArray).
            Optionally a dict with seperate values for each axis can be passed (see example)
        metric_weighted : str or tuple of str or dict, optional
            Optionally use metrics to multiply/divide with appropriate metrics before/after the operation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed.

        Returns
        -------
        da_i : xarray.DataArray
            The mimimum data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if we want to find the minimum of sourrounding grid cells for a global 2D dataset
        in both X and Y axis, but the fill value at the boundary should be different
        for each axis, we can do this:

        >>> grid.min(da, ["X", "Y"], fill_value={"X": 0, "Y": 100})
        """
        return self._grid_func("min", da, axis, **kwargs)

    def max(self, da, axis, **kwargs):
        """
        Maximum of neighboring points on the intermediate grid point.

        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            If not specified, default will be used.
            Optionally a dict with seperate values for each axis can be passed (see example)
        boundary : None or str or dict, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)

            Optionally a dict with separate values for each axis can be passed (see example)
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with seperate values for each axis can be passed (see example)
        vector_partner : dict, optional
            A single key (string), value (DataArray).
            Optionally a dict with seperate values for each axis can be passed (see example)
        metric_weighted : str or tuple of str or dict, optional
            Optionally use metrics to multiply/divide with appropriate metrics before/after the operation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed.

        Returns
        -------
        da_i : xarray.DataArray
            The maximum data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if we want to find the maximum of sourrounding grid cells for a global 2D dataset
        in both X and Y axis, but the fill value at the boundary should be different
        for each axis, we can do this:

        >>> grid.max(da, ["X", "Y"], fill_value={"X": 0, "Y": 100})
        """
        return self._grid_func("max", da, axis, **kwargs)

    def cumsum(self, da, axis, **kwargs):
        """
        Cumulatively sum a DataArray, transforming to the intermediate axis
        position.

        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            If not specified, default will be used.
            Optionally a dict with seperate values for each axis can be passed (see example)
        boundary : None or str or dict, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)

            Optionally a dict with separate values for each axis can be passed (see example)
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with seperate values for each axis can be passed (see example)
        metric_weighted : str or tuple of str or dict, optional
            Optionally use metrics to multiply/divide with appropriate metrics before/after the operation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed.

        Returns
        -------
        da_i : xarray.DataArray
            The cumsummed data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if we want to compute the cumulative sum of global 2D dataset
        in both X and Y axis, but the fill value at the boundary should be different
        for each axis, we can do this:

        >>> grid.max(da, ["X", "Y"], fill_value={"X": 0, "Y": 100})
        """
        return self._grid_func("cumsum", da, axis, **kwargs)

    def _apply_vector_function(self, function, vector, **kwargs):
        # the keys, should be axis names
        assert len(vector) == 2

        warnings.warn(
            "`interp_2d_vector` and `diff_2d_vector` will be removed from future releases."
            "The same functionality will be available under the `xgcm.Grid` methods.",
            category=DeprecationWarning,
        )

        # this is currently only tested for c-grid vectors defined on edges
        # moving to cell centers. We need to detect if we got something else
        to = kwargs.get("to", "center")
        if to != "center":
            raise NotImplementedError(
                "Only vector interpolation to cell "
                "center is implemented, but got "
                "to=%r" % to
            )
        for axis_name, component in vector.items():
            axis = self.axes[axis_name]
            position, coord = axis._get_position_name(component)
            if position == "center":
                raise NotImplementedError(
                    "Only vector interpolation to cell "
                    "center is implemented, but vector "
                    "%s component is defined at center "
                    "(dims: %r)" % (axis_name, component.dims)
                )

        x_axis_name, y_axis_name = list(vector)
        x_axis, y_axis = self.axes[x_axis_name], self.axes[y_axis_name]

        # apply for each component
        x_component = function(
            x_axis,
            vector[x_axis_name],
            vector_partner={y_axis_name: vector[y_axis_name]},
            **kwargs,
        )

        y_component = function(
            y_axis,
            vector[y_axis_name],
            vector_partner={x_axis_name: vector[x_axis_name]},
            **kwargs,
        )

        return {x_axis_name: x_component, y_axis_name: y_component}

    def interp_2d_vector(self, vector, **kwargs):
        """
        Interpolate a 2D vector to the intermediate grid point. This method is
        only necessary for complex grid topologies.

        Parameters
        ----------
        vector : dict
            A dictionary with two entries. Keys are axis names, values are
            vector components along each axis.
        to : {'center', 'left', 'right', 'inner', 'outer'}
            The direction in which to shift the array. If not specified,
            default will be used.
        boundary : {None, 'fill', 'extend'}
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
        fill_value : float, optional
            The value to use in the boundary condition with `boundary='fill'`.
        vector_partner : dict, optional
            A single key (string), value (DataArray)
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.

        Returns
        -------
        vector_interp : dict
            A dictionary with two entries. Keys are axis names, values
            are interpolated vector components along each axis
        """

        return self._apply_vector_function(Axis.interp, vector, **kwargs)

    def derivative(self, da, axis, **kwargs):
        """
        Take the centered-difference derivative along specified axis.

        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            If not specified, default will be used.
            Optionally a dict with seperate values for each axis can be passed (see example)
        boundary : None or str or dict, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)

            Optionally a dict with separate values for each axis can be passed (see example)
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with seperate values for each axis can be passed (see example)
        vector_partner : dict, optional
            A single key (string), value (DataArray).
            Optionally a dict with seperate values for each axis can be passed (see example)
        metric_weighted : str or tuple of str or dict, optional
            Optionally use metrics to multiply/divide with appropriate metrics before/after the operation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed.

        Returns
        -------
        da_i : xarray.DataArray
            The differentiated data
        """

        ax = self.axes[axis]
        diff = ax.diff(da, **kwargs)
        dx = self.get_metric(diff, (axis,))
        return diff / dx

    def integrate(self, da, axis, **kwargs):
        """
        Perform finite volume integration along specified axis or axes,
        accounting for grid metrics. (e.g. cell length, area, volume)

        Parameters
        ----------
        axis : str, list of str
            Name of the axis on which to act
        **kwargs: dict
            Additional arguments passed to `xarray.DataArray.sum`

        Returns
        -------
        da_i : xarray.DataArray
            The integrated data
        """

        weight = self.get_metric(da, axis)
        weighted = da * weight
        # TODO: We should integrate xarray.weighted once available.

        # get dimension(s) corresponding to `da` and `axis` input
        dim = self._get_dims_from_axis(da, axis)

        return weighted.sum(dim, **kwargs)

    def cumint(self, da, axis, **kwargs):
        """
        Perform cumulative integral along specified axis or axes,
        accounting for grid metrics. (e.g. cell length, area, volume)

        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            If not specified, default will be used.
            Optionally a dict with separate values for each axis can be passed (see example)
        boundary : None or str or dict, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)

            Optionally a dict with separate values for each axis can be passed.
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with separate values for each axis can be passed.
        metric_weighted : str or tuple of str or dict, optional
            Optionally use metrics to multiply/divide with appropriate metrics before/after the operation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed.

        Returns
        -------
        da_i : xarray.DataArray
            The cumulatively integrated data
        """

        weight = self.get_metric(da, axis)
        weighted = da * weight
        # TODO: We should integrate xarray.weighted once available.

        return self.cumsum(weighted, axis, **kwargs)

    def average(self, da, axis, **kwargs):
        """
        Perform weighted mean reduction along specified axis or axes,
        accounting for grid metrics. (e.g. cell length, area, volume)

        Parameters
        ----------
        axis : str, list of str
            Name of the axis on which to act
        **kwargs: dict
            Additional arguments passed to `xarray.DataArray.weighted.mean`

        Returns
        -------
        da_i : xarray.DataArray
            The averaged data
        """

        weight = self.get_metric(da, axis)
        weighted = da.weighted(weight)

        # get dimension(s) corresponding to `da` and `axis` input
        dim = self._get_dims_from_axis(da, axis)
        return weighted.mean(dim, **kwargs)

    def transform(self, da, axis, target, **kwargs):
        """Convert an array of data to new 1D-coordinates along `axis`.
        The method takes a multidimensional array of data `da` and
        transforms it onto another data_array `target_data` in the
        direction of the axis (for each 1-dimensional 'column').

        `target_data` can be e.g. the existing coordinate along an
        axis, like depth. xgcm automatically detects the appropriate
        coordinate and then transforms the data from the input
        positions to the desired positions defined in `target`. This
        is the default behavior. The method can also be used for more
        complex cases like transforming a dataarray into new
        coordinates that are defined by e.g. a tracer field like
        temperature, density, etc.

        Currently two methods are supported to carry out the
        transformation:

        - 'linear': Values are linear interpolated between 1D columns
          along `axis` of `da` and `target_data`. This method requires
          `target_data` to increase/decrease monotonically. `target`
          values are interpreted as new cell centers in this case. By
          default this method will return nan for values in `target` that
          are outside of the range of `target_data`, setting
          `mask_edges=False` results in the default np.interp behavior of
          repeated values.

        - 'conservative': Values are transformed while conserving the
          integral of `da` along each 1D column. This method can be used
          with non-monotonic values of `target_data`. Currently this will
          only work with extensive quantities (like heat, mass, transport)
          but not with intensive quantities (like temperature, density,
          velocity). N given `target` values are interpreted as cell-bounds
          and the returned array will have N-1 elements along the newly
          created coordinate, with coordinate values that are interpolated
          between `target` values.

        Parameters
        ----------
        da : xr.DataArray
            Input data
        axis : str
            Name of the axis on which to act
        target : {np.array, xr.DataArray}
            Target points for transformation. Dependin on the method is
            interpreted as cell center (method='linear') or cell bounds
            (method='conservative).
            Values correpond to `target_data` or the existing coordinate
            along the axis (if `target_data=None`). The name of the
            resulting new coordinate is determined by the input type.
            When passed as numpy array the resulting dimension is named
            according to `target_data`, if provided as xr.Dataarray
            naming is inferred from the `target` input.
        target_data : xr.DataArray, optional
            Data to transform onto (e.g. a tracer like density or temperature).
            Defaults to None, which infers the appropriate coordinate along
            `axis` (e.g. the depth).
        method : str, optional
            Method used to transform, by default "linear"
        mask_edges : bool, optional
            If activated, `target` values outside the range of `target_data`
            are masked with nan, by default True. Only applies to 'linear' method.
        bypass_checks : bool, optional
            Only applies for `method='linear'`.
            Option to bypass logic to flip data if monotonically decreasing along the axis.
            This will improve performance if True, but the user needs to ensure that values
            are increasing alon the axis.
        suffix : str, optional
            Customizable suffix to the name of the output array. This will
            be added to the original name of `da`. Defaults to `_transformed`.

        Returns
        -------
        xr.DataArray
            The transformed data


        """

        ax = self.axes[axis]
        return ax.transform(da, target, **kwargs)

    def diff_2d_vector(self, vector, **kwargs):
        """
        Difference a 2D vector to the intermediate grid point. This method is
        only necessary for complex grid topologies.

        Parameters
        ----------
        vector : dict
            A dictionary with two entries. Keys are axis names, values are
            vector components along each axis.
        to : {'center', 'left', 'right', 'inner', 'outer'}
            The direction in which to shift the array. If not specified,
            default will be used.
        boundary : {None, 'fill', 'extend'}
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
        fill_value : float, optional
            The value to use in the boundary condition with `boundary='fill'`.
        vector_partner : dict, optional
            A single key (string), value (DataArray)
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.
        Returns
        -------
        vector_diff : dict
            A dictionary with two entries. Keys are axis names, values
            are differenced vector components along each axis
        """

        return self._apply_vector_function(Axis.diff, vector, **kwargs)


def raw_interp_function(data_left, data_right):
    # linear, centered interpolation
    # TODO: generalize to higher order interpolation
    return 0.5 * (data_left + data_right)


def raw_diff_function(data_left, data_right):
    return data_right - data_left


def raw_min_function(data_left, data_right):
    return np.minimum(data_right, data_left)


def raw_max_function(data_left, data_right):
    return np.maximum(data_right, data_left)


_other_docstring_options = """
    * 'dirichlet'
       The value of the array at the boundary point is specified by
       `fill_value`.
    * 'neumann'
       The value of the array diff at the boundary point is
       specified[1]_ by `fill_value`.

        .. [1] https://en.wikipedia.org/wiki/Dirichlet_boundary_condition
        .. [2] https://en.wikipedia.org/wiki/Neumann_boundary_condition
"""
