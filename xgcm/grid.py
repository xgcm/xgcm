import functools
import inspect
import itertools
import operator
import warnings
from collections import OrderedDict

import numpy as np
import xarray as xr
from dask.array import Array as Dask_Array

from . import comodo, gridops
from .duck_array_ops import _apply_boundary_condition, _pad_array, concatenate
from .grid_ufunc import (
    GridUFunc,
    _check_data_input,
    _GridUFuncSignature,
    _has_chunked_core_dims,
    _maybe_unpack_vector_component,
    apply_as_grid_ufunc,
)
from .metrics import iterate_axis_combinations

try:
    import numba  # type: ignore

    from .transform import conservative_interpolation, linear_interpolation
except ImportError:
    numba = None

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Only need this until python 3.8
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


def _maybe_promote_str_to_list(a):
    # TODO: improve this
    if isinstance(a, str):
        return [a]
    else:
        return a


_VALID_BOUNDARY = [None, "fill", "extend", "periodic"]


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
            boundary can either be one of {None, 'fill', 'extend', 'extrapolate', 'periodic'}

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition where
              the difference at the boundary will be zero.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            * 'periodic' : Wrap arrays around. Equivalent to setting `periodic=True`
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
        warnings.warn(
            "From version 0.8.0 the Axis computation methods will be removed, "
            "in favour of using the Grid computation methods instead. "
            f"i.e. use `Grid.{f}` instead of `Axis.{f}`",
            FutureWarning,
        )

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
        warnings.warn(
            "From version 0.8.0 the Axis computation methods will be removed, "
            "in favour of using the Grid computation methods instead. "
            "i.e. use `Grid.cumsum` instead of `Axis.cumsum`",
            FutureWarning,
        )

        pos, dim = self._get_position_name(da)

        if to is None:
            to = self._default_shifts[pos]

        # first use xarray's cumsum method
        da_cum = da.cumsum(dim=dim)

        # _maybe_get_axis_kwarg_from_mapping is needed to ensure backwards compatibility
        # axis methods cannot deal with a dict input for e.g. boundary etc.

        boundary_kwargs = dict(
            boundary=_maybe_get_axis_kwarg_from_mapping(boundary, self.name),
            fill_value=_maybe_get_axis_kwarg_from_mapping(fill_value, self.name),
        )

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

        Currently three methods are supported to carry out the
        transformation:

        - 'linear': Values are linear interpolated between 1D columns
          along `axis` of `da` and `target_data`. This method requires
          `target_data` to increase/decrease monotonically. `target`
          values are interpreted as new cell centers in this case. By
          default this method will return nan for values in `target` that
          are outside of the range of `target_data`, setting
          `mask_edges=False` results in the default np.interp behavior of
          repeated values.

        - 'log': Same as 'linear', but with values interpolated
          logarithmically between 1D columns. Operates by applying `np.log`
          to the target and target data values prior to linear interpolation.

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
            interpreted as cell center (method='linear' and method='log') or
            cell bounds (method='conservative).
            Values correspond to `target_data` or the existing coordinate
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
            are masked with nan, by default True. Only applies to 'linear' and 'log'
            methods.
        bypass_checks : bool, optional
            Only applies for `method='linear'` and `method='log'`.
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
        warnings.warn(
            "From version 0.8.0 the Axis computation methods will be removed, "
            "in favour of using the Grid computation methods instead. "
            "i.e. use `Grid.transform` instead of `Axis.transform`",
            FutureWarning,
        )

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
        if method == "linear" or method == "log":
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
                logarithmic=(method == "log"),
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


_XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG = {
    "periodic": "wrap",
    "fill": "constant",
    "extend": "edge",
}


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

        # Convert all inputs to axes-kwarg mappings
        # TODO We need a way here to check valid input. Maybe also in _as_axis_kwargs?
        # Parse axis properties
        boundary = self._as_axis_kwarg_mapping(boundary, axes=all_axes)
        fill_value = self._as_axis_kwarg_mapping(fill_value, axes=all_axes)
        # TODO: In the future we want this the only place where we store these.
        # TODO: This info needs to then be accessible to e.g. pad()

        # Parse list input. This case does only apply to periodic.
        # Since we plan on deprecating it soon handle it here, so we can easily
        # remove it later
        if isinstance(periodic, list):
            periodic = {axname: True for axname in periodic}
        periodic = self._as_axis_kwarg_mapping(periodic, axes=all_axes)

        # Set properties on grid object.
        self._facedim = list(face_connections.keys())[0] if face_connections else None
        self._connections = face_connections if face_connections else None
        # TODO: I think of the face connection data as grid not axes properties, since they almost by defintion
        # TODO: involve multiple axes. In a future PR we should remove this info from the axes
        # TODO: but make sure to properly port the checking functionality!

        # Populate axes. Much of this is just for backward compatibility.
        self.axes = OrderedDict()
        for axis_name in all_axes:
            # periodic
            is_periodic = periodic.get(axis_name, False)

            # default_shifts
            if axis_name in default_shifts:
                axis_default_shifts = default_shifts[axis_name]
            else:
                axis_default_shifts = {}

            # boundary
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
                axis_fillvalue = fill_value.get(
                    axis_name, None
                )  # TODO: This again sets defaults. Dont do that here.
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

        # Finish setup

    def _as_axis_kwarg_mapping(
        self,
        kwargs: Union[Any, Dict[str, Any]],
        axes: Iterable[str] = None,
        ax_property_name=None,
        default_value: Any = None,
    ) -> Dict[str, Any]:
        """Convert kwarg input into dict for each available axis
        E.g. for a grid with 2 axes for the keyword argument `periodic`
        periodic = True --> periodic = {'X': True, 'Y':True}
        or if not all axes are provided, the other axes will be parsed as defaults (None)
        periodic = {'X':True} --> periodic={'X': True, 'Y':None}
        """
        if axes is None:
            axes = self.axes

        parsed_kwargs: Dict[str, Any] = dict()

        if isinstance(kwargs, dict):
            parsed_kwargs = kwargs
        else:
            for axname in axes:
                parsed_kwargs[axname] = kwargs

        # Check axis properties for values that were not provided (before using the default)
        if ax_property_name is not None:
            for axname in axes:
                if axname not in parsed_kwargs.keys() or parsed_kwargs[axname] is None:
                    ax_property = getattr(self.axes[axname], ax_property_name)
                    parsed_kwargs[axname] = ax_property

        # if None set to default value.
        parsed_kwargs_w_defaults = {
            k: default_value if v is None else v for k, v in parsed_kwargs.items()
        }
        # At this point the output should be guaranteed to have an entry per existing axis.
        # If neither a default value was given, nor an axis property was found, the value will be mapped to None.

        # temporary hack to get periodic conditions from axis
        if ax_property_name == "boundary":
            for axname in axes:
                if self.axes[axname]._periodic:
                    if axname not in parsed_kwargs_w_defaults.keys():
                        parsed_kwargs_w_defaults[axname] = "periodic"

        return parsed_kwargs_w_defaults

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
        da = _maybe_unpack_vector_component(da)
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

    def _1d_grid_ufunc_dispatch(
        self,
        funcname,
        data: Union[xr.DataArray, Dict[str, xr.DataArray]],
        axis,
        to=None,
        keep_coords=False,
        metric_weighted: Union[
            str, Iterable[str], Dict[str, Union[str, Iterable[str]]]
        ] = None,
        other_component: Optional[Dict[str, xr.DataArray]] = None,
        **kwargs,
    ):
        """
        Calls appropriate 1D grid ufuncs on data, along the specified axes, sequentially.

        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            Can be passed as a single str to use for all axis, or as a dict with separate values for each axis.
            If not specified, the `default_shifts` stored in each Axis object will be used for that axis.
        """

        if isinstance(axis, str):
            axis = [axis]

        # This function is restricted to a single data input, so we need to check the input validity
        # here early.
        # TODO: This will fail if a sequence of inputs is passed, but not with a very helpful error
        # TODO: message. @TOM do you think it is worth to check the type and raise another error in that case?
        data = _check_data_input(data, self)

        # Unpack data for various steps below
        data_unpacked = _maybe_unpack_vector_component(data)

        # convert input arguments into axes-kwarg mappings
        to = self._as_axis_kwarg_mapping(to)

        if isinstance(metric_weighted, str):
            metric_weighted = (metric_weighted,)
        metric_weighted = self._as_axis_kwarg_mapping(metric_weighted)

        signatures = self._create_1d_grid_ufunc_signatures(
            data_unpacked, axis=axis, to=to
        )

        # if any dims are chunked then we need dask
        if isinstance(data_unpacked.data, Dask_Array):
            dask = "parallelized"
        else:
            dask = "forbidden"

        if isinstance(data, dict):
            array = {k: v.copy(deep=False) for k, v in data.items()}
        else:
            # Need to copy to avoid modifying in-place. Ideally we would test for this behaviour specifically
            array = data.copy(deep=False)

        # Apply 1D function over multiple axes
        # TODO This will call xarray.apply_ufunc once for each axis, but if signatures + kwargs are the same then we
        # TODO only actually need to call apply_ufunc once for those axes
        for signature_1d, ax_name in zip(signatures, axis):

            grid_ufunc, remaining_kwargs = _select_grid_ufunc(
                funcname, signature_1d, module=gridops, **kwargs
            )
            ax_metric_weighted = metric_weighted[ax_name]

            if ax_metric_weighted:
                metric = self.get_metric(array, ax_metric_weighted)
                array = array * metric

            # if chunked along core dim then we need map_overlap
            core_dim = self._get_dims_from_axis(data, ax_name)
            if _has_chunked_core_dims(data_unpacked, core_dim):
                # cumsum is a special case because it can't be correctly applied chunk-wise with map_overlap
                # (it would need blockwise instead)
                map_overlap = True if funcname != "cumsum" else False
                dask = "allowed"
            else:
                map_overlap = False

            array = grid_ufunc(
                self,
                array,
                axis=[(ax_name,)],
                keep_coords=keep_coords,
                dask=dask,
                map_overlap=map_overlap,
                other_component=other_component,
                **remaining_kwargs,
            )

            if ax_metric_weighted:
                metric = self.get_metric(array, ax_metric_weighted)
                array = array / metric

        return self._transpose_to_keep_same_dim_order(data_unpacked, array, axis)

    def _create_1d_grid_ufunc_signatures(
        self, da, axis, to
    ) -> List[_GridUFuncSignature]:
        """
        Create a list of signatures to pass to apply_grid_ufunc.

        Created from data, list of input axes, and list of target axis positions.
        One separate signature is created for each axis the 1D ufunc is going to be applied over.
        """

        signatures = []
        for ax_name in axis:
            ax = self.axes[ax_name]

            from_pos, _ = ax._get_position_name(da)  # removed `dim` since it wasnt used

            to_pos = to[ax_name]
            if to_pos is None:
                to_pos = ax._default_shifts[from_pos]

            # TODO build this more directly?
            signature_1d = _GridUFuncSignature.from_string(
                f"({ax_name}:{from_pos})->({ax_name}:{to_pos})"
            )
            signatures.append(signature_1d)

        return signatures

    def _transpose_to_keep_same_dim_order(self, da, result, axis):
        """Reorder DataArray dimensions to match the original input."""

        initial_dims = da.dims

        shifted_dims = {}
        for ax_name in axis:
            ax = self.axes[ax_name]

            _, old_dim = ax._get_position_name(da)
            _, new_dim = ax._get_position_name(result)
            shifted_dims[old_dim] = new_dim

        output_dims_but_in_original_order = [
            shifted_dims[dim] if dim in shifted_dims else dim for dim in initial_dims
        ]

        return result.transpose(*output_dims_but_in_original_order)

    def apply_as_grid_ufunc(
        self,
        func: Callable,
        *args: xr.DataArray,
        axis: Sequence[Sequence[str]] = None,
        signature: Union[str, _GridUFuncSignature] = "",
        boundary_width: Mapping[str, Tuple[int, int]] = None,
        boundary: Union[str, Mapping[str, str]] = None,
        fill_value: Union[float, Mapping[str, float]] = None,
        dask: Literal["forbidden", "parallelized", "allowed"] = "forbidden",
        map_overlap: bool = False,
        **kwargs,
    ):
        """
        Apply a function to the given arguments in a grid-aware manner.

        The relationship between xgcm axes on the input and output are specified by
        `signature`. Wraps xarray.apply_ufunc, but determines the core dimensions
        from the grid and signature passed.

        Parameters
        ----------
        func : callable
            Function to call like `func(*args, **kwargs)` on numpy-like unlabeled
            arrays (`.data`).

            Passed directly on to `xarray.apply_ufunc`.
        *args : xarray.DataArray
            One or more xarray DataArray objects to apply the function to.
        axis : Sequence[Sequence[str]], optional
            Names of xgcm.Axes on which to act, for each array in args. Multiple axes can be passed as a sequence (e.g. ``['X', 'Y']``).
            Function will be executed over all Axes simultaneously, and each Axis must be present in the Grid.
        signature : string
            Grid universal function signature. Specifies the xgcm.Axis names and
            positions for each input and output variable, e.g.,

            ``"(X:center)->(X:left)"`` for ``diff_center_to_left(a)`.
        boundary_width : Dict[str: Tuple[int, int]
            The widths of the boundaries at the edge of each array.
            Supplied in a mapping of the form {axis_name: (lower_width, upper_width)}.
        boundary : {None, 'fill', 'extend', 'extrapolate', dict}, optional
            A flag indicating how to handle boundaries:
            * None: Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            Optionally a dict mapping axis name to separate values for each axis
            can be passed.
        fill_value : {float, dict}, optional
            The value to use in boundary conditions with `boundary='fill'`.
            Optionally a dict mapping axis name to separate values for each axis
            can be passed. Default is 0.
        dask : {"forbidden", "allowed", "parallelized"}, default: "forbidden"
            How to handle applying to objects containing lazy data in the form of
            dask arrays. Passed directly on to `xarray.apply_ufunc`.
        map_overlap : bool, optional
            Whether or not to automatically apply the function along chunked core dimensions using dask.array.map_overlap.
            Default is False. If True, will need to be accompanied by dask='allowed'.

        Returns
        -------
        results
            The result of the call to `xarray.apply_ufunc`, but including the coordinates
            given by the signature, which are read from the grid. Output is either a single
            object or a tuple of such objects.

        See Also
        --------
        apply_as_grid_ufunc
        as_grid_ufunc
        """
        return apply_as_grid_ufunc(
            func,
            *args,
            axis=axis,
            grid=self,
            signature=signature,
            boundary_width=boundary_width,
            boundary=boundary,
            fill_value=fill_value,
            dask=dask,
            map_overlap=map_overlap,
            **kwargs,
        )

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
        return self._1d_grid_ufunc_dispatch("interp", da, axis, **kwargs)

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
        return self._1d_grid_ufunc_dispatch("diff", da, axis, **kwargs)

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
        return self._1d_grid_ufunc_dispatch("min", da, axis, **kwargs)

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
        return self._1d_grid_ufunc_dispatch("max", da, axis, **kwargs)

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
        return self._1d_grid_ufunc_dispatch("cumsum", da, axis, **kwargs)

    def _apply_vector_function(self, function, vector, **kwargs):
        if not (len(vector) == 2 and isinstance(vector, dict)):
            raise ValueError(
                "Input is expected to be a dictionary with two key/value pairs which map grid axis to the vector component parallel to that axis"
            )

        warnings.warn(
            "`interp_2d_vector` and `diff_2d_vector` will be removed from future releases."
            "The same functionality will be accessible under the `xgcm.Grid.diff` and `xgcm.Grid.interp` methods, please see those docstrings for details.",
            category=DeprecationWarning,
        )

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

        # apply for each component
        x_component = function(
            {x_axis_name: vector[x_axis_name]},
            x_axis_name,
            other_component={y_axis_name: vector[y_axis_name]},
            **kwargs,
        )

        y_component = function(
            {y_axis_name: vector[y_axis_name]},
            y_axis_name,
            other_component={x_axis_name: vector[x_axis_name]},
            **kwargs,
        )
        return {x_axis_name: x_component, y_axis_name: y_component}

    def diff_2d_vector(self, vector, **kwargs):
        """
        Difference a 2D vector to the intermediate grid point. This method is
        only necessary for complex grid topologies.

        Parameters
        ----------
        vector : dict
            A dictionary with two entries. Keys are axis names, values are
            vector components along each axis.

        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        vector_diff : dict
            A dictionary with two entries. Keys are axis names, values
            are differenced vector components along each axis
        """
        return self._apply_vector_function(self.diff, vector, **kwargs)

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

        return self._apply_vector_function(self.interp, vector, **kwargs)

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
        diff = self.diff(da, axis, **kwargs)
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

        Currently three methods are supported to carry out the
        transformation:

        - 'linear': Values are linear interpolated between 1D columns
          along `axis` of `da` and `target_data`. This method requires
          `target_data` to increase/decrease monotonically. `target`
          values are interpreted as new cell centers in this case. By
          default this method will return nan for values in `target` that
          are outside of the range of `target_data`, setting
          `mask_edges=False` results in the default np.interp behavior of
          repeated values.

        - 'log': Same as 'linear', but with values interpolated
          logarithmically between 1D columns. Operates by applying `np.log`
          to the target and target data values prior to linear interpolation.

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
            Target points for transformation. Depending on the method is
            interpreted as cell center (method='linear' and method='log') or
            cell bounds (method='conservative).
            Values correspond to `target_data` or the existing coordinate
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
            are masked with nan, by default True. Only applies to 'linear' and
            'log' methods.
        bypass_checks : bool, optional
            Only applies for `method='linear'` and `method='log'`.
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

        ax = self.axes[axis]
        return ax.transform(da, target, **kwargs)


def _select_grid_ufunc(funcname, signature: _GridUFuncSignature, module, **kwargs):
    # TODO to select via other kwargs (e.g. boundary) the signature of this function needs to be generalised

    def is_grid_ufunc(obj):
        return isinstance(obj, GridUFunc)

    # This avoids defining a list of functions in gridops.py
    all_predefined_ufuncs = inspect.getmembers(module, is_grid_ufunc)

    name_matching_ufuncs = [
        f for name, f in all_predefined_ufuncs if name.startswith(funcname)
    ]
    if len(name_matching_ufuncs) == 0:
        raise NotImplementedError(
            f"Could not find any pre-defined {funcname} grid ufuncs"
        )

    signature_matching_ufuncs = [
        f for f in name_matching_ufuncs if f.signature.equivalent(signature)
    ]
    if len(signature_matching_ufuncs) == 0:
        raise NotImplementedError(
            f"Could not find any pre-defined {funcname} grid ufuncs with signature {signature}"
        )

    matching_ufuncs = signature_matching_ufuncs

    # TODO select via any other kwargs (such as boundary? metrics?) once implemented
    all_kwargs = kwargs.copy()
    # boundary = kwargs.pop("boundary", None)
    # if boundary:
    #    matching_ufuncs = [uf for uf in matching_ufuncs if uf.boundary == boundary]

    if len(matching_ufuncs) > 1:
        # TODO include kwargs used to match in this error message
        raise ValueError(
            f"Function {funcname} with signature='{signature}' and kwargs={all_kwargs} is an ambiguous selection"
        )
    elif len(matching_ufuncs) == 0:
        raise NotImplementedError(
            f"Could not find any pre-defined {funcname} grid ufuncs with signature='{signature}' and kwargs"
            f"={all_kwargs}"
        )
    else:
        # Exactly 1 matching function
        return matching_ufuncs[0], kwargs


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


def _maybe_get_axis_kwarg_from_mapping(
    kwargs: Union[str, float, Dict[str, Union[str, float]]], axname: str
) -> Union[str, float]:
    if isinstance(kwargs, dict):
        return kwargs[axname]
    else:
        return kwargs


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
