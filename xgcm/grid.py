from __future__ import print_function
from __future__ import absolute_import
from future.utils import iteritems
from collections import OrderedDict
import functools
import itertools
import operator

import docrep
import xarray as xr
import numpy as np

from . import comodo
from .duck_array_ops import _pad_array, _apply_boundary_condition, concatenate

docstrings = docrep.DocstringProcessor(doc_key="My doc string")


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
    There are four possible positions::

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
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            This sets the default value. It can be overriden by specifying the
            boundary kwarg when calling specific methods.
        fill_value : {float}, optional
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
            raise ValueError(f"Expected 'fill_value' to be a number.")
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
        for name, cname in iteritems(self.coords):
            coord_info = "  * %-8s %s" % (name, cname)
            if name in self._default_shifts:
                coord_info += " --> %s" % self._default_shifts[name]
            summary.append(coord_info)
        return summary

    @docstrings.get_sectionsf("neighbor_binary_func")
    @docstrings.dedent
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
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.)
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
        position_from, dim = self._get_axis_coord(da)
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

    docstrings.delete_params("neighbor_binary_func.parameters", "f")

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

        position, this_dim = self._get_axis_coord(da)
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

        position_from, dim = self._get_axis_coord(da)
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

    @docstrings.dedent
    def interp(
        self,
        da,
        to=None,
        boundary=None,
        fill_value=0.0,
        boundary_discontinuity=None,
        vector_partner=None,
        keep_coords=False,
    ):
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

    @docstrings.dedent
    def diff(
        self,
        da,
        to=None,
        boundary=None,
        fill_value=0.0,
        boundary_discontinuity=None,
        vector_partner=None,
        keep_coords=False,
    ):
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

    @docstrings.dedent
    def cumsum(self, da, to=None, boundary=None, fill_value=0.0, keep_coords=False):
        """
        Cumulatively sum a DataArray, transforming to the intermediate axis
        position.

        Parameters
        ----------
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_cum : xarray.DataArray
            The cumsummed data
        """

        pos, dim = self._get_axis_coord(da)

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

    @docstrings.dedent
    def min(
        self,
        da,
        to=None,
        boundary=None,
        fill_value=0.0,
        boundary_discontinuity=None,
        vector_partner=None,
        keep_coords=False,
    ):
        """
        Minimum of neighboring points on intermediate grid point.

        Parameters
        ----------
        %(neighbor_binary_func.parameters.no_f)s

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

    @docstrings.dedent
    def max(
        self,
        da,
        to=None,
        boundary=None,
        fill_value=0.0,
        boundary_discontinuity=None,
        vector_partner=None,
        keep_coords=False,
    ):
        """
        Maximum of neighboring points on intermediate grid point.

        Parameters
        ----------
        %(neighbor_binary_func.parameters.no_f)s

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

    def _wrap_and_replace_coords(self, da, data_new, position_to, keep_coords=False):
        """
        Take the base coords from da, the data from data_new, and return
        a new DataArray with a coordinate on position_to.
        """
        position_from, old_dim = self._get_axis_coord(da)
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

    def _get_axis_coord(self, da):
        """Return the position and name of the axis coordiante in a DataArray."""
        for position, coord_name in iteritems(self.coords):
            # TODO: should we have more careful checking of alignment here?
            if coord_name in da.dims:
                return position, coord_name

        raise KeyError(
            "None of the DataArray's dims %s were found in axis "
            "coords." % repr(da.dims)
        )

    def _get_axis_dim_num(self, da):
        """Return the dimension number of the axis coordinate in a DataArray."""
        _, coord_name = self._get_axis_coord(da)
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
            Excplicit specification of axis coordinates, e.g
            ``{'X': {'center': 'XC', 'left: 'XG'}}``.
            Each key should be the name of an axis. The value should be
            a dictionary mapping positions (e.g. ``'left'``) to names of
            coordinates in ``ds``.
        metrics : dict, optional
            Specification of grid metrics
        boundary : {None, 'fill', 'extend', 'extrapolate', dict}, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.)
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

        if coords:
            all_axes = coords.keys()
        else:
            all_axes = comodo.get_all_axes(ds)
            coords = {}

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

        if metrics is not None:
            self._assign_metrics(metrics)

    def _parse_axes_kwargs(self, kwargs):
        """Convvert kwarg input into dict for each available axis
        E.g. for a grid with 2 axes for the keyword argument `periodid`
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

    def _assign_metrics(self, metrics):
        """
        metrics should look like
           {('X', 'Y'): ['rAC']}
        check to make sure everything is a valid dimension
        """

        self._metrics = {}

        for key, metric_vars in metrics.items():
            metric_axes = frozenset(_maybe_promote_str_to_list(key))
            if not all([ma in self.axes for ma in metric_axes]):
                raise KeyError(
                    "Metric axes %r not compatible with grid axes %r"
                    % (metric_axes, tuple(self.axes))
                )
            # initialize empty list
            self._metrics[metric_axes] = []
            for metric_varname in _maybe_promote_str_to_list(metric_vars):
                if metric_varname not in self._ds:
                    raise KeyError(
                        "Metric variable %s not found in dataset." % metric_varname
                    )
                # resetting coords avoids potential broadcasting / alignment issues
                metric_var = self._ds[metric_varname].reset_coords(drop=True)
                # TODO: check for consistency of metric_var dims with axis dims
                # check for duplicate dimensions among each axis metric
                self._metrics[metric_axes].append(metric_var)

    def _get_dims_from_axis(self, da, axis):
        dim = []
        for ax in axis:
            all_dim = self.axes[ax].coords.values()
            matching_dim = [di for di in all_dim if di in da.dims]
            if len(matching_dim) == 1:
                dim.append(matching_dim[0])
            else:
                raise ValueError(
                    "Did not find single matching dimension corresponding to axis %s. Got (%s)"
                    % (ax, matching_dim)
                )
        return dim

    def get_metric(self, array, axes):
        """
        Find the metric variable associated with a set of axes for a particular
        array.

        Parameters
        ----------
        array : xarray.DataArray
            The array for which we are looking for a metric. Only its
            dimensions are considered.
        axes : iterable
            A list of axes for which to find the metric.

        Returns
        -------
        metric : xarray.DataArray
            A metric which can broadcast against ``array``
        """

        # a function to find the right combination of metrics
        def iterate_axis_combinations(items):
            items_set = frozenset(items)
            yield (items_set,)
            N = len(items)
            for nleft in range(N - 1, 0, -1):
                nright = N - nleft
                for sub_loop, sub_items in itertools.product(
                    range(min(nright, nleft), 0, -1),
                    itertools.combinations(items_set, nleft),
                ):
                    these = frozenset(sub_items)
                    those = items_set - these
                    others = [
                        frozenset(i) for i in itertools.combinations(those, sub_loop)
                    ]
                    yield (these,) + tuple(others)

        metric_vars = None
        array_dims = set(array.dims)
        for axis_combinations in iterate_axis_combinations(axes):
            try:
                # will raise KeyError if the axis combination is not in metrics
                possible_metric_vars = [self._metrics[ac] for ac in axis_combinations]
                for possible_combinations in itertools.product(*possible_metric_vars):
                    metric_dims = set(
                        [d for mv in possible_combinations for d in mv.dims]
                    )
                    if metric_dims.issubset(array_dims):
                        # we found a set of metrics with dimensions compatible
                        # with the array
                        metric_vars = possible_combinations
                        break
                if metric_vars is not None:
                    break
            except KeyError:
                pass
        if metric_vars is None:
            raise KeyError(
                "Unable to find any combinations of metrics for "
                "array dims %r and axes %r" % (array_dims, axes)
            )

        # return the product of the metrics
        return functools.reduce(operator.mul, metric_vars, 1)

    def __repr__(self):
        summary = ["<xgcm.Grid>"]
        for name, axis in iteritems(self.axes):
            is_periodic = "periodic" if axis._periodic else "not periodic"
            summary.append(
                "%s Axis (%s, boundary=%r):" % (name, is_periodic, axis.boundary)
            )
            summary += axis._coord_desc()
        return "\n".join(summary)

    @docstrings.get_sectionsf("grid_func", sections=["Parameters", "Examples"])
    @docstrings.dedent
    def _grid_func(self, funcname, da, axis, **kwargs):
        """this function calls appropriate functions from `Axis` objects.
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
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.)

            Optionally a dict with seperate values for each axis can be passed (see example)
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with seperate values for each axis can be passed (see example)
        vector_partner : dict, optional
            A single key (string), value (DataArray).
            Optionally a dict with seperate values for each axis can be passed (see example)
        metric_weighted : str or tuple of str or dict, optional
            If an axis or list of axes is specified,
            the appropriate grid metrics will be used to determined the weight for interpolation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed (see example)

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

    @docstrings.dedent
    def interp(self, da, axis, **kwargs):
        """
        Interpolate neighboring points to the intermediate grid point along
        this axis.


        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

        Returns
        -------
        da_i : xarray.DataArray
            The interpolated data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if a global 2D dataset should be interpolated on both X and Y axis, but it is
        only periodic in the X axis, we can do this:

        >>> grid.interp(da, ['X', 'Y'], periodic={'X':True, 'Y':False})
        """
        return self._grid_func("interp", da, axis, **kwargs)

    @docstrings.dedent
    def diff(self, da, axis, **kwargs):
        """
        Difference neighboring points to the intermediate grid point.

        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if a global 2D dataset should be differenced on both X and Y axis, but the fill
        value at the boundary should be differenc for each axis, we can do this:

        >>> grid.diff(da, ['X', 'Y'], fill_value={'X':0, 'Y':100})
        """
        return self._grid_func("diff", da, axis, **kwargs)

    @docstrings.dedent
    def min(self, da, axis, **kwargs):
        """
        Minimum of neighboring points on the intermediate grid point.

        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

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

        >>> grid.min(da, ['X', 'Y'], fill_value={'X':0, 'Y':100})
        """
        return self._grid_func("min", da, axis, **kwargs)

    @docstrings.dedent
    def max(self, da, axis, **kwargs):
        """
        Maximum of neighboring points on the intermediate grid point.

        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

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

        >>> grid.max(da, ['X', 'Y'], fill_value={'X':0, 'Y':100})
        """
        return self._grid_func("max", da, axis, **kwargs)

    @docstrings.dedent
    def cumsum(self, da, axis, **kwargs):
        """
        Cumulatively sum a DataArray, transforming to the intermediate axis
        position.

        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

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

        >>> grid.max(da, ['X', 'Y'], fill_value={'X':0, 'Y':100})
        """
        return self._grid_func("cumsum", da, axis, **kwargs)

    @docstrings.dedent
    def _apply_vector_function(self, function, vector, **kwargs):
        # the keys, should be axis names
        assert len(vector) == 2

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
            position, coord = axis._get_axis_coord(component)
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

    @docstrings.dedent
    def interp_2d_vector(self, vector, **kwargs):
        """
        Interpolate a 2D vector to the intermediate grid point. This method is
        only necessary for complex grid topologies.

        Parameters
        ----------
        vector : dict
            A dictionary with two entries. Keys are axis names, values are
            vector components along each axis.

        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        vector_interp : dict
            A dictionary with two entries. Keys are axis names, values
            are interpolated vector components along each axis
        """

        return self._apply_vector_function(Axis.interp, vector, **kwargs)

    @docstrings.dedent
    def derivative(self, da, axis, **kwargs):
        """
        Take the centered-difference derivative along specified axis.

        Parameters
        ----------
        axis : str
            Name of the axis on which to act
        %(grid_func.parameters)s

        Returns
        -------
        da_i : xarray.DataArray
            The differentiated data
        """

        ax = self.axes[axis]
        diff = ax.diff(da, **kwargs)
        dx = self.get_metric(diff, (axis,))
        return diff / dx

    @docstrings.dedent
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

        # get dimension(s) corresponding
        # to `da` and `axis` input
        dim = self._get_dims_from_axis(da, axis)

        return weighted.sum(dim, **kwargs)

    @docstrings.dedent
    def cumint(self, da, axis, **kwargs):
        """
        Perform cumulative integral along specified axis or axes,
        accounting for grid metrics. (e.g. cell length, area, volume)

        Parameters
        ----------
        axis : str, list of str
            Name of the axis on which to act
        %(grid_func.parameters)s

        Returns
        -------
        da_i : xarray.DataArray
            The cumulatively integrated data
        """

        weight = self.get_metric(da, axis)
        weighted = da * weight
        # TODO: We should integrate xarray.weighted once available.

        return self.cumsum(weighted, axis, **kwargs)

    @docstrings.dedent
    def average(self, da, axis, **kwargs):
        """
        Perform weighted mean reduction along specified axis or axes,
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
            The averaged data
        """

        weight = self.get_metric(da, axis)
        weighted = da * weight
        # TODO: We should integrate xarray.weighted once available.

        # get dimension(s) corresponding
        # to `da` and `axis` input
        dim = self._get_dims_from_axis(da, axis)
        # do we need to pass kwargs?
        return weighted.sum(dim, **kwargs) / weight.sum(dim, **kwargs)

    @docstrings.dedent
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
