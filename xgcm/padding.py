"""
Handle all padding.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from .grid import Grid

_XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG = {
    "periodic": "wrap",
    "fill": "constant",
    "extend": "edge",
    None: "wrap",  # current default is periodic. This should achieve that.
}


def _maybe_rename_grid_positions(grid, arr_source, arr_target):
    # Checks and renames all dimensions in arr_source to the grid
    # position in arr_target (only if dims are valid axis positions)
    rename_dict = {}
    for di in arr_target.dims:
        # in case the dimension is already in the source, do nothing.
        if di not in arr_source:
            # find associated axis
            for axname in grid.axes:
                all_positions = grid.axes[axname].coords.values()
                if di in all_positions:
                    source_dim = [p for p in all_positions if p in arr_source.dims][
                        0
                    ]  # TODO: there must be a more elegant way?
                    rename_dict[source_dim] = di
    return arr_source.rename(rename_dict)


def _maybe_swap_dimension_names(da, from_name, to_name):
    # renames 1D slices and swaps dimension names for higher dimensional slices
    if to_name in da.dims:
        da = da.rename({to_name: to_name + "dummy"})
        if from_name in da.dims:
            da = da.rename({from_name: to_name})
        da = da.rename({to_name + "dummy": from_name})
    else:
        da = da.rename({from_name: to_name})
    return da


def _get_all_connection_axes(connections, facedim):
    all_axes = []
    for c in connections[facedim].values():
        all_axes.extend(list(c.keys()))
    return list(set(all_axes))


def _strip_all_coords(obj: xr.DataArray):
    if isinstance(obj, dict):
        return {k: _strip_all_coords(v) for k, v in obj.items()}
    else:
        obj_stripped = obj.reset_coords(drop=True).reset_index(
            [di for di in obj.dims if di in obj.coords], drop=True
        )
        return obj_stripped


def _pad_face_connections(
    da, grid, padding_width, padding, other_component, fill_value
):

    # i made some alterations to the grid init to store the
    # following info on the grid level. We can discuss if that makes more sense
    # TODO: Do we need that information on the axis still?
    facedim = grid._facedim
    connections = grid._connections

    if isinstance(da, dict):
        assert len(da) == 1  # TODO raise a better error here
        isvector = True
        vectoraxis, da = da.popitem()
    else:
        isvector = False

    if isvector:
        assert other_component is not None  # TODO raise a better error here
        _, da_partner = other_component.popitem()

    # Detect all the axes we have to deal with during padding
    # all the axes defined in the connections + the axes of the padding width should give all axes we need to iterate over
    pad_axes = list(
        set(_get_all_connection_axes(connections, facedim) + list(padding_width.keys()))
    )

    padding_width = {axname: padding_width.get(axname, (0, 0)) for axname in pad_axes}

    # This method below works really nicely if all the boundary widths have the same size.
    # This is however not very common. We often have boundary_width with (0,1).
    # I had a ton of trouble accomodating with convoluted logic. The new approach:
    # we find the largest boundary width value, and pad every boundary/axis with this max
    # value. As a final step we trim the padded dataset according to the original boundary
    # widths.

    def _max_boundary_width(padding_width):
        all_widths = []
        for widths in padding_width.values():
            all_widths.extend(list(widths))
        return max(all_widths)

    width = _max_boundary_width(padding_width)

    # TODO should i just return da if width is 0 here? I have a check further down, that I could eliminate that way.
    max_padding_width = {k: (width, width) for k in padding_width.keys()}

    # The edges of the array which are not connected, need to be padded with the
    # 'basic' padding. This also ensures that the resulting padded faces result in
    # the same shape. We will pad everything now and then replace the connections.
    # That might however not be the most computational efficient way to do it.

    da_prepadded = _pad_basic(
        da,
        grid,
        max_padding_width,
        padding,
        fill_value,
    )

    if isvector:
        da_partner_prepadded = _pad_basic(
            da_partner,
            grid,
            max_padding_width,
            padding,
            fill_value,
        )

    n_facedim = len(da[facedim])
    faces = []

    # Iterate over each face and pad accordingly
    for i in range(n_facedim):
        target_da = da_prepadded.isel({facedim: i})
        connection_single = connections[facedim][i]
        for axname in pad_axes:
            # get any connections relevant to the current axis, default to None.
            (left_connection, right_connection) = connection_single.get(
                axname, (None, None)
            )
            _, target_dim = grid.axes[axname]._get_position_name(target_da)

            for connection, is_right in [
                (left_connection, False),
                (right_connection, True),
            ]:
                if width > 0:
                    if connection:
                        # apply face connection logic #
                        source_face, source_axis, reverse = connection

                        # is the connection along the same axis or not
                        swap_axis = False
                        if axname != source_axis:
                            swap_axis = True

                        # choose the source for padding
                        source_da = da_prepadded.isel({facedim: source_face})
                        if isvector:
                            if swap_axis:
                                source_da = da_partner_prepadded.isel(
                                    {facedim: source_face}
                                )
                                # adjust the dimension naming (only ever needed when swapping variables)
                                source_da = _maybe_rename_grid_positions(
                                    grid, source_da, target_da
                                )

                        _, source_dim = grid.axes[source_axis]._get_position_name(
                            source_da
                        )

                        # I guess this could be more elegant. Basically I only want to replace the
                        # unpadded part of the source/target, since padding methods could be different for
                        # different axes...
                        # I am thinking about how to make this easier later
                        if is_right:
                            # the right connection should be padded with the leftmost index
                            # of the source unless reverse is true
                            if reverse:
                                source_slice_index = slice(-2 * width, -width)
                            else:
                                source_slice_index = slice(width, 2 * width)

                            target_slice_index = slice(0, -width)

                        else:
                            # the left connection should be padded with the rightmost index
                            # of the source unless reverse is true
                            if reverse:
                                source_slice_index = slice(width, 2 * width)
                            else:
                                source_slice_index = slice(-2 * width, -width)

                            target_slice_index = slice(width, None)

                        # after all this logic, start slicing
                        # we get the appropriate slice to add and
                        # remove the previously padded values from the target
                        source_slice = source_da.isel({source_dim: source_slice_index})
                        target_slice = target_da.isel({target_dim: target_slice_index})

                        # swap dimension names to target when axis is swapped
                        if swap_axis:
                            source_slice = _maybe_swap_dimension_names(
                                source_slice,
                                source_dim,
                                target_dim,
                            )
                        # At this point any addition should have a fixed set of orthogonal/tangential dimensions
                        ortho_dim = target_dim
                        tangential_dim = source_dim

                        # Orthogonal flip
                        if reverse:
                            source_slice = source_slice.isel(
                                {ortho_dim: slice(None, None, -1)}
                            )
                            if isvector:
                                if vectoraxis == axname:
                                    # If the input is an orthogonal vector this flip needs
                                    # to be accompanied by a sign change
                                    source_slice = -source_slice
                            # TODO: Flip sign if vector is tangential

                        # Tangential flip
                        if swap_axis and not reverse:
                            source_slice = source_slice.isel(
                                {tangential_dim: slice(None, None, -1)}
                            )
                            # If the input is a tangential vector this flip needs
                            # to be accompanied by a sign change
                            if isvector:
                                if vectoraxis != axname:
                                    source_slice = -source_slice

                        source_slice = source_slice.squeeze()

                        # clean out everything on source_slice coordinates
                        source_slice = source_slice.drop_vars(
                            [co for co in source_slice.coords]
                        )

                        # Here I am trying to emulate the way xarray.pad deals with dimension coordinates
                        # I will set them to nan in any case. This might change later.
                        if target_dim in target_slice.coords:
                            if (
                                target_dim not in source_slice.dims
                            ):  # in case this a 1 element padding slice
                                source_slice = source_slice.expand_dims([target_dim])

                            source_slice = source_slice.assign_coords(
                                {
                                    target_dim: np.full_like(
                                        source_slice[target_dim].data,
                                        np.nan,
                                        dtype=float,  # Needed to properly convert int indicies to nan (super weird)
                                    )
                                }
                            )

                        # # FUCK THIS: FOR NOW BRUTE FORCE THIS STUFF
                        # target_slice = _strip_all_coords(target_slice)
                        # source_slice = _strip_all_coords(source_slice)

                        # assemble the padded array
                        if is_right:
                            concat_list = [target_slice, source_slice]
                        else:
                            concat_list = [source_slice, target_slice]

                        target_da = xr.concat(
                            concat_list,
                            dim=target_dim,
                            coords="minimal",
                            compat="override",
                            join="override",
                        )
                        # TODO: Can we do this with an assignment in xarray? Maybe not important yet.
        faces.append(target_da)

    da_padded = xr.concat(faces, dim=facedim)

    # trim back to original shape
    def _trim_expanded_padding_width(da, grid, padding_width, padding_width_expanded):
        for axname in padding_width.keys():
            _, dim = grid.axes[axname]._get_position_name(da)
            start = padding_width_expanded[axname][0] - padding_width[axname][0]
            stop = padding_width_expanded[axname][1] - padding_width[axname][1]
            # if stop is zero we want to index the full end of the dimension
            if stop == 0:
                stop = None
            else:
                stop = -stop
            da = da.isel({dim: slice(start, stop)})
        return da

    return _trim_expanded_padding_width(
        da_padded, grid, padding_width, max_padding_width
    )


def _pad_basic(da, grid, padding_width, padding, fill_value):
    """Implement basic xarray/numpy padding methods"""

    # legacy default fill value is 0, instead of xarrays nan.
    # TODO: I think nan makes more sense as a default.
    # this should not accept anything else than an axis-kwarg mapping
    # Currently some of the map_blocks tests seem to pass None?
    if fill_value is None:
        fill_value = 0.0
    elif isinstance(fill_value, dict):
        fill_value = {k: 0.0 if v is None else v for k, v in fill_value.items()}
    fill_value = grid._as_axis_kwarg_mapping(fill_value)

    # Always promote the padding to a dict? For now this is the only input type
    # TODO: Refactor this with grid logic that creates a per axis mapping, checks values in grid object and sets defaults.
    padding = grid._as_axis_kwarg_mapping(padding)

    # TODO: We should set all defaults on the grid init and avoid this here.
    # # set defaults
    for axname, axis in grid.axes.items():
        if axname not in padding.keys():
            # Use default axis boundary for axis unless overridden
            padding[
                axname
            ] = (
                axis.boundary
            )  # TODO: rename the axis property, or are we not keeping this on the axis level?

    da_padded = da.copy()

    for ax, widths in padding_width.items():
        axis = grid.axes[ax]
        _, dim = axis._get_position_name(da)
        ax_padding = padding[ax]
        # translate padding and kwargs to xarray.pad syntax
        ax_padding = _XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG[ax_padding]
        if ax_padding == "constant":
            kwargs = dict(constant_values=fill_value[ax])
        else:
            kwargs = dict()
        da_padded = da_padded.pad({dim: widths}, ax_padding, **kwargs)
    return da_padded


def pad(
    da: Union[xr.DataArray, Dict[str, xr.DataArray]],
    grid: Grid,
    boundary_width: Optional[Dict[str, Tuple[int, int]]] = None,
    boundary: Optional[str] = "periodic",
    fill_value: Optional[float] = 0.0,
    other_component: Optional[Dict[str, xr.DataArray]] = None,
):
    """
    Pads the boundary of given arrays along given Axes, according to information in Axes.boundary.
    Parameters
    ----------
    da :
        Array to pad according to boundary and boundary_width.
        If a dictionary is passed the input is assumed to be a vector component
        (with the direction of that component identified by the dict key, matching one of the grid axes)
    boundary_width :
        The widths of the boundaries at the edge of each array.
        Supplied in a mapping of the form {axis_name: (lower_width, upper_width)}.
    boundary : {None, 'fill', 'extend', 'extrapolate', dict}, optional
        A flag indicating how to handle boundaries:
        * None: Defaults to `periodic`
        * 'periodic' : Wrap array along the specified axes
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Dirichlet boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Neumann boundary condition.)
        * 'extrapolate': Set values by extrapolating linearly from the two
          points nearest to the edge
        Optionally a dict mapping axis name to separate values for each axis
        can be passed.
    fill_value :
        The value to use in boundary conditions with `boundary='fill'`.
        Optionally a dict mapping axis name to separate values for each axis
        can be passed. Default is 0.
    other_component :
        If passing a vector some padding operations require the orthogonal vector component for padding.
        This needs to be passed as a dictionary (with the direction of that component identified by the
        dict key, matching one of the grid axes)
    """
    # TODO rename this globally
    padding = boundary
    padding_width = boundary_width

    # TODO: Refactor, if the max value is 0, complain.
    # Maybe move this check upstream, so that either none or 0 everywhere does not even call pad
    if not boundary_width:
        raise ValueError("Must provide the widths of the boundaries")

    # The problem of what values to pad coordinates with is hard to solve.
    # Instead of attempting that we will strip all coordinates (including dimension coordinates)
    # before dispatching to the utility pad functions.
    # This ensures that any output from this function is stripped.
    # TODO: The coordinate values need to be reattached as part of the `apply_as_grid_ufunc` logic
    da = _strip_all_coords(da)

    # If any axis has connections we need to use the complex padding
    if any([any(grid.axes[ax]._connections.keys()) for ax in grid.axes]):
        da_padded = _pad_face_connections(
            da, grid, padding_width, padding, other_component, fill_value
        )
    else:
        # TODO: we need to have a better detection and checking here. For now just pipe everything else into the
        # Legacy cases
        da_padded = _pad_basic(da, grid, padding_width, padding, fill_value)

    return da_padded
