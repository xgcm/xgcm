"""
Handle all padding.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple, Union

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
    da: Union[xr.DataArray, Dict[str, xr.DataArray]],
    grid: Grid,
    padding_width: Dict[str, Tuple[int, int]],
    padding: Dict[str, str],
    fill_value: Dict[str, float],
    other_component: Dict[str, xr.DataArray] = None,
):
    facedim = grid._facedim
    connections = grid._connections

    if isinstance(da, dict):
        isvector = True
        vectoraxis, da = da.popitem()
    else:
        isvector = False

    if isvector:
        # TODO: Using the logic above I could save a bunch of operations below. If we are never swapping axes (_get_all_connection_axes(connections, facedim)) = 1)
        # TODO: We do not need to deal with other components
        # TODO: Need to integrate that choice deeper in the loop\.
        if other_component:
            _, da_partner = other_component.popitem()
        else:
            # TODO: cover with a test.
            raise ValueError(
                "Padding vector components requires `other_component` input."
            )

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


def _pad_basic(
    da: xr.DataArray,
    grid: Grid,
    padding_width: Dict[str, Tuple[int, int]],
    padding: Dict[str, str],
    fill_value: Dict[str, float],
):
    """Implement basic xarray/numpy padding methods"""

    da_padded = da.copy(deep=False)

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
    data: Union[xr.DataArray, Dict[str, xr.DataArray]],
    grid: Grid,
    boundary_width: Optional[Dict[str, Tuple[int, int]]],
    boundary: Union[str, Mapping[str, str], None] = None,
    fill_value: Union[float, Mapping[str, float]] = None,
    other_component: Dict[str, xr.DataArray] = None,
):
    """
    Pads the boundary of given arrays along given Axes, according to information in Axes.boundary.
    Parameters
    ----------
    data :
        Array to pad according to boundary and boundary_width.
        If a dictionary is passed the input is assumed to be a vector component
        (with the directionof that component identified by the dict key, matching one of the grid axes)
    grid : xgcm.Grid
        Grid object specifiying the topology and default boundary conditions to use for padding.
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

    # Always promote the padding/fill_value to a axed dict.
    padding = grid._as_axis_kwarg_mapping(
        padding, ax_property_name="boundary", default_value="periodic"
    )
    fill_value = grid._as_axis_kwarg_mapping(
        fill_value, ax_property_name="fill_value", default_value=0.0
    )

    # Exit without padding if all widths are zero
    if padding_width is None or all(
        width == (0, 0) for width in padding_width.values()
    ):
        # TODO: Think about case when boundary is specified but boundary_width is None or (0,0).
        # TODO: No padding would occur in that situation. Should we warn the user?
        return data

    # Check axis properties for padding/fill_value, but do not overwrite
    for axname, axis in grid.axes.items():
        if axname not in padding.keys():
            # Use default axis boundary for axis unless overridden
            padding[axname] = axis.boundary
        if axname not in fill_value.keys():
            # Use default axis boundary for axis unless overridden
            fill_value[axname] = axis.fill_value

    # TODO: Refactor, if the max value is 0, complain.
    # Maybe move this check upstream, so that either none or 0 everywhere does not even call pad
    if padding_width is None:
        raise ValueError("Must provide the widths of the boundaries")

    # The problem of what values to pad coordinates with is hard to solve.
    # Instead of attempting that we will strip all coordinates (including dimension coordinates)
    # before dispatching to the utility pad functions.
    # This ensures that any output from this function is stripped.
    data = _strip_all_coords(data)

    # If any axis has connections we need to use the complex padding
    if any([any(grid.axes[ax]._connections.keys()) for ax in grid.axes]):
        da_padded = _pad_face_connections(
            data,
            grid,
            padding_width,
            padding,
            fill_value,
            other_component=other_component,
        )
    else:
        da_padded = _pad_basic(data, grid, padding_width, padding, fill_value)  # type: ignore

    return da_padded
