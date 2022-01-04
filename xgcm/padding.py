"""
Handle all padding.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import xarray as xr

if TYPE_CHECKING:
    from .grid import Grid

_XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG = {
    "periodic": "wrap",
    "fill": "constant",
    "extend": "edge",
    None: "wrap",  # current default is periodic. This should achieve that.
}


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


def _pad_face_connections(da, grid, padding_width, **kwargs):

    # Can I get rid of all the mess with the coordinates by stripping them here?
    # The output of the padding will not fit any coordinates anymore anyways and
    # after the ufunc is applied they will all be back on place. So lets make our
    # lives easier here.
    da = da.reset_coords(drop=True).reset_index(
        [di for di in da.dims if di in da.coords], drop=True
    )

    # This method works really nicely if all the boundary widths have the same size.
    # This is however not very common. We often have boundary_width with (0,1).
    # I had a ton of trouble accomodating with convoluted logic. The new approach
    # we find the largest boundary width value, and pad every boundary/axis with this max
    # value. As a final step we trim the padded dataset according to the original boundary
    # widths.

    def _expand_boundary_width(padding_width):
        all_widths = []
        for widths in padding_width.values():
            all_widths.extend(list(widths))
        max_width = max(all_widths)
        expanded_padding_width = {
            k: (max_width, max_width) for k in padding_width.keys()
        }
        return expanded_padding_width

    # !!! Hacky. This fixes the cubed-sphere test. I am not quite sure yet what the underlying cause is.
    # TODO: Why is this problem not caught in the padding tests?
    # I am concerned that we are really hardcoding the axes in here. Is there a way to make this more general?
    padding_width = {axname: padding_width.get(axname, (0, 0)) for axname in ["X", "Y"]}
    padding_width_original = {k: v for k, v in padding_width.items()}

    padding_width = _expand_boundary_width(padding_width)

    # The edges of the array which are not connected, need to be padded with the
    # legacy padding. This also ensures that the resulting padded faces result in
    # the same shape. We will pad everything now and then replace the connections.
    # That might however not be the most computational efficient way to do it.

    da_prepadded = _pad_basic(da, grid, padding_width, **kwargs)

    pad_axes = padding_width.keys()

    # i made some alterations to the grid init to store the
    # following info on the grid level. We can discuss if that makes more sense
    # TODO: Do we need that information on the axis still?
    facedim = grid._facedim
    connections = grid._connections

    # TODO: I will have to modify this when using vector partners?
    n_facedim = len(da[facedim])
    da_split = []

    # Iterate over each face and pad accordingly
    for i in range(n_facedim):
        da_single = da_prepadded.isel({facedim: i})
        connection_single = connections[facedim][i]

        da_single_padded = da_single
        for axname in pad_axes:
            # get any connections relevant to the current axis, default to None.
            (left_connection, right_connection) = connection_single.get(
                axname, (None, None)
            )
            _, target_dim = grid.axes[axname]._get_position_name(da_single_padded)
            widths = padding_width[axname]

            for connection, width, is_right in [
                (left_connection, widths[0], False),
                (right_connection, widths[1], True),
            ]:
                if width > 0:
                    if connection:
                        # apply face connection logic #
                        source_face, source_axis, reverse = connection
                        source_da = da_prepadded.isel({facedim: source_face})
                        _, source_dim = grid.axes[source_axis]._get_position_name(
                            source_da
                        )
                        if is_right:
                            # the right connection should be padded with the leftmost index
                            # of the source unless reverse is true
                            if reverse:
                                source_slice_index = slice(
                                    -widths[1] - widths[0], -widths[1]
                                )
                            else:
                                source_slice_index = slice(
                                    widths[0], widths[0] + widths[1]
                                )

                            target_slice_index = slice(0, -widths[1])

                        else:
                            # the left connection should be padded with the rightmost index
                            # of the source unless reverse is true
                            if reverse:
                                source_slice_index = slice(
                                    widths[0], widths[0] + widths[1]
                                )
                            else:
                                source_slice_index = slice(
                                    -widths[1] - widths[0], -widths[1]
                                )

                            target_slice_index = slice(widths[0], None)

                        # after all this logic, start slicing
                        # we get the appropriate slice to add and
                        # remove the previously padded values from the target
                        source_slice = source_da.isel({source_dim: source_slice_index})

                        target_slice = da_single_padded.isel(
                            {target_dim: target_slice_index}
                        )

                        # rename dimension if different
                        if source_dim != target_dim:
                            if not reverse:
                                # Flip along the orthogonal axis
                                source_slice = source_slice.isel(
                                    {target_dim: slice(None, None, -1)}
                                )
                            source_slice = _maybe_swap_dimension_names(
                                source_slice,
                                source_dim,
                                target_dim,
                            )

                        # Apply parallel flip if reverse
                        if reverse:
                            source_slice = source_slice.isel(
                                {target_dim: slice(None, None, -1)}
                            )
                        source_slice = source_slice.squeeze()

                        # assemble the padded array
                        if is_right:
                            concat_list = [target_slice, source_slice]
                        else:
                            concat_list = [source_slice, target_slice]

                        da_single_padded = xr.concat(
                            concat_list,
                            dim=target_dim,
                            coords="minimal",
                        )
                        # TODO: Can we do this with an assignment in xarray? Maybe not important yet.
        da_split.append(da_single_padded)

    da_padded = xr.concat(da_split, dim=facedim)

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
        da_padded, grid, padding_width_original, padding_width
    )


def _pad_basic(da, grid, padding_width, padding="fill", fill_value=0):
    """Implement basic xarray/numpy padding methods"""

    # Always promote the padding to a dict? For now this is the only input type
    # TODO: Refactor this with grid logic that creates a per axis mapping, checks values in grid object and sets defaults.
    padding = grid._parse_axes_kwargs(padding)

    # # set defaults
    for axname, axis in grid.axes.items():
        if axname not in padding.keys():
            # Use default axis boundary for axis unless overridden
            padding[
                axname
            ] = (
                axis.boundary
            )  # TODO: rename the axis property, or are we not keeping this on the axis level?

    da_padded = da

    fill_value = grid._parse_axes_kwargs(fill_value)

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
    boundary: Optional[str] = None,
    # fill_value: Optional[Union[float, Dict[str, float]]] = None,
    **kwargs
):
    """
    Pads the boundary of given arrays along given Axes, according to information in Axes.boundary.

    Parameters
    ----------
    arrays :
        Arrays to pad according to boundary and boundary_width.
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
    """
    # TODO accept a general padding function like numpy.pad does as an argument to boundary

    # TODO: boundary width should not really be optional here I think?
    if not boundary_width:
        raise ValueError("Must provide the widths of the boundaries")

    # TODO rename this globally
    padding = boundary
    padding_width = boundary_width

    padding_args = (da, grid, padding_width)
    # Brute force approach for now. If any axis has connections use
    # face connection method, this dispatches the simple pad stuff internally if needed.
    if any([any(grid.axes[ax]._connections.keys()) for ax in grid.axes]):
        da_padded = _pad_face_connections(*padding_args, padding=padding, **kwargs)
    else:
        # TODO: we need to have a better detection and checking here. For now just pipe everything else into the
        # legacy cases

        # Legacy cases
        da_padded = _pad_basic(*padding_args, padding=padding, **kwargs)

    return da_padded
