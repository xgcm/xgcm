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


def _pad_face_connections(da, grid, padding_width, **kwargs):
    # The old logic is too convoluted, lets rewrite this!
    # Step 1: separate each face into a separate dataset
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
        da_single = da.isel({facedim: i}).reset_coords(drop=True)
        connection_single = connections[facedim][i]

        reset_indicies = []
        da_single_padded = da_single
        for axname in pad_axes:
            # get any connections relevant to the current axis, default to None (regular padding otherwise)
            (left_connection, right_connection) = connection_single.get(
                axname, (None, None)
            )
            _, target_dim = grid.axes[axname]._get_position_name(da_single_padded)
            reset_indicies.append(target_dim)
            widths = padding_width[axname]
            # Deal with the 'left' side

            if widths[0] > 0:  # only pad if required by given widths
                if left_connection is None:
                    # fall back to our legacy padding method
                    da_single_padded = _pad_basic(
                        da_single_padded,
                        grid,
                        {axname: (widths[0], 0)},
                    )
                else:
                    source_face, source_axis, reverse = left_connection
                    source_da = da.isel({facedim: source_face}).reset_coords(drop=True)
                    _, source_dim = grid.axes[source_axis]._get_position_name(source_da)
                    # the left connection should be padded with the rightmost index
                    # of the source unless reverse is true
                    souce_slice_index = (
                        slice(0, widths[0]) if reverse else slice(-widths[0], None)
                    )
                    source_slice = source_da.isel({source_dim: souce_slice_index})
                    # rename dimension if different
                    if source_dim != target_dim:
                        source_slice = source_slice.squeeze()
                        source_slice = source_slice.rename({source_dim: "dummy"})
                        source_slice = source_slice.rename({target_dim: source_dim})
                        source_slice = source_slice.rename({"dummy": target_dim})
                    da_single_padded = xr.concat(
                        [source_slice, da_single_padded],
                        dim=target_dim,
                        coords="minimal",
                    )

            # The right side
            if widths[1] > 0:  # only pad if required by given widths
                if right_connection is None:
                    # fall back to our legacy padding method
                    da_single_padded = _pad_basic(
                        da_single_padded, grid, {axname: (0, widths[1])}
                    )
                else:
                    source_face, source_axis, reverse = right_connection
                    source_da = da.isel({facedim: source_face}).reset_coords(drop=True)
                    _, source_dim = grid.axes[source_axis]._get_position_name(source_da)
                    # the right connection should be padded with the leftmost index
                    # of the source unless reverse is true
                    souce_slice_index = (
                        slice(-widths[1], None) if reverse else slice(0, widths[1])
                    )
                    source_slice = source_da.isel({source_dim: souce_slice_index})
                    # rename dimension if different
                    if source_dim != target_dim:
                        source_slice = source_slice.squeeze()
                        source_slice = source_slice.rename({source_dim: "dummy"})
                        source_slice = source_slice.rename({target_dim: source_dim})
                        source_slice = source_slice.rename({"dummy": target_dim})
                    da_single_padded = xr.concat(
                        [da_single_padded, source_slice],
                        coords="minimal",
                    )

        da_split.append(da_single_padded.reset_index(reset_indicies, drop=True))

    da_padded = xr.concat(da_split, dim=facedim)
    return da_padded
    # TODO restore all coordinates? Probably not necessary


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

    # Brute force approach for now. If any axis has connections overwrite the passed padding method and force
    # face connection method
    if any([any(grid.axes[ax]._connections.keys()) for ax in grid.axes]):
        padding = "face_connections"

    padding_args = (da, grid, padding_width)
    # For now we only allow to pass a per axis mapping for our 'legacy' padding. More fancy/custom functions need
    # to handle all steps internally (e.g. looping over axes etc)
    if padding == "face_connections":
        da_padded = _pad_face_connections(*padding_args)
    else:
        # TODO: we need to have a better detection and checking here. For now just pipe everything else into the
        # legacy cases

        # Legacy cases
        da_padded = _pad_basic(*padding_args, padding=padding, **kwargs)

    return da_padded
