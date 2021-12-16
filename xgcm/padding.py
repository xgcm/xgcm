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
}


def _pad_fill(da, boundary_width, fill_value):
    pass


def pad(
    da: Union[xr.DataArray, Dict[str, xr.DataArray]],
    grid: Grid,
    boundary_width: Optional[Dict[str, Tuple[int, int]]] = None,
    boundary: Optional[str] = None,
    fill_value: Optional[Union[float, Dict[str, float]]] = None,
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

    # TODO: Refactor this with grid logic that creates a per axis mapping, checks values in grid object and sets defaults.
    if boundary and isinstance(boundary, str):
        boundary = {ax_name: boundary for ax_name in grid.axes.keys()}
    if fill_value is None:
        fill_value = 0.0
    if isinstance(fill_value, float) or isinstance(fill_value, int):
        # TODO: not sure if we should allow ints?
        fill_value = {ax_name: fill_value for ax_name in grid.axes.keys()}

    new_da = da
    for ax, widths in boundary_width.items():
        axis = grid.axes[ax]
        _, dim = axis._get_position_name(da)

        # Use default boundary for axis unless overridden
        if boundary:
            ax_boundary = boundary[ax]
        else:
            ax_boundary = axis.boundary

        if ax_boundary == "extrapolate":
            # TODO implement extrapolation
            raise NotImplementedError
        elif ax_boundary is None:
            # TODO this is necessary, but also seems inconsistent with the docstring, which says that None = "no boundary condition"
            ax_boundary = "periodic"

        # TODO avoid repeatedly calling xarray pad
        try:
            mode = _XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG[ax_boundary]
        except KeyError:
            raise ValueError(f"{ax_boundary} is not a supported type of boundary")

        if mode == "constant":
            new_da = new_da.pad({dim: widths}, mode, constant_values=fill_value[ax])
        else:
            new_da = new_da.pad({dim: widths}, mode)

    return new_da
