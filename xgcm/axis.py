from typing import Mapping, Tuple

import xarray as xr

from .padding import _XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG

VALID_POSITION_NAMES = "center|left|right|inner|outer"
FALLBACK_SHIFTS = {
    "center": ("left", "right", "outer", "inner"),
    "left": ("center",),
    "right": ("center",),
    "outer": ("center",),
    "inner": ("center",),
}


class Axis:
    _name: str
    _coords: Mapping[
        str, str
    ]  # TODO give this mapping from positions to dimension names a better name?
    _default_shifts: Mapping[str, str]
    _boundary: str

    def __init__(
        self,
        name,
        ds,
        coords,  # TODO rename to dims
        default_shifts=None,
        boundary="periodic",
    ):
        """
        Create a new Axis object from an input dataset.

        Parameters
        ----------
        name : str
            Name of this Axis.
        ds : xarray.Dataset
            Contains the relevant grid information.
        coords : dict
            Mapping of axis positions to coordinate names
            (e.g. `{'center': 'XC', 'left: 'XG'}`)
        default_shifts : dict, optional
            Default mapping from and to grid positions
            (e.g. `{'center': 'left'}`). Will be inferred if not specified.
        boundary : str or dict
            boundary can either be one of {None, 'fill', 'extend', 'extrapolate', 'periodic'}
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition where
              the difference at the boundary will be zero.)
            * 'periodic' : Wrap arrays around. Equivalent to setting `periodic=True`
            This sets the default value. It can be overriden by specifying the
            boundary kwarg when calling specific methods.
        """

        # check all inputted values are valid here

        for pos, dim in coords.items():
            if pos not in VALID_POSITION_NAMES.split("|"):
                raise ValueError(f"Axis position must be one of {VALID_POSITION_NAMES.split('|')}, but got {pos}")
            if dim not in ds.dims:
                raise ValueError(
                    f"Could not find dimension `{dim}` (for the `{pos}` position on axis `{name}`) in input dataset."
                )
        self._coords = coords

        # set default position shifts
        _default_shifts = {}
        if default_shifts is None:
            default_shifts = {}
        for pos in self.coords:
            # use user-specified value if present
            if pos in default_shifts:
                _default_shifts[pos] = default_shifts[pos]
            else:
                for possible_shift in FALLBACK_SHIFTS[pos]:
                    if possible_shift in self.coords:
                        _default_shifts[pos] = possible_shift
                        break
        self._default_shifts = _default_shifts

        if boundary not in _XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG:
            raise ValueError(f"boundary must be one of {_XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG.keys()}, but got {boundary}")
        self._boundary = boundary

    @property
    def coords(self) -> Mapping[str, str]:
        return self._coords

    @property
    def default_shifts(self) -> Mapping[str, str]:
        return self._default_shifts

    @property
    def boundary(self) -> str:
        return self._boundary

    def _get_position_name(self, da: xr.DataArray) -> Tuple[str, str]:
        """Return the position and name of the axis coordinate in a DataArray."""

        axis_dims = self.coords.values()

        candidates = set(da.dims).intersection(set(axis_dims))

        if len(candidates) == 0:
            raise KeyError(
                f"None of the DataArray's dims {da.dims} were found in axis "
                "coords."
            )
        elif len(candidates) > 1:
            raise KeyError(
                f"DataArray cannot have more than 1 axis dimension, but found {candidates}"
            )
        else:
            for axis_position, axis_dim in self.coords.items():
                if axis_dim in da.dims:
                    return axis_position, axis_dim
            raise

    def _get_axis_dim_num(self, da: xr.DataArray):
        """Return the dimension number of the axis coordinate in a DataArray."""
        _, coord_name = self._get_position_name(da)
        return da.get_axis_num(coord_name)
