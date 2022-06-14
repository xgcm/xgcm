import re
import string
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_type_hints,
)

import numpy as np
import xarray as xr

from .padding import pad

if TYPE_CHECKING:
    # Avoids circular references when type checking

    from .grid import Grid

# Modified version of `numpy.lib.function_base._parse_gufunc_signature`
# Modifications:
#   - Specify xgcm.Axis name and "axis positions" instead of numpy axes as (ax_name:ax_pos)

_AXIS_NAME = r"\w+"
_AXIS_POSITION = "(?:center|left|right|inner|outer)"
_AXIS_NAME_POSITION_PAIR = f"{_AXIS_NAME}:{_AXIS_POSITION}"
_AXIS_NAME_POSITION_PAIR_LIST = (
    f"(?:{_AXIS_NAME_POSITION_PAIR}(?:,{_AXIS_NAME_POSITION_PAIR})*,?)*"
)
_ARGUMENT = rf"\({_AXIS_NAME_POSITION_PAIR_LIST}\)"
_ARGUMENT_LIST = f"{_ARGUMENT}(?:,{_ARGUMENT})*"
_SIGNATURE = f"^{_ARGUMENT_LIST}->{_ARGUMENT_LIST}$"


def _maybe_unpack_vector_component(
    data: Union[xr.DataArray, Dict[str, xr.DataArray]]
) -> xr.DataArray:
    if isinstance(data, dict):
        [da] = list(data.values())  # this will raise if more than one element
    else:
        da = data
    return da


def _check_data_input(
    data: Union[xr.DataArray, Dict[str, xr.DataArray]],
    grid: "Grid",
) -> Union[xr.DataArray, Dict[str, xr.DataArray]]:
    """
    Checks for valid data input (either a scalar or a single vector component). Checks types and that vector component axes actually exist
    """
    if data is not None:
        if not isinstance(data, (xr.DataArray, dict)):
            raise TypeError(
                "All data arguments must be either a DataArray or Dictionary"
                f" Got {type(data)}."
            )

        if isinstance(data, dict):
            if len(data.keys()) != 1:
                raise ValueError(
                    "Vector components provided as dictionaries"
                    " should contain exactly one key/value pair."
                    f" Found {len(data)}. Full input:{data}"
                )
            else:
                [key] = list(data.keys())
                value = data[key]
                # Check that axis is in grid object
                if key not in grid.axes:
                    raise ValueError(
                        f"Vector component with unknown axis provided. Grid has axes ({list(grid.axes)}), got  ({key})"
                    )
                # Check that component is dataarray
                if not isinstance(value, xr.DataArray):
                    raise TypeError(
                        f"Dictionary inputs must have a DataArray as value. Got {type(value)}."
                    )

    return data


T_AX_POS_LIST = List[Tuple[str, ...]]


class _GridUFuncSignature:
    """
    Core xGCM Axes and grid positions signature for a given function.

    Based on the signature provided by generalized ufuncs in NumPy, and in xarray.
    """

    in_ax_names: T_AX_POS_LIST
    in_ax_positions: T_AX_POS_LIST
    out_ax_names: T_AX_POS_LIST
    out_ax_positions: T_AX_POS_LIST

    _REPLACEMENT_DUMMY_INDEX_NAMES = [f"__{char}" for char in string.ascii_letters]

    def __init__(
        self,
        in_ax_names: T_AX_POS_LIST,
        in_ax_positions: T_AX_POS_LIST,
        out_ax_names: T_AX_POS_LIST,
        out_ax_positions: T_AX_POS_LIST,
    ):
        """Construct the grid signature directly from its internal attributes."""

        if not in_ax_names or not in_ax_positions:
            raise ValueError(
                "At least one input argument of the Grid UFunc signature must have "
                "axis names and positions"
            )
        else:
            self.in_ax_names = in_ax_names
            self.in_ax_positions = in_ax_positions

        # Can imagine grid ufuncs where outputs have no core dimensions (e.g. result of inner product)
        self.out_ax_names = out_ax_names
        self.out_ax_positions = out_ax_positions

    def __str__(self):
        """The string representation of this signature object"""

        in_arg_sigs = [
            ",".join(f"{ax}:{pos}" for ax, pos in zip(arg_in_names, arg_in_pos))
            for arg_in_names, arg_in_pos in zip(self.in_ax_names, self.in_ax_positions)
        ]
        lhs = ",".join(f"({arg_sig})" for arg_sig in in_arg_sigs)

        out_arg_sigs = [
            ",".join(f"{ax}:{pos}" for ax, pos in zip(arg_out_names, arg_out_pos))
            for arg_out_names, arg_out_pos in zip(
                self.out_ax_names, self.out_ax_positions
            )
        ]
        rhs = ",".join(f"({arg_sig})" for arg_sig in out_arg_sigs)

        return f"{lhs}->{rhs}"

    @classmethod
    def from_string(cls, signature: str) -> "_GridUFuncSignature":
        """Constructs the grid signature from its string representation."""
        (
            in_ax_names,
            in_ax_positions,
            out_ax_names,
            out_ax_positions,
        ) = _parse_signature_from_string(signature)

        return cls(in_ax_names, in_ax_positions, out_ax_names, out_ax_positions)

    @classmethod
    def from_type_hints(cls, hints: Dict[str, Any]) -> "_GridUFuncSignature":
        """
        Constructs the grid signature from the type hints of a function, and returns it ready for parsing.

        Type hints must first be obtained using `typing.get_type_hints(ufunc, include_extras=True)`.
        """
        (
            in_ax_names,
            in_ax_positions,
            out_ax_names,
            out_ax_positions,
        ) = _parse_signature_from_type_hints(hints)

        return cls(in_ax_names, in_ax_positions, out_ax_names, out_ax_positions)

    def equivalent(self, other: "_GridUFuncSignature") -> bool:
        """
        Whether or not two signatures are equivalent.

        Axes names in signatures are dummy variables, so an exact string match is not required.
        Our comparison strategy is to instead work through both signatures left to right, replacing all occurrences
        of each dummy index with names drawn from a common list. If after this process the replaced names are not
        identical, the signatures must not be equivalent. Axes positions do have to match exactly.
        """

        def set_unique_inds(sig_part):
            return set([i for arg in sig_part for i in arg])

        all_unique_sig1_indices = set_unique_inds(self.in_ax_names) | set_unique_inds(
            self.out_ax_names
        )
        all_unique_sig2_indices = set_unique_inds(other.in_ax_names) | set_unique_inds(
            other.out_ax_names
        )

        if len(all_unique_sig1_indices) != len(all_unique_sig2_indices):
            return False

        sig1_replaced = str(self)
        sig2_replaced = str(other)
        for dummy1, dummy2, common_replacement in zip(
            all_unique_sig1_indices,
            all_unique_sig2_indices,
            self._REPLACEMENT_DUMMY_INDEX_NAMES,
        ):
            sig1_replaced = sig1_replaced.replace(dummy1, common_replacement)
            sig2_replaced = sig2_replaced.replace(dummy2, common_replacement)

        return sig1_replaced == sig2_replaced


def _parse_signature_from_string(
    signature: str,
) -> Tuple[T_AX_POS_LIST, T_AX_POS_LIST, T_AX_POS_LIST, T_AX_POS_LIST]:
    """
    Parse string signatures for a grid-aware universal function.

    The way this parser works excludes using Axis names that match possible xgcm
    axis positions, i.e. ['center', 'left', 'right', 'inner', 'outer'].
    """

    signature = signature.replace(" ", "")

    if not re.match(_SIGNATURE, signature):
        raise ValueError(f"Not a valid grid ufunc signature: {signature}")

    in_txt, out_txt = signature.split("->")

    in_ax_names = []
    for arg in re.findall(_ARGUMENT, in_txt):
        # Delete the axis positions so they aren't matched as axis names
        only_names = re.sub(_AXIS_POSITION, "", arg)
        in_ax_names.append(tuple(re.findall(_AXIS_NAME, only_names)))

    out_ax_names = []
    for arg in re.findall(_ARGUMENT, out_txt):
        only_names = re.sub(_AXIS_POSITION, "", arg)
        out_ax_names.append(tuple(re.findall(_AXIS_NAME, only_names)))

    in_ax_pos = [
        tuple(re.findall(_AXIS_POSITION, arg)) for arg in re.findall(_ARGUMENT, in_txt)
    ]
    out_ax_pos = [
        tuple(re.findall(_AXIS_POSITION, arg)) for arg in re.findall(_ARGUMENT, out_txt)
    ]

    return in_ax_names, in_ax_pos, out_ax_names, out_ax_pos


def _parse_signature_from_type_hints(
    hints: Dict[str, Any]
) -> Tuple[T_AX_POS_LIST, T_AX_POS_LIST, T_AX_POS_LIST, T_AX_POS_LIST]:
    """
    Parse signatures from type annotations for a grid-aware universal function.

    The way this parser works excludes using Axis names that match possible xgcm
    axis positions, i.e. ['center', 'left', 'right', 'inner', 'outer'].
    """

    # First do output args
    try:
        return_hint = hints.pop("return")
    except KeyError:
        # TODO does this cause a problem if the output has >1 return arguments none of which have grid positions?
        out_ax_names: T_AX_POS_LIST = [()]
        out_ax_pos: T_AX_POS_LIST = [()]
    else:
        return_hints = _maybe_multiple_return_vals(return_hint)

        return_annotations = [
            hint.__metadata__[0]
            for hint in return_hints
            if hasattr(hint, "__metadata__")
        ]

        out_ax_names = []
        for arg in return_annotations:
            # Delete the axis positions so they aren't matched as axis names
            only_names = re.sub(_AXIS_POSITION, "", arg)
            out_ax_names.append(tuple(re.findall(_AXIS_NAME, only_names)))

        out_ax_pos = [
            tuple(re.findall(_AXIS_POSITION, arg)) for arg in return_annotations
        ]

    # Now do input args
    arg_annotations = [
        hint.__metadata__[0] for hint in hints.values() if hasattr(hint, "__metadata__")
    ]

    # TODO check number of annotations?

    in_ax_names = []
    for arg in arg_annotations:
        # Delete the axis positions so they aren't matched as axis names
        only_names = re.sub(_AXIS_POSITION, "", arg)
        in_ax_names.append(tuple(re.findall(_AXIS_NAME, only_names)))

    in_ax_pos = [tuple(re.findall(_AXIS_POSITION, arg)) for arg in arg_annotations]

    # Do a sanity check before going any further
    str_signature = str(
        _GridUFuncSignature(in_ax_names, in_ax_pos, out_ax_names, out_ax_pos)
    )
    if not re.match(_SIGNATURE, str_signature):
        raise ValueError(f"Not a valid grid ufunc signature: {str_signature}")

    return in_ax_names, in_ax_pos, out_ax_names, out_ax_pos


def _maybe_multiple_return_vals(return_hint):
    """if ufunc returns multiple values (each of which might be annotated) we must extract from Tuple first"""
    return_hints = (
        list(return_hint.__args__) if return_hint._name == "Tuple" else [return_hint]
    )
    return return_hints


class GridUFunc:
    """
    Binds a numpy ufunc into a "grid-aware ufunc", meaning that when called ufunc is wrapped by `apply_as_grid_ufunc`.

    Parameters
    ----------
    ufunc : function
        Function to call like `func(*args, **kwargs)` on numpy-like unlabeled
        arrays (`.data`). Passed directly on to `xarray.apply_ufunc`.
    signature : string
        Grid universal function signature. Specifies the xgcm.Axis names and
        positions for each input and output variable, e.g.,

        ``"(X:center)->(X:left)"`` for ``diff_center_to_left(a)`.
    boundary_width : Dict[str: Tuple[int, int], optional
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
    **kwargs
        Keyword arguments are passed directly onto xarray.apply_ufunc.
        (As such then kwargs should not be xarray data objects, as they will not be subject to
        alignment, nor downcast to numpy-like arrays.)

    Returns
    -------
    grid_ufunc : callable
        Class which when called consumes and produces xarray objects, whose xgcm Axis
        names and positions must conform to the pattern specified by `signature`.
        Calling function has an additional positional argument `grid`, of type `xgcm.Grid`,
        and another additional positional argument `axis`, of type Sequence[Tuple[str]],
        so that `func`'s new signature is `func(grid, *args, axis, **kwargs)`.
        The grid and axis arguments are passed on to `apply_grid_ufunc`.

    See Also
    --------
    as_grid_ufunc
    apply_as_grid_ufunc
    Grid.apply_as_grid_ufunc
    """

    ufunc: Callable
    signature: _GridUFuncSignature
    boundary_width: Optional[Mapping[str, Tuple[int, int]]]
    boundary: Optional[Union[str, Mapping[str, str]]]
    fill_value: Optional[Union[float, Mapping[str, float]]]
    dask: Literal["forbidden", "parallelized", "allowed"]
    map_overlap: bool
    pad_before_func: bool

    def __init__(self, ufunc: Callable, **kwargs):
        self.ufunc = ufunc  # type: ignore  # see mypy issue 2427

        str_sig = kwargs.pop("signature")
        self.signature = self._get_signature_from_str_or_type_hints(ufunc, str_sig)
        self.boundary_width = kwargs.pop("boundary_width", None)
        self.boundary = kwargs.pop("boundary", None)
        self.fill_value = kwargs.pop("fill_value", None)
        self.dask = kwargs.pop("dask", "forbidden")
        self.map_overlap = kwargs.pop("map_overlap", False)
        self.pad_before_func = kwargs.pop("pad_before_func", True)
        if kwargs:
            raise TypeError(
                f"Unsupported keyword argument(s) provided: {list(kwargs.keys())}"
            )

    @staticmethod
    def _get_signature_from_str_or_type_hints(
        ufunc, str_sig: Optional[str]
    ) -> _GridUFuncSignature:
        """Get grid ufunc signature, either from type hints or from string signature kwarg"""

        hints = get_type_hints(ufunc, include_extras=True)

        def _has_annotations(hints):
            try:
                # TODO I want this to be .pop but then I get problems with variable scope
                return_hint = hints["return"]
            except KeyError:
                pass
            else:
                return_hints = _maybe_multiple_return_vals(return_hint)
                if any(hasattr(hint, "__metadata__") for hint in return_hints):
                    return True

            return any(hasattr(hint, "__metadata__") for hint in hints.values())

        if str_sig:
            if _has_annotations(hints):
                raise ValueError(
                    "Must specify axis positions through only one of either type hints or signature kwarg, not both."
                )

            return _GridUFuncSignature.from_string(str_sig)
        else:
            if not _has_annotations(hints):
                raise ValueError(
                    "Must specify axis positions through either type hints or signature kwarg"
                )

            return _GridUFuncSignature.from_type_hints(hints)

    def __repr__(self):
        return (
            f"GridUFunc(ufunc={self.ufunc}, signature='{self.signature}', boundary_width='{self.boundary_width}', "
            f"          dask='{self.dask})', map_overlap={self.map_overlap}, pad_before_func={self.pad_before_func})"
        )

    def __call__(
        self,
        grid: "Grid" = None,
        *args: xr.DataArray,
        axis: Sequence[str],
        **kwargs,
    ):
        boundary = kwargs.pop("boundary", self.boundary)
        dask = kwargs.pop("dask", self.dask)
        map_overlap = kwargs.pop("map_overlap", self.map_overlap)
        pad_before_func = kwargs.pop("pad_before_func", self.pad_before_func)
        return apply_as_grid_ufunc(
            self.ufunc,
            *args,
            axis=axis,
            grid=grid,
            signature=self.signature,
            boundary_width=self.boundary_width,
            boundary=boundary,
            dask=dask,
            map_overlap=map_overlap,
            pad_before_func=pad_before_func,
            **kwargs,
        )


def as_grid_ufunc(
    signature: str = "", boundary_width: Mapping[str, Tuple[int, int]] = None, **kwargs
) -> Callable:
    """
    Decorator which turns a numpy ufunc into a "grid-aware ufunc".

    Parameters
    ----------
    ufunc : callable
        Function to call like `func(*args, **kwargs)` on numpy-like unlabeled
        arrays (`.data`). Passed directly on to `xarray.apply_ufunc`.
    signature : string
        Grid universal function signature. Specifies the xgcm.Axis names and
        positions for each input and output variable, e.g.,

        ``"(X:center)->(X:left)"`` for ``diff_center_to_left(a)`.
    boundary_width : Dict[str: Tuple[int, int], optional
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
    **kwargs
        Keyword arguments are passed directly onto xarray.apply_ufunc.
        (As such then kwargs should not be xarray data objects, as they will not be subject to
        alignment, nor downcast to numpy-like arrays.)

    Returns
    -------
    grid_ufunc : callable
        Function which consumes and produces xarray objects, whose xgcm Axis
        names and positions must conform to the pattern specified by `signature`.
        Function has an additional positional argument `grid`, of type `xgcm.Grid`,
        and another additional positional argument `axis`, of type Sequence[Tuple[str]],
        so that `func`'s new signature is `func(grid, *args, axis, **kwargs)`.
        The grid and axis arguments are passed on to `apply_grid_ufunc`.

    See Also
    --------
    apply_as_grid_ufunc
    Grid.apply_as_grid_ufunc
    """
    _allowedkwargs = {
        "boundary",
        "fill_value",
        "dask",
        "map_overlap",
        "pad_before_func",
    }
    forbidden_kwargs = list(kwargs.keys() - _allowedkwargs)
    if forbidden_kwargs:
        raise TypeError(f"Unsupported keyword argument(s) provided: {forbidden_kwargs}")

    def _as_grid_ufunc(ufunc):
        return GridUFunc(
            ufunc, signature=signature, boundary_width=boundary_width, **kwargs
        )

    return _as_grid_ufunc


def apply_as_grid_ufunc(
    func: Callable,
    *args: Union[xr.DataArray, Dict[str, xr.DataArray]],
    axis: Sequence[Sequence[str]] = None,
    grid: "Grid" = None,
    signature: Union[str, _GridUFuncSignature] = "",
    boundary_width: Mapping[str, Tuple[int, int]] = None,
    boundary: Union[str, Mapping[str, str]] = None,
    fill_value: Union[float, Mapping[str, float]] = None,
    keep_coords: bool = True,
    dask: Literal["forbidden", "parallelized", "allowed"] = "forbidden",
    map_overlap: bool = False,
    pad_before_func: bool = True,
    other_component: Union[
        Dict[str, xr.DataArray], Sequence[Dict[str, xr.DataArray]]
    ] = None,
    **kwargs,
) -> List[Any]:
    """
    Apply a function to the given arguments in a grid-aware manner.

    The relationship between xgcm axes on the input and output are specified by
    `signature`. Wraps xarray.apply_ufunc, but determines the core dimensions
    from the grid and signature passed.

    Parameters
    ----------
    func : function
        Function to call like `func(*args, **kwargs)` on numpy-like unlabeled
        arrays (`.data`).

        Passed directly on to `xarray.apply_ufunc`.
    *args : xarray.DataArray
        One or more input argument to apply the function to. Inputs can be either scalar fields (xr.Dataarray)
        Or vector components (Dictionaries mapping the axis parallel to the vector direction to an xr.Dataarray).
        If vector components are provided, complex grids may require input to `other_component` (see below).
    axis : Sequence[Sequence[str]], optional
        Names of xgcm.Axes on which to act, for each array in args. Multiple axes can be passed as a sequence (e.g. ``['X', 'Y']``).
        Function will be executed over all Axes simultaneously, and each Axis must be present in the Grid.
    grid : xgcm.Grid
        The xgcm Grid object which contains the various xgcm.Axis named in the axis kwarg, with positions matching the
         first half of the `signature`.
    signature : string
        Grid universal function signature. Specifies the relationship between xgcm.Axis positions before and after the
        operation for each input and output variable, e.g.,

        ``signature="(X:center)->(X:left)"`` for ``func=diff_center_to_left(a)`.

        The axis names in the signature are dummy variables, so do not have to present in the Grid. Instead, these dummy
        variables will be identified with the actual named Axes in the `axis` kwarg in order of appearance. For
        instance, ``"(Z:center)->(Z:left)"`` is equivalent to ``"(X:center)->(X:left)"`` - both choices of `signature`
        require only that there is exactly one xgcm.Axis name in `axis` which exists in Grid and starts on position
        `center`.
    boundary_width : Dict[str: Tuple[int, int]
        The widths of the boundaries at the edge of each array.
        Supplied in a mapping of the form {dummy_axis_name: (lower_width, upper_width)}.
        The axis names here are again dummy variables, each of which must be present in the signature.
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
    pad_before_func : bool, optional
        Whether padding should occur before applying func or after it. Default is True.
        (For no padding at all pass `boundary_width=None`).
    other_component : Union[None, Dict[str,xr.DataArray], Sequence[Dict[str,xr.DataArray]]], default: None
        Matching vector component for input provided as dictionary. Needed for complex vector padding.
        For multiple arguments, `other_components` needs to provide one element per input.
    **kwargs
        Keyword arguments are passed directly onto xarray.apply_ufunc.
        (As such then kwargs should not be xarray data objects, as they will not be subject to
        alignment, nor downcast to numpy-like arrays.)

    Returns
    -------
    results
        The result of the call to `xarray.apply_ufunc`, but including the coordinates
        given by the signature, which are read from the grid. Output is either a single
        object or a tuple of such objects.

    See Also
    --------
    as_grid_ufunc
    Grid.apply_as_grid_ufunc
    xarray.apply_ufunc
    """

    if grid is None:
        raise ValueError("Must provide a grid object to describe the Axes")
    # ? Why is this actually an optional input? This causes some mypy issues on pre-commit too.

    # Check data input arguments
    args = _promote_to_sequence_and_check(args, grid)  # type: ignore
    other_component = _promote_to_sequence_and_check(other_component, grid)

    if len(other_component) == 1 and other_component[0] is None:
        # Make sure that the default (None) for other_component is properly broadcasted
        other_component = other_component * len(args)

    if not len(args) == len(other_component):
        raise ValueError(
            "When providing multiple input arguments, `other_component`"
            " needs to provide one dictionary per input."
        )

    if axis is None:
        raise ValueError("Must provide an axis along which to apply the grid ufunc")

    if len(args) != len(axis):
        raise ValueError(
            "Number of entries in `axis` does not match the number of data arguments supplied"
        )

    # Extract Axes information from signature
    if not isinstance(signature, _GridUFuncSignature):
        sig = _GridUFuncSignature.from_string(signature)
    else:
        sig = signature

    dummy_to_real_axes_mapping = _identify_dummy_axes_with_real_axes(
        sig.in_ax_names, axis
    )

    # Determine names of output axes from names in signature
    # TODO what if we need to add a new core dim to the output that does match an input axis? Where do we get the name from?
    out_ax_names = [
        [dummy_to_real_axes_mapping[ax] for ax in arg] for arg in sig.out_ax_names
    ]

    # Check that input args are in correct grid positions
    for i, (arg_ns, arg_ps, arg) in enumerate(zip(axis, sig.in_ax_positions, args)):
        for n, p in zip(arg_ns, arg_ps):
            try:
                ax_pos = grid.axes[n].coords[p]
            except KeyError:
                raise ValueError(f"Axis position ({n}:{p}) does not exist in grid")

            arg = _maybe_unpack_vector_component(arg)
            if ax_pos not in arg.dims:
                raise ValueError(
                    f"Mismatch between signature and input argument {i}: "
                    f"Signature specified data to lie at Axis Position ({n}:{p}), "
                    f"but the corresponding grid coordinate {grid.axes[n].coords[p]} "
                    f"does not appear in argument"
                    f"{arg}"
                )

            # TODO also check that dims are the right length for their stated Axis positions on inputs?

    # Determine core dimensions for apply_ufunc
    in_core_dims = [
        [grid.axes[n].coords[p] for n, p in zip(arg_ns, arg_ps)]
        for arg_ns, arg_ps in zip(axis, sig.in_ax_positions)
    ]
    out_core_dims = [
        [grid.axes[n].coords[p] for n, p in zip(arg_ns, arg_ps)]
        for arg_ns, arg_ps in zip(out_ax_names, sig.out_ax_positions)
    ]

    # TODO allow users to specify new output dtypes
    n_output_vars = len(sig.out_ax_names)
    out_dtypes = [
        _maybe_unpack_vector_component(args[0]).dtype
    ] * n_output_vars  # assume uniformity of dtypes

    # Pad arrays according to boundary condition information
    boundary_width_real_axes = _substitute_dummy_axis_names(
        boundary_width, dummy_to_real_axes_mapping
    )

    # Maybe map function over chunked core dims using dask.array.map_overlap
    if map_overlap:
        # Disallow situations where shifting axis position would cause chunk size to change
        _check_if_length_would_change(sig)

        mapped_func = _map_func_over_core_dims(
            func,
            args,
            grid,
            in_core_dims,
            boundary_width_real_axes,
            out_dtypes,
        )
    else:
        mapped_func = func

    # For most ufuncs we want to pad before applying, but for some (especially cumsum) we must apply then pad
    # TODO could we bind a bunch of these arguments into a namedtuple/dataclass or something to save space?
    if pad_before_func:
        rechunked_padded_args = _pad_then_rechunk(
            args,
            grid,
            in_core_dims,
            boundary_width_real_axes,
            boundary,
            fill_value,
            other_component,
        )
        results = _apply(
            mapped_func,
            rechunked_padded_args,
            grid,
            in_core_dims,
            out_core_dims,
            out_dtypes,
            dask,
            **kwargs,
        )
    else:  # pad after func
        unpadded_results = _apply(
            mapped_func,
            args,
            grid,
            in_core_dims,
            out_core_dims,
            out_dtypes,
            dask,
            **kwargs,
        )
        results = _pad_then_rechunk(
            unpadded_results,
            grid,
            out_core_dims,
            boundary_width_real_axes,
            boundary,
            fill_value,
            other_component,
        )

    # TODO add option to trim result if not done in ufunc

    # Restore any dimension coordinates associated with new output dims that are present in grid
    # Also throws loud warning if ufunc returns array of incorrect size
    results_with_coords = _reattach_coords(results, grid, boundary_width, keep_coords)

    # Return single results not wrapped in 1-element tuple, like xr.apply_ufunc does
    if len(results_with_coords) == 1:
        (results_with_coords,) = results_with_coords

    # TODO handle metrics and boundary? Or should that happen in the ufuncs themselves?

    return results_with_coords


def _apply(
    mapped_func: Callable,
    rechunked_padded_args: Sequence[xr.DataArray],
    grid: "Grid",
    in_core_dims,
    out_core_dims,
    out_dtypes,
    dask,
    **kwargs,
) -> Sequence[xr.DataArray]:

    # Determine expected output dimension sizes from grid._ds
    # Only required when dask='parallelized'
    out_sizes = {
        out_dim: grid._ds.dims[out_dim] for arg in out_core_dims for out_dim in arg
    }

    # Perform operation via xarray.apply_ufunc
    set_in_core_dims = set(d for arg in in_core_dims for d in arg)
    set_out_core_dims = set(d for arg in out_core_dims for d in arg)
    common_dims = set_in_core_dims.union(set_out_core_dims)
    results = xr.apply_ufunc(
        mapped_func,
        *rechunked_padded_args,
        input_core_dims=in_core_dims,
        output_core_dims=out_core_dims,
        exclude_dims=common_dims,
        dask=dask,
        **kwargs,
        dask_gufunc_kwargs={"output_sizes": out_sizes},
        output_dtypes=out_dtypes,
    )

    # apply_ufunc might return multiple objects, but we temporarily promote them for internal consistency
    if not isinstance(results, tuple):
        results = (results,)

    return results


def _substitute_dummy_axis_names(boundary_width, dummy_to_real_axes_mapping):
    if boundary_width:
        # convert dummy axes names in boundary_width to match real names of given axes
        boundary_width_real_axes = {
            dummy_to_real_axes_mapping[ax]: width
            for ax, width in boundary_width.items()
        }
    else:
        # If the boundary_width kwarg was not specified assume that zero padding is required
        boundary_width_real_axes = {
            real_ax: (0, 0) for real_ax in dummy_to_real_axes_mapping.values()
        }
    return boundary_width_real_axes


def _pad_then_rechunk(
    args,
    grid,
    in_core_dims,
    boundary_width_real_axes,
    boundary,
    fill_value,
    other_component,
):

    padded_args = [
        pad(
            a,
            grid=grid,
            boundary_width=boundary_width_real_axes,
            boundary=boundary,
            fill_value=fill_value,
            other_component=oc,
        )
        for a, oc in zip(args, other_component)
    ]

    if any(
        _has_chunked_core_dims(padded_arg, core_dims)
        for padded_arg, core_dims in zip(padded_args, in_core_dims)
    ):
        # merge any lonely chunks on either end created by padding
        rechunked_padded_args = _rechunk_to_merge_in_boundary_chunks(
            padded_args,
            args,
            boundary_width_real_axes,
            grid,
        )
    else:
        rechunked_padded_args = padded_args

    return rechunked_padded_args


def _is_dim_chunked(a, dim):
    # TODO this func can't handle Datasets - it will error if you check multiple variables with different chunking
    return len(a.variable.chunksizes[dim]) > 0


def _has_chunked_core_dims(obj: xr.DataArray, core_dims: Sequence[str]) -> bool:
    # TODO what if only some of the core dimensions are chunked?
    return obj.chunks is not None and any(
        _is_dim_chunked(obj, dim) for dim in core_dims
    )


def _map_func_over_core_dims(
    func,
    original_args,
    grid,
    in_core_dims,
    boundary_width_real_axes,
    out_dtypes,
):
    """
    Map operation over dask chunks along core dimensions.

    Must accept original (unpadded) args in order to get depth of overlap correct.
    """

    from dask.array import map_overlap as dask_map_overlap  # type: ignore

    # Need to transpose the numpy axis arguments to leave core dims at end
    # else they won't match up inside mapped_func after xr.apply_ufunc does its transposition
    transposed_original_args = [
        arg.transpose(..., *in_core_dims[i]) for i, arg in enumerate(original_args)
    ]

    boundary_width_per_numpy_axis = {
        grid.axes[ax_name]._get_axis_dim_num(transposed_original_args[0]): width
        for ax_name, width in boundary_width_real_axes.items()
    }

    single_dim_chunktype = Tuple[int, ...]

    def _dict_to_numbered_axes(
        sizes: Mapping[str, single_dim_chunktype]
    ) -> Tuple[single_dim_chunktype, ...]:
        """This implicitly crystallises the order of the given mapping"""
        return tuple(sizes.values())

    # Our rechunking means dask.map_overlap needs to be explicitly told what chunks output should have
    # But in this case output chunks are the same as input chunks
    # (as we disallowed axis positions for which this is not the case)
    original_chunksizes = [arg.variable.chunksizes for arg in transposed_original_args]
    # TODO first argument only because map_overlap can't handle multiple return values (I think)
    true_chunksizes = original_chunksizes[0]
    # dask.map_overlap needs chunks in terms of axis number, not axis name (i.e. (chunks, ...), not {str: chunks})
    true_chunksizes_per_numpy_axis = _dict_to_numbered_axes(true_chunksizes)

    # (we don't need a separate code path using bare map_blocks if boundary_widths are zero because map_overlap just
    # calls map_blocks automatically in that scenario)
    def mapped_func(*a, **kw):
        return dask_map_overlap(
            func,
            *a,
            **kw,
            depth=boundary_width_per_numpy_axis,
            boundary="none",
            trim=False,
            meta=np.array([], dtype=out_dtypes[0]),
            chunks=true_chunksizes_per_numpy_axis,
        )

    return mapped_func


DISALLOWED_OVERLAP_POSITIONS = ["inner", "outer"]


def _check_if_length_would_change(signature: _GridUFuncSignature):
    """Check if map_overlap can actually handle the complexity of this signature."""

    # TODO this restriction is because dask.array.map_overlap does not currently allow for multiple return arrays
    if len(signature.out_ax_names) > 1:
        raise NotImplementedError(
            "Currently cannot automatically map a ufunc over multiple outputs when the core "
            "dimension is chunked"
        )

    all_ax_positions = set(
        p
        for arg_ps in signature.in_ax_positions + signature.out_ax_positions
        for p in arg_ps
    )
    if any(pos in DISALLOWED_OVERLAP_POSITIONS for pos in all_ax_positions):
        raise NotImplementedError(
            "Cannot chunk along a core dimension for a grid ufunc which has a signature which "
            f"includes one of the axis positions {DISALLOWED_OVERLAP_POSITIONS}"
        )


def _rechunk_to_merge_in_boundary_chunks(
    padded_args: Sequence[xr.DataArray],
    original_args: Sequence[xr.DataArray],
    boundary_width_real_axes: Mapping[str, Tuple[int, int]],
    grid: "Grid",
) -> List[xr.DataArray]:
    """Merges in any small floating chunks at the edges that were created by the padding operation"""

    rechunked_padded_args = []
    for padded_arg, original_arg in zip(padded_args, original_args):

        original_arg_chunks = original_arg.variable.chunksizes
        merged_boundary_chunks = _get_chunk_pattern_for_merging_boundary(
            grid,
            padded_arg,
            original_arg_chunks,
            boundary_width_real_axes,
        )
        rechunked_arg = padded_arg.chunk(merged_boundary_chunks)
        rechunked_padded_args.append(rechunked_arg)

    return rechunked_padded_args


def _get_chunk_pattern_for_merging_boundary(
    grid: "Grid",
    da: xr.DataArray,
    original_chunks: Mapping[str, Tuple[int, ...]],
    boundary_width_real_axes: Mapping[str, Tuple[int, int]],
) -> Mapping[str, Tuple[int, ...]]:
    """Calculates the pattern of chunking needed to merge back in small chunks left on boundaries after padding"""

    # Easier to work with width of boundaries in terms of str dimension names rather than int axis numbers
    boundary_width_dims = {
        _get_dim(grid, da, ax): width for ax, width in boundary_width_real_axes.items()
    }

    new_chunks: Dict[str, Tuple[int, ...]] = {}
    for dim, width in boundary_width_dims.items():
        lower_boundary_width, upper_boundary_width = boundary_width_dims[dim]

        new_chunks_along_dim: Tuple[int, ...]
        if len(original_chunks[dim]) == 1:
            # unpadded array had only one chunk, but padding has meant new array is extended
            original_array_length = original_chunks[dim][0]
            new_chunks_along_dim = (
                lower_boundary_width + original_array_length + upper_boundary_width,
            )
        else:
            first_chunk_width, *other_chunks_widths, last_chunk_width = original_chunks[
                dim
            ]
            new_chunks_along_dim = tuple(
                [
                    first_chunk_width + lower_boundary_width,
                    *other_chunks_widths,
                    last_chunk_width + upper_boundary_width,
                ]
            )
        new_chunks[dim] = new_chunks_along_dim

    return new_chunks


def _get_dim(grid: "Grid", da: xr.DataArray, ax_name: str) -> str:
    ax = grid.axes[ax_name]
    from_pos, dim = ax._get_position_name(da)
    return dim


def _identify_dummy_axes_with_real_axes(
    sig_in_dummy_ax_names: List[Tuple[str, ...]], axis: Sequence[Sequence[str]]
) -> Mapping[str, str]:
    """Create a mapping between the dummy axis names in the signature and the real axis names of the data passed."""

    if len(axis) != len(sig_in_dummy_ax_names):
        raise ValueError(
            "Number of entries in `axis` does not match the number of variables in the input signature"
        )
    for i, (arg_axes, dummy_arg_axes) in enumerate(zip(axis, sig_in_dummy_ax_names)):
        if len(arg_axes) != len(dummy_arg_axes):
            raise ValueError(
                f"Number of Axes in `axis` entry number {i} does not match the number of Axes in that entry in the input signature"
            )

    # We can't just use set because we need these two lists to retain their ordering relative to one another
    unique_dummy_axes = list(
        dict.fromkeys(ax for arg in sig_in_dummy_ax_names for ax in arg)
    )
    unique_real_axes = list(dict.fromkeys(ax for arg in axis for ax in arg))

    if len(unique_dummy_axes) != len(unique_real_axes):
        raise ValueError(
            f"Found {len(unique_dummy_axes)} unique input axes in signature but {len(unique_real_axes)} "
            f"real unique input axes were supplied to the grid ufunc when called"
        )

    return dict(zip(unique_dummy_axes, unique_real_axes))


def _reattach_coords(
    results: Sequence[xr.DataArray], grid: "Grid", boundary_width, keep_coords: bool
) -> List[xr.DataArray]:
    results_with_coords = []
    for res in results:

        # padding strips all coordinates (inlcuding dimension coordinates).
        # Here we centrally restore them from the grid._ds.
        all_matching_coords = {
            coord: da_coord
            for coord, da_coord in grid._ds.coords.items()
            if all(dim in res.dims for dim in da_coord.dims)
        }

        try:
            res = res.assign_coords(all_matching_coords)
        except ValueError as err:
            if boundary_width and str(err).startswith("conflicting sizes"):
                # TODO make this error more informative?
                raise ValueError(
                    f"{str(err)} - does your grid ufunc correctly trim off the same number of elements "
                    f"which were added by padding using boundary_width={boundary_width}?"
                )
            else:
                raise

        if not keep_coords:
            # TODO I don't like the `keep_coords` argument in general and think it should be removed for clarity.
            warnings.warn(
                "The keep_coords keyword argument is being deprecated - in future it will be removed "
                "entirely, and the behaviour will always be that currently given by keep_coords=True.",
                category=DeprecationWarning,
            )

            # Drop any non-dimension coordinates on the output
            non_dim_coords = [coord for coord in res.coords if coord not in res.dims]
            res = res.drop_vars(non_dim_coords)

        results_with_coords.append(res)

    return results_with_coords


def _promote_to_sequence_and_check(
    data: Union[
        xr.DataArray,
        Dict[str, xr.DataArray],
        Sequence[Union[xr.DataArray, Dict[str, xr.DataArray]]],
    ],
    grid: "Grid",
) -> Sequence[Union[xr.DataArray, Dict[str, xr.DataArray]]]:
    if not isinstance(data, Sequence):
        data = [data]
    # Check individual data inputs for validity
    data = [_check_data_input(d, grid) for d in data]
    return data
