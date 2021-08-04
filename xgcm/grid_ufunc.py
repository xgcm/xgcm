import re

import xarray as xr

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


def _parse_grid_ufunc_signature(signature):
    """
    Parse string signatures for a grid-aware universal function.

    The way this parser works excludes using Axis names that match possible xgcm
    axis positions, i.e. ['center', 'left', 'right', 'inner', 'outer'].

    Arguments
    ---------
    signature : string
        Grid universal function signature. Specifies the xgcm.Axis names and
        positions for each input and output variable, e.g.,

        ``"(X:center)->(X:left)"`` for ``diff_center_to_left(a)`.

    Returns
    -------
    input_axes_names : List[Tuple[str, ...]]
        Input Axes names parsed from the signature
    output_axes_names : List[Tuple[str, ...]]
        Output Axes names parsed from the signature
    input_axes_positions : List[Tuple[str, ...]]
        Input Axes positions parsed from the signature
    output_axes_positions : List[Tuple[str, ...]]
        Output Axes positions parsed from the signature
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

    return in_ax_names, out_ax_names, in_ax_pos, out_ax_pos


def as_grid_ufunc(grid=None, signature="", dask="forbidden"):
    """
    Decorator which turns a numpy ufunc into a "grid-aware ufunc" by wrapping
    it with `apply_grid_ufunc`.

    Parameters
    ----------
    func : callable
        Function to call like `func(*args, **kwargs)` on numpy-like unlabeled
        arrays (`.data`).

        Passed directly on to `xarray.apply_ufunc`.
    grid : xgcm.Grid
    signature : string
        Grid universal function signature. Specifies the xgcm.Axis names and
        positions for each input and output variable, e.g.,

        ``"(X:center)->(X:left)"`` for ``diff_center_to_left(a)`.
    dask : {"forbidden", "allowed", "parallelized"}, default: "forbidden"
        How to handle applying to objects containing lazy data in the form of
        dask arrays. Passed directly on to `xarray.apply_ufunc`.

    Returns
    -------
    grid_ufunc : callable
        Function which consumes and produces xarray objects, whose xgcm Axis
        names and positions must conform to the pattern specified by `signature`.
    """

    def _as_grid_ufunc_decorator(func):
        def _as_grid_ufunc_wrapper(*args, **kwargs):
            return apply_grid_ufunc(
                func, *args, grid=grid, signature=signature, dask=dask, **kwargs
            )

        return _as_grid_ufunc_wrapper

    return _as_grid_ufunc_decorator


def apply_grid_ufunc(func, *args, grid=None, signature="", dask="forbidden", **kwargs):
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
    grid : xgcm.Grid
    signature : string
        Grid universal function signature. Specifies the xgcm.Axis names and
        positions for each input and output variable, e.g.,

        ``"(X:center)->(X:left)"`` for ``diff_center_to_left(a)`.
    dask : {"forbidden", "allowed", "parallelized"}, default: "forbidden"
        How to handle applying to objects containing lazy data in the form of
        dask arrays. Passed directly on to `xarray.apply_ufunc`.

    Returns
    -------
    results
        The result of the call to `xarray.apply_ufunc`, but including the coordinates
        given by the signature, which are read from the grid. Output is either a single
        object or a tuple of such objects.
    """

    # Extract Axes information from signature
    in_ax_names, out_ax_names, in_ax_pos, out_ax_pos = _parse_grid_ufunc_signature(
        signature
    )

    # Check that input args are in correct grid positions
    for i, (arg_ns, arg_ps, arg) in enumerate(zip(in_ax_names, in_ax_pos, args)):
        for n, p in zip(arg_ns, arg_ps):
            try:
                ax_pos = grid.axes[n].coords[p]
            except KeyError:
                raise ValueError(f"Axis position ({n}:{p}) does not exist in grid")

            if ax_pos not in arg.coords:
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
        for arg_ns, arg_ps in zip(in_ax_names, in_ax_pos)
    ]
    out_core_dims = [
        [grid.axes[n].coords[p] for n, p in zip(arg_ns, arg_ps)]
        for arg_ns, arg_ps in zip(out_ax_names, out_ax_pos)
    ]

    all_out_core_dims = set(dim for arg in out_core_dims for dim in arg)

    # Determine expected output dimension sizes from grid._ds
    # Only required when dask='parallelized'
    out_sizes = {out_dim: grid._ds.dims[out_dim] for out_dim in all_out_core_dims}

    # Perform operation via xarray.apply_ufunc
    results = xr.apply_ufunc(
        func,
        *args,
        input_core_dims=in_core_dims,
        output_core_dims=out_core_dims,
        dask=dask,
        **kwargs,
        dask_gufunc_kwargs={"output_sizes": out_sizes},
    )

    # apply_ufunc might return multiple objects
    if not isinstance(results, tuple):
        results = (results,)

    # Restore any coordinates associated with new output dims that are present in grid
    # TODO should this be optional via a `keep_coords` arg?
    results_with_coords = []
    for res, arg_out_core_dims in zip(results, out_core_dims):
        new_core_dim_coords = {
            dim: grid._ds.coords[dim]
            for dim in arg_out_core_dims
            if dim in grid._ds.dims and dim not in res.coords
        }
        res = res.assign_coords(new_core_dim_coords)
        results_with_coords.append(res)

    # Return single results not wrapped in 1-element tuple, like xr.apply_ufunc
    if len(results_with_coords) == 1:
        (results_with_coords,) = results_with_coords

    # TODO handle metrics and boundary?

    return results_with_coords
