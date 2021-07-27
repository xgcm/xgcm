import re

import xarray as xr

# TODO Handle output dimensions of fixed size?

# Modified version of `numpy.lib.function_base._parse_gufunc_signature`
# Modifications:
#   - Specify xgcm.Axis and "axis positions" instead of numpy axes as (dim:ax_pos)
_DIMENSION_NAME = r"\w+"
_AXIS_POSITION = "(?:center|left|right|inner|outer)"
_DIMENSION_AXIS_PAIR = "{0:}:{1:}".format(_DIMENSION_NAME, _AXIS_POSITION)
_DIMENSION_AXIS_PAIR_LIST = "(?:{0:}(?:,{0:})*,?)*".format(_DIMENSION_AXIS_PAIR)
_ARGUMENT = r"\({0:}\)".format(_DIMENSION_AXIS_PAIR_LIST)
_ARGUMENT_LIST = "{0:}(?:,{0:})*".format(_ARGUMENT)
_SIGNATURE = "^{0:}->{0:}$".format(_ARGUMENT_LIST)


def _parse_grid_ufunc_signature(signature):
    """
    Parse string signatures for a grid-aware universal function.

    The way this parser works excludes using Axis names that match possible xgcm
    axis positions, i.e. ['center', 'left', 'right', 'inner', 'outer'].

    Arguments
    ---------
    signature : string
        Generalized universal function signature, e.g., ``"(X:center)->(X:left)"``
        for ``diff_center_to_left(a)`.

    Returns
    -------
    Tuple of input and output core dimensions parsed from the signature, each
    of the form List[Tuple[str, ...]].
    """

    signature = signature.replace(" ", "")

    if not re.match(_SIGNATURE, signature):
        raise ValueError(f"Not a valid grid ufunc signature: {signature}")

    in_txt, out_txt = signature.split("->")

    in_core_dims = []
    for arg in re.findall(_ARGUMENT, in_txt):
        # Delete the axis positions so they aren't matched as dimension names
        only_dims = re.sub(_AXIS_POSITION, "", arg)
        in_core_dims.append(tuple(re.findall(_DIMENSION_NAME, only_dims)))

    out_core_dims = []
    for arg in re.findall(_ARGUMENT, out_txt):
        only_dims = re.sub(_AXIS_POSITION, "", arg)
        out_core_dims.append(tuple(re.findall(_DIMENSION_NAME, only_dims)))

    in_ax_pos = [
        tuple(re.findall(_AXIS_POSITION, arg)) for arg in re.findall(_ARGUMENT, in_txt)
    ]
    out_ax_pos = [
        tuple(re.findall(_AXIS_POSITION, arg)) for arg in re.findall(_ARGUMENT, out_txt)
    ]

    return in_core_dims, out_core_dims, in_ax_pos, out_ax_pos


def as_grid_ufunc(signature):
    """
    Decorator version of `grid_ufunc`.

    Parameters
    ----------
    signature

    Returns
    -------

    """

    def _as_grid_ufunc(func, *args, **kwargs):
        return grid_ufunc(func, signature=signature, *args, **kwargs)

    return _as_grid_ufunc


def grid_ufunc(func, signature, *args, **kwargs):
    """
    Turns a numpy ufunc into a "grid-aware ufunc", where the relationship between
    xgcm axes on the input and output are specified by `signature`.


    Parameters
    ----------
    func
    signature

    Returns
    -------

    """

    # Translate signature
    in_core, out_core, in_ax_pos, out_ax_pos = _parse_grid_ufunc_signature(signature)

    _validate_positions(args, in_ax_pos)

    # perform operation via xarray.apply_ufunc
    result = xr.apply_ufunc(
        func,
        *args,
        input_core_dims=in_core,
        output_core_dims=out_core,
        **kwargs,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": "out_sizes"},
    )
    # how to determine expected output sizes - not present in signature?

    # handle metrics and boundary?

    return result


def _validate_positions(args, input_axis_pos):
    for arg, ax_pos in zip(args, input_axis_pos):
        # TODO actually check that the args have the expected xgcm axis positions
        ...
