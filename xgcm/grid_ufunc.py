import re
import string

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


class GridUFunc:
    """
    Binds a numpy ufunc into a "grid-aware ufunc", meaning that when called ufunc is wrapped by `apply_as_grid_ufunc`.

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
    dask : {"forbidden", "allowed", "parallelized"}, default: "forbidden"
        How to handle applying to objects containing lazy data in the form of
        dask arrays. Passed directly on to `xarray.apply_ufunc`.

    Returns
    -------
    grid_ufunc : callable
        Class which when called consumes and produces xarray objects, whose xgcm Axis
        names and positions must conform to the pattern specified by `signature`.
        Calling function has an additional positional argument `grid`, of type `xgcm.Grid`,
        so that `func`'s new signature is `func(grid, *args, **kwargs)`. This grid
        argument is passed on to `apply_grid_ufunc`.

    See Also
    --------
    as_grid_ufunc
    apply_as_grid_ufunc
    Grid.apply_as_grid_ufunc
    """

    def __init__(self, ufunc, **kwargs):
        self.ufunc = ufunc
        self.signature = kwargs.pop("signature", "")
        self.boundary_width = kwargs.pop("boundary_width", None)
        self.dask = kwargs.pop("dask", "forbidden")
        if kwargs:
            raise TypeError("Unsupported keyword argument(s) provided")

    def __repr__(self):
        return f"GridUFunc(ufunc={self.ufunc}, signature='{self.signature}', boundary_width='{self.boundary_width}', dask='{self.dask})'"

    def __call__(self, grid, *args, boundary=None, **kwargs):
        return apply_as_grid_ufunc(
            self.ufunc,
            *args,
            grid=grid,
            signature=self.signature,
            boundary_width=self.boundary_width,
            boundary=boundary,
            dask=self.dask,
            **kwargs,
        )


def as_grid_ufunc(signature="", boundary_width=None, **kwargs):
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
    dask : {"forbidden", "allowed", "parallelized"}, default: "forbidden"
        How to handle applying to objects containing lazy data in the form of
        dask arrays. Passed directly on to `xarray.apply_ufunc`.

    Returns
    -------
    grid_ufunc : callable
        Function which consumes and produces xarray objects, whose xgcm Axis
        names and positions must conform to the pattern specified by `signature`.
        Function has an additional positional argument `grid`, of type `xgcm.Grid`,
        so that `func`'s new signature is `func(grid, *args, boundary=None, **kwargs)`. This grid
        argument is passed on to `apply_grid_ufunc`.

    See Also
    --------
    apply_as_grid_ufunc
    Grid.apply_as_grid_ufunc
    """
    _allowedkwargs = {
        "dask",
    }
    if kwargs.keys() - _allowedkwargs:
        raise TypeError("Unsupported keyword argument(s) provided")

    def _as_grid_ufunc(ufunc):
        return GridUFunc(
            ufunc, signature=signature, boundary_width=boundary_width, **kwargs
        )

    return _as_grid_ufunc


def apply_as_grid_ufunc(
    func,
    *args,
    grid=None,
    signature="",
    boundary_width=None,
    boundary=None,
    fill_value=None,
    dask="forbidden",
    **kwargs,
):
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
        The xgcm Grid object which contains the various xgcm.Axis described by
        `signature`.
    signature : string
        Grid universal function signature. Specifies the xgcm.Axis names and
        positions for each input and output variable, e.g.,

        ``"(X:center)->(X:left)"`` for ``diff_center_to_left(a)`.
    boundary_width : Dict[str: Tuple[int, int]
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
    """

    if grid is None:
        raise ValueError("Must provide a grid object to describe the Axes")

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

    # Pad arrays according to internal boundary condition information
    if boundary and not boundary_width:
        raise ValueError(
            "To apply a boundary condition you must provide the widths of the boundaries"
        )
    if boundary_width:
        args = grid.pad(
            *args,
            boundary_width=boundary_width,
            boundary=boundary,
            fill_value=fill_value,
        )

    # Determine expected output dimension sizes from grid._ds
    # Only required when dask='parallelized'
    # TODO does padding change this?
    out_sizes = {out_dim: grid._ds.dims[out_dim] for out_dim in all_out_core_dims}

    # TODO Map operation over dask chunks?
    # def mapped_func(*a, **kw):
    #    return map_overlap(func, *a, **kw, depths=boundary_depths, boundary=None, trim=True)

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

    # TODO add option to trim result if not done in ufunc
    # TODO loud warning if ufunc returns array of incorrect size

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

        try:
            res = res.assign_coords(new_core_dim_coords)
        except ValueError as err:
            if boundary_width and str(err).startswith("conflicting sizes"):
                # TODO make this error more informative?
                raise ValueError(
                    f"{str(err)} - does your grid ufunc correctly trim off the same number of elements "
                    f"which were added by padding using boundary_width={boundary_width}?"
                )
            else:
                raise
        results_with_coords.append(res)

    # Return single results not wrapped in 1-element tuple, like xr.apply_ufunc
    if len(results_with_coords) == 1:
        (results_with_coords,) = results_with_coords

    # TODO handle metrics and boundary? Or should that happen in the ufuncs themselves?

    return results_with_coords


_REPLACEMENT_DUMMY_INDEX_NAMES = [f"__{char}" for char in string.ascii_letters]


def _signatures_equivalent(sig1, sig2):
    """
    Axes names in signatures are dummy variables, so an exact string match is not required.

    Our comparison strategy is to instead work through both signatures left to right, replacing all occurrences
    of each dummy index with names drawn from a common list. If after this process the replaced names are not
    identical, the signatures must not be equivalent. Axes positions do have to match exactly.
    """
    sig1_in, sig1_out, _, _ = _parse_grid_ufunc_signature(sig1)
    sig2_in, sig2_out, _, _ = _parse_grid_ufunc_signature(sig2)

    all_unique_sig1_indices = set([i for arg in sig1_in for i in arg])
    all_unique_sig2_indices = set([i for arg in sig2_in for i in arg])

    if len(all_unique_sig1_indices) != len(all_unique_sig2_indices):
        return False

    sig1_replaced = sig1
    sig2_replaced = sig2
    for dummy1, dummy2, common_replacement in zip(
        all_unique_sig1_indices, all_unique_sig2_indices, _REPLACEMENT_DUMMY_INDEX_NAMES
    ):
        sig1_replaced = sig1_replaced.replace(dummy1, common_replacement)
        sig2_replaced = sig2_replaced.replace(dummy2, common_replacement)

    if sig1_replaced == sig2_replaced:
        return True
    else:
        return False
