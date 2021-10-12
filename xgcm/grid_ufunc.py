import re
import string

import numpy as np
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
    dask : {"forbidden", "allowed", "parallelized"}, default: "forbidden"
        How to handle applying to objects containing lazy data in the form of
        dask arrays. Passed directly on to `xarray.apply_ufunc`.
    map_overlap : bool, optional
        Whether or not to automatically apply the function along chunked core dimensions using dask.array.map_overlap.
        Default is False. If True, will need to be accompanied by dask='allowed'.

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

    def __init__(self, ufunc, **kwargs):
        self.ufunc = ufunc
        self.signature = kwargs.pop("signature", "")
        self.boundary_width = kwargs.pop("boundary_width", None)
        self.dask = kwargs.pop("dask", "forbidden")
        self.map_overlap = kwargs.pop("map_overlap", False)
        if kwargs:
            raise TypeError("Unsupported keyword argument(s) provided")

    def __repr__(self):
        return (
            f"GridUFunc(ufunc={self.ufunc}, signature='{self.signature}', boundary_width='{self.boundary_width}', "
            f"          dask='{self.dask})', map_overlap={self.map_overlap})"
        )

    def __call__(self, grid, *args, axis, boundary=None, **kwargs):
        dask = kwargs.pop("dask", self.dask)
        map_overlap = kwargs.pop("map_overlap", self.map_overlap)
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
    map_overlap : bool, optional
        Whether or not to automatically apply the function along chunked core dimensions using dask.array.map_overlap.
        Default is False. If True, will need to be accompanied by dask='allowed'.

    Returns
    -------
    grid_ufunc : callable
        Function which consumes and produces xarray objects, whose xgcm Axis
        names and positions must conform to the pattern specified by `signature`.
        Function has an additional positional argument `grid`, of type `xgcm.Grid`,
        and another additional positional argument `axis`, of type Sequence[Tuple[str]],
        so that `func`'s new signature is `func(grid, *args, axis, boundary=None, **kwargs)`.
        The grid and axis arguments are passed on to `apply_grid_ufunc`.

    See Also
    --------
    apply_as_grid_ufunc
    Grid.apply_as_grid_ufunc
    """
    _allowedkwargs = {
        "dask",
        "map_overlap",
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
    axis,
    grid=None,
    signature="",
    boundary_width=None,
    boundary=None,
    fill_value=None,
    dask="forbidden",
    map_overlap=False,
    **kwargs,
):
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
    args : xarray.DataArray or xarray.Dataset
        One or more xarray objects to apply the function to.
    axis : Sequence[Tuple[str]]
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

    if any(not isinstance(arg, (xr.DataArray, xr.Dataset)) for arg in args):
        raise TypeError("All data arguments must be of type DataArray or Dataset")

    if len(args) != len(axis):
        raise ValueError(
            "Number of entries in `axis` does not match the number of data arguments supplied"
        )

    # Extract Axes information from signature
    (
        in_dummy_ax_names,
        out_dummy_ax_names,
        in_ax_pos,
        out_ax_pos,
    ) = _parse_grid_ufunc_signature(signature)

    dummy_to_real_axes_mapping = _identify_dummy_axes_with_real_axes(
        in_dummy_ax_names, axis
    )

    # Determine names of output axes from names in signature
    # TODO what if we need to add a new core dim to the output that does match an input axis? Where do we get the name from?
    out_ax_names = [
        [dummy_to_real_axes_mapping[ax] for ax in arg] for arg in out_dummy_ax_names
    ]

    # Check that input args are in correct grid positions
    for i, (arg_ns, arg_ps, arg) in enumerate(zip(axis, in_ax_pos, args)):
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
        for arg_ns, arg_ps in zip(axis, in_ax_pos)
    ]
    out_core_dims = [
        [grid.axes[n].coords[p] for n, p in zip(arg_ns, arg_ps)]
        for arg_ns, arg_ps in zip(out_ax_names, out_ax_pos)
    ]

    # TODO allow users to specify new output dtypes
    out_dtypes = [a.dtype for a in args]

    # Pad arrays according to boundary condition information
    if boundary and not boundary_width:
        raise ValueError(
            "To apply a boundary condition you must provide the widths of the boundaries"
        )
    if boundary_width:

        # convert dummy axes names in boundary_width to match real names of given axes
        boundary_width_real_axes = {
            dummy_to_real_axes_mapping[ax]: width
            for ax, width in boundary_width.items()
        }

        original_chunks = args[0].chunks

        padded_args = grid.pad(
            *args,
            boundary_width=boundary_width_real_axes,
            boundary=boundary,
            fill_value=fill_value,
        )
    else:
        padded_args = args

    if (
        any(
            _has_chunked_core_dims(arg, core_dims)
            for arg, core_dims in zip(args, in_core_dims)
        )
        and map_overlap
    ):
        # map operation over dask chunks along core dimensions
        from dask.array import map_overlap

        # merge any lonely chunks from padding
        da = args[0]
        merged_boundary_chunks = _get_chunk_pattern_for_merging_boundary(
            grid,
            da,
            _chunks_as_dict(da),
            boundary_width_real_axes,
        )
        rechunked_padded_args = [
            arg.chunk(merged_boundary_chunks) for arg in padded_args
        ]
        # TODO the true chunks will differ from the original chunks if the ufunc changes the array's chunking or size
        true_chunks = original_chunks

        boundary_width_numpy_axes = {
            grid.axes[ax_name]._get_axis_dim_num(da): width
            for ax_name, width in boundary_width_real_axes.items()
        }

        # (we don't need a separate code path using bare map_blocks if boundary_widths are zero because map_overlap just
        # calls map_blocks automatically in that scenario)
        def mapped_func(*a, **kw):
            return map_overlap(
                func,
                *a,
                **kw,
                depth=boundary_width_numpy_axes,
                boundary="none",
                trim=False,
                meta=np.array(
                    [], dtype=out_dtypes[0]
                ),  # TODO can map_overlap handle multiple return values?
                chunks=true_chunks,
            )

    else:
        rechunked_padded_args = padded_args
        mapped_func = func

    # Determine expected output dimension sizes from grid._ds
    # Only required when dask='parallelized'
    out_sizes = {
        out_dim: grid._ds.dims[out_dim] for arg in out_core_dims for out_dim in arg
    }

    # Perform operation via xarray.apply_ufunc
    results = xr.apply_ufunc(
        mapped_func,
        *rechunked_padded_args,
        input_core_dims=in_core_dims,
        output_core_dims=out_core_dims,
        dask=dask,
        **kwargs,
        dask_gufunc_kwargs={"output_sizes": out_sizes},
        output_dtypes=out_dtypes,
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

    # Return single results not wrapped in 1-element tuple, like xr.apply_ufunc does
    if len(results_with_coords) == 1:
        (results_with_coords,) = results_with_coords

    # TODO handle metrics and boundary? Or should that happen in the ufuncs themselves?

    return results_with_coords


def _has_chunked_core_dims(da, core_dims):
    for core_dim in core_dims:
        axis_num = list(da.dims).index(core_dim)

        if da.chunks is None:
            continue
        else:
            # TODO what if only some of the core dimensions are chunked?
            core_dim_chunks = da.chunks[axis_num]
            if len(core_dim_chunks) > 1:
                return True
    return False


def _chunks_as_dict(data_obj):
    """Need to special-case DataArrays due to inconsistency of xarray issue #5843"""

    # TODO generalize to multiple arguments!
    if isinstance(data_obj, xr.DataArray):
        chunks_dict = dict(zip(data_obj.dims, data_obj.chunks))
    else:  # must be a Dataset
        chunks_dict = data_obj.chunks

    return chunks_dict


def _get_chunk_pattern_for_merging_boundary(
    grid, da, original_chunks, boundary_width_real_axes
):
    """Calculates the pattern of chunking needed to merge back in small chunks left on boundaries after padding"""

    # TODO refactor this ugly logic to live elsewhere
    boundary_width_dims = {
        _get_dim(grid, da, ax): width for ax, width in boundary_width_real_axes.items()
    }

    # TODO generalise to multiple data arguments
    new_chunks = {}
    for dim, width in boundary_width_dims.items():
        lower_boundary_width, upper_boundary_width = boundary_width_dims[dim]
        first_chunk_width, *other_chunks_widths, last_chunk_width = original_chunks[dim]
        new_chunks_along_dim = tuple(
            [
                first_chunk_width + lower_boundary_width,
                *other_chunks_widths,
                last_chunk_width + upper_boundary_width,
            ]
        )
        new_chunks[dim] = new_chunks_along_dim

    return new_chunks


def _get_dim(grid, da, ax_name):
    ax = grid.axes[ax_name]
    from_pos, dim = ax._get_position_name(da)
    return dim


def _identify_dummy_axes_with_real_axes(sig_in_dummy_ax_names, axis):
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
