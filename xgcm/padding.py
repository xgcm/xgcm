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

# ---------------------------------------------------------------------------
# Bipolar north-fold boundary
# ---------------------------------------------------------------------------
# A "fold" boundary expresses the bipolar north fold of a tripolar ocean grid
# (MOM6, NEMO, MOM5, Oceananigans). Such grids are *tripolar* -- they carry three
# singularities: the ordinary South Pole plus two poles displaced over Arctic
# land. The northern edge of the (single-tile) logical grid folds onto itself
# along the *bipolar seam* -- the line joining those two northern poles -- so the
# fold mirrors the zonal axis about the seam and reverses the sign of vector
# components (a 180-degree pivot about each pole).
#
# Axis convention: the *fold axis is the meridional "Y" axis* (the one whose
# northern edge folds -- you declare the fold boundary on it), and the *seam
# axis is the zonal "X" axis* (the periodic axis it mirrors along, inferred
# automatically as the periodic partner). Everything below is therefore written
# purely in terms of X and Y, e.g.
# ``boundary={"X": "periodic", "Y": {"fold": "corner"}}``.
#
# The value names the *pivot type*: the cell sublattice the pole sits on in each
# direction -- a cell ``center``, or a cell ``edge`` (a face/corner, i.e. offset
# half a cell) -- in the usual ocean-model T/U/V/F shorthand:
#
#   center / T : (X=center, Y=center)   -- pole on a tracer (T) point
#   corner / F : (X=edge,   Y=edge)     -- pole on a cell corner (F) point
#   U          : (X=edge,   Y=center)   -- pole on a U (east/west face) point
#   V          : (X=center, Y=edge)     -- pole on a V (north/south face) point
#
# Only the *north* (upper) edge of the Y axis folds; the south edge uses a
# per-call ``boundary`` if one is given, else the construction-time ``"south"``
# mode (default ``"fill"``).
#
# (Internally the two roles are keyed "seam" (= X) and "fold" (= Y), so the grid
# need not literally name its axes "X"/"Y"; "X" and "Y" here just mean the zonal
# seam axis and the meridional fold axis.)
_PIVOT_ALIASES = {
    # keyed by role: "seam" == the zonal X axis, "fold" == the meridional Y axis
    "center": {"seam": "center", "fold": "center"},  # T  : X=center, Y=center
    "t": {"seam": "center", "fold": "center"},
    "corner": {"seam": "edge", "fold": "edge"},  # F  : X=edge,   Y=edge
    "f": {"seam": "edge", "fold": "edge"},
    "u": {"seam": "edge", "fold": "center"},  # U  : X=edge,   Y=center
    "v": {"seam": "center", "fold": "edge"},  # V  : X=center, Y=edge
}


def _is_fold_boundary(boundary) -> bool:
    """True if a (per-axis) boundary value requests a north fold."""
    return isinstance(boundary, Mapping) and "fold" in boundary


def _position_kind(position: str) -> str:
    """Collapse an xgcm position to the center/edge sublattice it lives on."""
    return "center" if position == "center" else "edge"


# For the seam (X) mirror we reflect each field about the pole using its physical
# cell coordinate, which is robust for every position -- including ``outer`` /
# symmetric memory (length N+1, with a duplicated periodic endpoint) where a
# plain reversal+roll would be off by the wrap. With cell coordinate
# ``x_k = k + offset`` (offset 0 on a left/outer face, 0.5 at a center, 1 on a
# right/inner face), the partner index of output point ``k`` is
# ``(C - k - 2*offset) mod N`` with ``N`` the number of cells (the zonal period)
# and ``C = 0`` for an edge pivot or ``1`` for a center pivot.
_SEAM_POSITION = {
    # position: (2*offset, delta) with N = len(dim) - delta
    "center": (1, 0),
    "left": (0, 0),
    "right": (2, 0),
    "outer": (0, 1),
    "inner": (2, -1),
}


def _seam_partner_indices(position: str, pivot_seam: str, length: int) -> np.ndarray:
    """Source indices mirroring a seam-axis dim about the pole (see note above)."""
    two_offset, delta = _SEAM_POSITION[position]
    n_cells = length - delta
    c = 0 if pivot_seam == "edge" else 1
    k = np.arange(length)
    return (c - k - two_offset) % n_cells


def _parse_fold_boundary(boundary: Mapping) -> Dict:
    """Validate a fold boundary value and return a normalized dict.

    Returns ``{"fold": <alias str or {axis: position} mapping>, "south": <mode>}``.
    The pivot is resolved to X/Y (seam/fold) center-vs-edge roles later (in
    ``_pad_fold``), once the seam (zonal X) axis is known.
    """
    if not _is_fold_boundary(boundary):
        raise ValueError(f"Not a fold boundary value: {boundary!r}")
    extra = set(boundary) - {"fold", "south"}
    if extra:
        raise ValueError(
            f"Unknown keys {sorted(extra)} in fold boundary {dict(boundary)!r}. "
            "Allowed keys are 'fold' (pivot type) and 'south' (south-edge mode)."
        )
    pivot = boundary["fold"]
    if isinstance(pivot, str):
        if pivot.lower() not in _PIVOT_ALIASES:
            raise ValueError(
                f"Unknown fold pivot {pivot!r}. Use one of "
                f"{sorted({k for k in _PIVOT_ALIASES})} "
                "or an explicit {axis: position} mapping."
            )
    elif isinstance(pivot, Mapping):
        # explicit per-axis pivot positions, e.g. {"X": "right", "Y": "center"}
        if not pivot:
            raise ValueError("Explicit fold pivot mapping must not be empty.")
    else:
        raise ValueError(
            f"Fold pivot must be a name ({sorted({k for k in _PIVOT_ALIASES})}) "
            f"or an {{axis: position}} mapping, got {pivot!r}."
        )
    south = boundary.get("south", "fill")
    if south not in _XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG:
        raise ValueError(
            f"Fold 'south' mode must be one of "
            f"{list(_XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG)}, got {south!r}."
        )
    return {"fold": pivot, "south": south}


def _resolve_pivot(pivot, fold_axis: str, seam_axis: str) -> Dict[str, str]:
    """Resolve a pivot spec to ``{'seam': center|edge, 'fold': center|edge}``.

    Here ``'seam'`` is the zonal (X) axis and ``'fold'`` is the meridional (Y)
    axis. An explicit ``{axis: position}`` pivot is keyed by the grid's actual
    axis names (``seam_axis``/``fold_axis``); a named alias (center/corner/U/V)
    maps straight to the X/Y center-edge pair.
    """
    if isinstance(pivot, str):
        return dict(_PIVOT_ALIASES[pivot.lower()])
    # explicit {axis_name: position} mapping
    roles = {}
    for axname, position in pivot.items():
        if axname == fold_axis:
            roles["fold"] = _position_kind(position)
        elif axname == seam_axis:
            roles["seam"] = _position_kind(position)
        else:
            raise ValueError(
                f"Fold pivot axis {axname!r} is neither the fold axis "
                f"{fold_axis!r} nor the seam axis {seam_axis!r}."
            )
    roles.setdefault("seam", "center")
    roles.setdefault("fold", "center")
    return roles


def _maybe_rename_grid_positions(grid, arr_source, arr_target):
    # Checks and renames all dimensions in arr_source to the grid
    # position in arr_target (only if dims are valid axis positions)
    rename_dict = {}
    for di in arr_target.dims:
        # in case the dimension is already in the source, do nothing.
        if di not in arr_source.dims:
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
    other_component: Optional[Dict[str, xr.DataArray]] = None,
):
    facedim = grid._facedim
    connections = grid._face_connections

    # Add explicit checks for mypy
    if connections is None:
        raise ValueError("Grid connections cannot be None")
    if facedim is None:
        raise ValueError("Face dimension cannot be None")

    if isinstance(da, dict):
        isvector = True
        vectoraxis, da = da.popitem()
    else:
        isvector = False

    if isvector:
        # TODO: Using the logic above I could save a bunch of operations below. If we are never swapping axes (_get_all_connection_axes(connections, facedim)) = 1)
        # TODO: We do not need to deal with other components
        # TODO: Need to integrate that choice deeper in the loop\.
        if other_component is not None:
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

                        # The squeeze above drops the length-1 `target_dim` from the
                        # source slice. Restore it unconditionally so the concat below
                        # always sees matching dimensions. Previously this only happened
                        # when `target_dim` carried a coordinate variable; on a grid
                        # whose padded dims have no coordinates the slice stayed 1-D and
                        # `xr.concat(..., join="override")` silently transposed and
                        # clobbered the orthogonal connected edge -- and, because the
                        # axes are iterated in `set` (hash-seed) order, which edge ended
                        # up wrong varied from run to run.
                        if (
                            target_dim not in source_slice.dims
                        ):  # 1-element padding slice
                            source_slice = source_slice.expand_dims([target_dim])

                        # Emulate the way xarray.pad deals with dimension coordinates:
                        # blank them out (only relevant when the dim has a coordinate).
                        if target_dim in target_slice.coords:
                            source_slice = source_slice.assign_coords(
                                {
                                    target_dim: np.full_like(
                                        source_slice[target_dim].data,
                                        np.nan,
                                        dtype=float,  # Needed to properly convert int indicies to nan (super weird)
                                    )
                                }
                            )

                        # Match the source slice's dimension order to the target's.
                        # `expand_dims` prepends the restored `target_dim`, so without
                        # this the two operands can disagree on axis order and
                        # `xr.concat(..., join="override")` produces a transposed/
                        # clobbered result (order-dependent, hence hash-seed dependent).
                        source_slice = source_slice.transpose(*target_slice.dims)

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


def _fold_north_halo(
    da: xr.DataArray,
    grid: Grid,
    fold_axis: str,
    seam_axis: str,
    pivot: Dict[str, str],
    width: int,
    isvector: bool,
) -> xr.DataArray:
    """Build the northern (upper) halo for a single field across a north fold.

    ``fold_axis`` is the meridional (Y) axis whose north edge folds; ``seam_axis``
    is the zonal (X) axis it mirrors along. The halo rows are the interior rows
    just below the fold, mirrored along X about the pole; vector components
    additionally flip sign (the fold is a 180-degree pivot about the pole). See
    the module-level note on fold boundaries for the pivot/offset conventions.
    """
    fold_position, fold_dim = grid.axes[fold_axis]._get_position_name(da)
    seam_position, seam_dim = grid.axes[seam_axis]._get_position_name(da)

    fold_kind = _position_kind(fold_position)

    # --- pick the source rows along the fold (Y) axis -----------------------
    # The topmost row is the (redundant) pole row exactly when the field's
    # Y position coincides with the pivot's; then we start one row in.
    skip = 1 if fold_kind == pivot["fold"] else 0
    # reverse the fold (Y) axis so index 0 is the northern interior edge, then
    # take `width` rows starting `skip` in -> rows ordered north-to-further-south,
    # i.e. the correct order to concatenate above the interior.
    reversed_fold = da.isel({fold_dim: slice(None, None, -1)})
    # The fold can only mirror rows that exist below it: after dropping the
    # `skip` (redundant pole) rows there are `n_interior` interior rows to
    # supply the halo. A wider request would silently clamp `isel` and yield a
    # too-short array, so fail loudly instead.
    n_interior = reversed_fold.sizes[fold_dim] - skip
    if width > n_interior:
        raise ValueError(
            f"North-fold halo width {width} requested on fold axis "
            f"{fold_axis!r} exceeds the {n_interior} interior row(s) available "
            f"to mirror along {fold_dim!r} (grid length {reversed_fold.sizes[fold_dim]}"
            f"{f', minus {skip} redundant pole row' if skip else ''}). "
            "The fold can supply at most that many halo rows."
        )
    halo = reversed_fold.isel({fold_dim: slice(skip, skip + width)})

    # --- mirror along the seam (X) axis about the pole ----------------------
    # gather each output column from its mirror partner (handles every position,
    # including outer/symmetric-memory grids).
    idx = _seam_partner_indices(seam_position, pivot["seam"], halo.sizes[seam_dim])
    if idx.max() >= halo.sizes[seam_dim]:
        # e.g. an `inner` seam position under a center-type pivot: reflecting the
        # inner sublattice about a cell-center pole lands on an excluded endpoint,
        # so there is no mirror partner.
        raise NotImplementedError(
            f"A {seam_position!r} seam position is incompatible with a "
            f"center-type fold pivot (seam role {pivot['seam']!r}): the mirror "
            "about a cell-center pole has no partner on this sublattice. Use an "
            "edge-type pivot, or a center/left/right/outer seam position."
        )
    halo = halo.isel({seam_dim: idx})

    if isvector:
        # both horizontal components reverse sign across the fold
        halo = -halo

    return halo


def _pad_fold(
    da: Union[xr.DataArray, Dict[str, xr.DataArray]],
    grid: Grid,
    padding_width: Dict[str, Tuple[int, int]],
    padding: Dict[str, str],
    fill_value: Dict[str, float],
):
    """Pad a single-tile grid that has one or more north-fold boundaries.

    The *north* (upper) edge of each fold (Y) axis is filled from the mirrored
    interior (`_fold_north_halo`); the south edge and every non-fold axis use the
    ordinary `_pad_basic` machinery.
    """
    if isinstance(da, dict):
        isvector = True
        _, da = da.popitem()
    else:
        isvector = False

    fold_axes = {
        ax for ax in padding_width if ax in grid._folds and padding_width[ax][1] > 0
    }

    # 1. Attach the northern fold halo for each fold axis (computed from the
    #    unpadded interior so the seam mirror sees the full periodic row).
    for fax in fold_axes:
        info = grid._folds[fax]
        pivot = _resolve_pivot(info["pivot"], fax, info["seam_axis"])
        _, fold_dim = grid.axes[fax]._get_position_name(da)
        width = padding_width[fax][1]
        halo = _fold_north_halo(
            da, grid, fax, info["seam_axis"], pivot, width, isvector
        )
        halo = halo.drop_vars([co for co in halo.coords])
        da = xr.concat(
            [da, halo],
            dim=fold_dim,
            coords="minimal",
            compat="override",
            join="override",
        )

    # 2. Basic-pad everything else: non-fold axes at full width, and the south
    #    edge of each fold axis with its `south` mode (north already folded).
    basic_width = {}
    basic_padding = {}
    for ax, widths in padding_width.items():
        if ax in grid._folds:
            # the north edge is folded (or not padded). The south edge takes an
            # explicit per-call boundary if the user gave one, else the fold's
            # construction-time `south` mode. (A fold axis's default `padding`
            # entry is the fold-spec dict, which must never reach `_pad_basic`;
            # only a plain string is a genuine per-call override.)
            per_call = padding[ax]
            basic_width[ax] = (widths[0], 0)
            basic_padding[ax] = (
                per_call if isinstance(per_call, str) else grid._folds[ax]["south"]
            )
        else:
            basic_width[ax] = widths
            basic_padding[ax] = padding[ax]

    return _pad_basic(da, grid, basic_width, basic_padding, fill_value)


def pad(
    data: Union[xr.DataArray, Dict[str, xr.DataArray]],
    grid: Grid,
    boundary_width: Optional[Dict[str, Tuple[int, int]]],
    boundary: Optional[Union[str, Mapping[str, str]]] = None,
    fill_value: Optional[Union[float, Mapping[str, float]]] = None,
    other_component: Optional[Dict[str, xr.DataArray]] = None,
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
    boundary : {None, 'fill', 'extend', 'periodic', dict}, optional
        A flag indicating how to handle boundaries:

        * None:  Do not apply any boundary conditions. Raise an error if
            boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
            (i.e. a Dirichlet boundary condition.)
        * 'extend': Set values outside the array to the nearest array
            value. (i.e. a limited form of Neumann boundary condition.)
        * 'periodic': Set values by wrapping around the array on the specified
            axes. (i.e. a periodic boundary condition.)
        Optionally a dict mapping axis name to seperate values for each axis
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

    # Always promote the padding/fill_value to a dict of form {ax: kwarg}.
    padding = grid._complete_user_kwargs_using_axis_defaults(padding, "boundary")
    fill_value = grid._complete_user_kwargs_using_axis_defaults(
        fill_value, "fill_value"
    )

    # Exit without padding if all widths are zero
    if padding_width is None or all(
        width == (0, 0) for width in padding_width.values()
    ):
        # TODO: Think about case when boundary is specified but boundary_width is None or (0,0).
        # TODO: No padding would occur in that situation. Should we warn the user?
        return data

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
    if grid._face_connections is not None:
        da_padded = _pad_face_connections(
            data,
            grid,
            padding_width,
            padding,
            fill_value,
            other_component=other_component,
        )
    elif getattr(grid, "_folds", None) and any(
        ax in grid._folds for ax in padding_width
    ):
        da_padded = _pad_fold(data, grid, padding_width, padding, fill_value)
    else:
        da_padded = _pad_basic(data, grid, padding_width, padding, fill_value)  # type: ignore

    return da_padded
