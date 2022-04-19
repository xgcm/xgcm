import numpy as np

from .grid_ufunc import as_grid_ufunc

"""
This module is intended to only contain "grid ufuncs".

(It is however fine to add other non-"grid ufunc" functions to this list, as xgcm.grid._select_grid_ufunc will ignore
anything that does not return an instance of the GridUFunc class.)

If adding a new function to this list, make sure the function name starts with the name of the xgcm.Grid method you
want it to be called from. e.g. `diff_centre_to_left_second_order` will be added to the list of functions to search
through when `xgcm.Grid.diff()` is called. See xgcm.grid_ufunc._select_grid_ufunc for more details.
"""


# TODO can we allow for grouping these definitions into classes? Similar to pytest tests?


# Diff


def diff_forward(a):
    return a[..., 1:] - a[..., :-1]


@as_grid_ufunc(signature="(X:center)->(X:left)", boundary_width={"X": (1, 0)})
def diff_center_to_left(a):
    return diff_forward(a)


@as_grid_ufunc(signature="(X:left)->(X:center)", boundary_width={"X": (0, 1)})
def diff_left_to_center(a):
    return diff_forward(a)


@as_grid_ufunc(signature="(X:center)->(X:right)", boundary_width={"X": (0, 1)})
def diff_center_to_right(a):
    return diff_forward(a)


@as_grid_ufunc(signature="(X:right)->(X:center)", boundary_width={"X": (1, 0)})
def diff_right_to_center(a):
    return diff_forward(a)


@as_grid_ufunc(signature="(X:center)->(X:outer)", boundary_width={"X": (1, 1)})
def diff_center_to_outer(a):
    return diff_forward(a)


# TODO this actually makes the array end up smaller, but boundary_width={"X": (-1, -1)} is not the correct kwarg value.
# TODO rename `boundary_width` argument to `pad_width` to better reflect this possibility?
@as_grid_ufunc(signature="(X:outer)->(X:center)", boundary_width={"X": (0, 0)})
def diff_outer_to_center(a):
    return diff_forward(a)


@as_grid_ufunc(signature="(X:center)->(X:inner)", boundary_width={"X": (0, 0)})
def diff_center_to_inner(a):
    return diff_forward(a)


@as_grid_ufunc(signature="(X:inner)->(X:center)", boundary_width={"X": (1, 1)})
def diff_inner_to_center(a):
    return diff_forward(a)


@as_grid_ufunc(signature="(X:left)->(X:inner)")
def diff_left_to_inner(a):
    raise NotImplementedError


# Interp


def interp_forward(a):
    return (a[..., :-1] + a[..., 1:]) / 2.0


@as_grid_ufunc(signature="(X:center)->(X:left)", boundary_width={"X": (1, 0)})
def interp_center_to_left(a):
    return interp_forward(a)


@as_grid_ufunc(signature="(X:left)->(X:center)", boundary_width={"X": (0, 1)})
def interp_left_to_center(a):
    return interp_forward(a)


@as_grid_ufunc(signature="(X:center)->(X:right)", boundary_width={"X": (0, 1)})
def interp_center_to_right(a):
    return interp_forward(a)


@as_grid_ufunc(signature="(X:right)->(X:center)", boundary_width={"X": (1, 0)})
def interp_right_to_center(a):
    return interp_forward(a)


@as_grid_ufunc(signature="(X:center)->(X:outer)", boundary_width={"X": (1, 1)})
def interp_center_to_outer(a):
    return interp_forward(a)


@as_grid_ufunc(signature="(X:outer)->(X:center)", boundary_width={"X": (0, 0)})
def interp_outer_to_center(a):
    return interp_forward(a)


@as_grid_ufunc(signature="(X:center)->(X:inner)", boundary_width={"X": (0, 0)})
def interp_center_to_inner(a):
    return interp_forward(a)


@as_grid_ufunc(signature="(X:inner)->(X:center)", boundary_width={"X": (1, 1)})
def interp_inner_to_center(a):
    return interp_forward(a)


# Min


def pairwise_forward_min(a):
    left, right = a[..., :-1], a[..., 1:]
    stacked_pairs = np.stack([left, right], axis=-1)
    return np.min(stacked_pairs, axis=-1)


@as_grid_ufunc(signature="(X:center)->(X:left)", boundary_width={"X": (1, 0)})
def min_center_to_left(a):
    return pairwise_forward_min(a)


@as_grid_ufunc(signature="(X:left)->(X:center)", boundary_width={"X": (0, 1)})
def min_left_to_center(a):
    return pairwise_forward_min(a)


@as_grid_ufunc(signature="(X:center)->(X:right)", boundary_width={"X": (0, 1)})
def min_center_to_right(a):
    return pairwise_forward_min(a)


@as_grid_ufunc(signature="(X:right)->(X:center)", boundary_width={"X": (1, 0)})
def min_right_to_center(a):
    return pairwise_forward_min(a)


@as_grid_ufunc(signature="(X:center)->(X:outer)", boundary_width={"X": (1, 1)})
def min_center_to_outer(a):
    return pairwise_forward_min(a)


@as_grid_ufunc(signature="(X:outer)->(X:center)", boundary_width={"X": (0, 0)})
def min_outer_to_center(a):
    return pairwise_forward_min(a)


@as_grid_ufunc(signature="(X:center)->(X:inner)", boundary_width={"X": (0, 0)})
def min_center_to_inner(a):
    return pairwise_forward_min(a)


@as_grid_ufunc(signature="(X:inner)->(X:center)", boundary_width={"X": (1, 1)})
def min_inner_to_center(a):
    return pairwise_forward_min(a)


# Max


def pairwise_forward_max(a):
    left, right = a[..., :-1], a[..., 1:]
    stacked_pairs = np.stack([left, right], axis=-1)
    return np.max(stacked_pairs, axis=-1)


@as_grid_ufunc(signature="(X:center)->(X:left)", boundary_width={"X": (1, 0)})
def max_center_to_left(a):
    return pairwise_forward_max(a)


@as_grid_ufunc(signature="(X:left)->(X:center)", boundary_width={"X": (0, 1)})
def max_left_to_center(a):
    return pairwise_forward_max(a)


@as_grid_ufunc(signature="(X:center)->(X:right)", boundary_width={"X": (0, 1)})
def max_center_to_right(a):
    return pairwise_forward_max(a)


@as_grid_ufunc(signature="(X:right)->(X:center)", boundary_width={"X": (1, 0)})
def max_right_to_center(a):
    return pairwise_forward_max(a)


@as_grid_ufunc(signature="(X:center)->(X:outer)", boundary_width={"X": (1, 1)})
def max_center_to_outer(a):
    return pairwise_forward_max(a)


@as_grid_ufunc(signature="(X:outer)->(X:center)", boundary_width={"X": (0, 0)})
def max_outer_to_center(a):
    return pairwise_forward_max(a)


@as_grid_ufunc(signature="(X:center)->(X:inner)", boundary_width={"X": (0, 0)})
def max_center_to_inner(a):
    return pairwise_forward_max(a)


@as_grid_ufunc(signature="(X:inner)->(X:center)", boundary_width={"X": (1, 1)})
def max_inner_to_center(a):
    return pairwise_forward_max(a)


# Cumsum


@as_grid_ufunc(
    signature="(X:center)->(X:left)",
    boundary_width={"X": (1, 0)},
    fill_value=0,
    pad_before_func=False,
)
def cumsum_center_to_left(a):
    return np.cumsum(a, axis=-1)[..., :-1]


@as_grid_ufunc(signature="(X:left)->(X:center)", boundary_width={"X": (0, 0)})
def cumsum_left_to_center(a):
    return np.cumsum(a, axis=-1)


@as_grid_ufunc(signature="(X:center)->(X:right)", boundary_width={"X": (0, 0)})
def cumsum_center_to_right(a):
    return np.cumsum(a, axis=-1)


@as_grid_ufunc(
    signature="(X:right)->(X:center)",
    boundary_width={"X": (1, 0)},
    fill_value=0,
    pad_before_func=False,
)
def cumsum_right_to_center(a):
    return np.cumsum(a, axis=-1)[..., :-1]


@as_grid_ufunc(
    signature="(X:center)->(X:outer)",
    boundary_width={"X": (1, 0)},
    fill_value=0,
    pad_before_func=False,
)
def cumsum_center_to_outer(a):
    return np.cumsum(a, axis=-1)


@as_grid_ufunc(signature="(X:outer)->(X:center)", boundary_width={"X": (0, 0)})
def cumsum_outer_to_center(a):
    return np.cumsum(a, axis=-1)[..., :-1]


@as_grid_ufunc(signature="(X:center)->(X:inner)", boundary_width={"X": (0, 0)})
def cumsum_center_to_inner(a):
    return np.cumsum(a, axis=-1)[..., :-1]


@as_grid_ufunc(
    signature="(X:inner)->(X:center)",
    boundary_width={"X": (1, 0)},
    fill_value=0,
    pad_before_func=False,
)
def cumsum_inner_to_center(a):
    return np.cumsum(a, axis=-1)
