from .grid_ufunc import as_grid_ufunc

"""
This module is intended to only contain "grid ufuncs".

(It is however fine to add other non-"grid ufunc" functions to this list, as xgcm.grid._select_grid_ufunc will ignore
anything that does not return an instance of the GridUFunc class.)

If adding a new function to this list, make sure the function name starts with the name of the xgcm.Grid method you
want it to be called from. e.g. `diff_centre_to_left_second_order` will be added to the list of functions to search
through when `xgcm.Grid.diff()` is called. See xgcm.grid_ufunc._select_grid_ufunc for more details.
"""


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


# TODO fill out all the other ufuncs...
