import numpy as np

from .grid_ufunc import as_grid_ufunc

"""
This module is intended to only contain "grid ufuncs".

(It is however fine to add other non-"grid ufunc" functions to this list, as xgcm.grid._select_grid_ufunc will ignore
anything 3that does not return an instance of the GridUFunc class.)

If adding a new function to this list, make sure the function name starts with the name of the xgcm.Grid method you
want it to be called from. e.g. `diff_centre_to_left_second_order` will be added to the list of functions to search
through when `xgcm.Grid.diff()` is called. See xgcm.grid_ufunc._select_grid_ufunc for more details.
"""


@as_grid_ufunc(signature="(X:center)->(X:left)")
def diff_center_to_left(a):
    return a - np.roll(a, -1)


@as_grid_ufunc(signature="(X:center)->(X:right)")
def diff_center_to_right(a):
    return np.roll(a, 1) - a


@as_grid_ufunc(signature="(X:center)->(X:outer)")
def diff_center_to_outer(a):
    raise NotImplementedError


@as_grid_ufunc(signature="(X:left)->(X:center)")
def diff_left_to_center(a):
    raise NotImplementedError


@as_grid_ufunc(signature="(X:left)->(X:inner)")
def diff_left_to_inner(a):
    raise NotImplementedError


# TODO fill out all the other ufuncs...
