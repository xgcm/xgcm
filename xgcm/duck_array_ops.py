"""Compatibility module defining operations on duck numpy-arrays.

Shamelessly copied from xarray."""

from __future__ import division
from __future__ import print_function

import numpy as np

try:
    import dask.array as dsa

    has_dask = True
except ImportError:
    has_dask = False


def _dask_or_eager_func(name, eager_module=np, list_of_args=False, n_array_args=1):
    """Create a function that dispatches to dask for dask array inputs."""
    if has_dask:

        def f(*args, **kwargs):
            dispatch_args = args[0] if list_of_args else args
            if any(isinstance(a, dsa.Array) for a in dispatch_args[:n_array_args]):
                module = dsa
            else:
                module = eager_module
            return getattr(module, name)(*args, **kwargs)

    else:

        def f(data, *args, **kwargs):
            return getattr(eager_module, name)(data, *args, **kwargs)

    return f


insert = _dask_or_eager_func("insert")
take = _dask_or_eager_func("take")
concatenate = _dask_or_eager_func("concatenate", list_of_args=True)


def _apply_boundary_condition(da, dim, left, boundary=None, fill_value=0.0):
    """Supply boundary conditions for  an xarray.DataArray da according along
    the specified dimension. Returns a raw dask or numpy array, depending on
    the underlying data.

    Parameters
    ----------
    da : xarray.DataArray
        The data on which to operate
    dim : str
        Dimenson on which to act
    left : bool
        If `True`, boundary condition is at left (beginning of array).
        If `False`, boundary condition is at the right (end of the array).
    boundary : {'fill', 'extend', 'extrapolate'}
        A flag indicating how the boundary values are determined.

        * 'fill':  All values outside the array set to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.)
        * 'extrapolate': Set values by extrapolating linearly from the two
          points nearest to the edge

    fill_value : float, optional
         The value to use in the boundary condition with `boundary='fill'`.
    """

    if boundary not in ["fill", "extend", "extrapolate"]:
        raise ValueError(
            "`boundary` must be 'fill', 'extend' or "
            "'extrapolate', not %r." % boundary
        )

    axis_num = da.get_axis_num(dim)

    # the shape for the edge array
    shape = list(da.shape)
    shape[axis_num] = 1

    base_array = da.data
    index = slice(0, 1) if left else slice(-1, None)
    edge_array = da.isel(**{dim: index}).data

    use_dask = has_dask and isinstance(base_array, dsa.Array)

    if boundary == "extend":
        boundary_array = edge_array
    elif boundary == "fill":
        args = shape, fill_value
        kwargs = {"dtype": base_array.dtype}
        if use_dask:
            full_func = dsa.full
            kwargs["chunks"] = edge_array.chunks
        else:
            full_func = np.full
        boundary_array = full_func(*args, **kwargs)
    elif boundary == "extrapolate":
        gradient_slice = slice(0, 2) if left else slice(-2, None)
        gradient_sign = -1 if left else 1
        linear_gradient = da.isel(**{dim: gradient_slice}).diff(dim=dim).data
        boundary_array = edge_array + gradient_sign * linear_gradient

    return boundary_array


def _pad_array(da, dim, left=False, boundary=None, fill_value=0.0):
    """
    Pad an xarray.DataArray da according to the boundary conditions along dim.
    Return a raw dask or numpy array, depending on the underlying data.

    Parameters
    ----------
    da : xarray.DataArray
        The data on which to operate
    dim : str
        Dimenson to pad
    left : bool
        If `False`, data is padded at the right (end of the array). If `True`,
        padded at left (beginning of array).
    boundary : {'fill', 'extend'}
        A flag indicating how to handle boundaries:

        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.)

    fill_value : float, optional
         The value to use in the boundary condition with `boundary='fill'`.
    """

    if boundary not in ["fill", "extend"]:
        raise ValueError("`boundary` must be `'fill'` or `'extend'`")

    axis_num = da.get_axis_num(dim)
    shape = list(da.shape)
    shape[axis_num] = 1

    base_array = da.data
    index = slice(0, 1) if left else slice(-1, None)
    edge_array = da.isel(**{dim: index}).data

    use_dask = has_dask and isinstance(base_array, dsa.Array)

    if boundary == "extend":
        boundary_array = edge_array
    elif boundary == "fill":
        args = shape, fill_value
        kwargs = {"dtype": base_array.dtype}
        if use_dask:
            full_func = dsa.full
            kwargs["chunks"] = edge_array.chunks
        else:
            full_func = np.full
        boundary_array = full_func(*args, **kwargs)

    arrays_to_concat = [base_array, boundary_array]
    if left:
        arrays_to_concat.reverse()

    return concatenate(arrays_to_concat, axis=axis_num)
