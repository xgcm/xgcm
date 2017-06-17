"""Compatibility module defining operations on duck numpy-arrays.

Shamelessly copied from xarray."""

from __future__ import division
from __future__ import print_function

import numpy as np

try:
    import dask.array as da
    has_dask = True
except ImportError:
    has_dask = False


def _dask_or_eager_func(name, eager_module=np, list_of_args=False,
                        n_array_args=1):
    """Create a function that dispatches to dask for dask array inputs."""
    if has_dask:
        def f(*args, **kwargs):
            dispatch_args = args[0] if list_of_args else args
            if any(isinstance(a, da.Array)
                   for a in dispatch_args[:n_array_args]):
                module = da
            else:
                module = eager_module
            return getattr(module, name)(*args, **kwargs)
    else:
        def f(data, *args, **kwargs):
            return getattr(eager_module, name)(data, *args, **kwargs)
    return f


insert = _dask_or_eager_func('insert')
take = _dask_or_eager_func('take')
concatenate = _dask_or_eager_func('concatenate', list_of_args=True)
