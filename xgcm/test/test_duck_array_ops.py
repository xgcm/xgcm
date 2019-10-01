from __future__ import print_function

# experimental fix for inconsistent numpy behavior
import os

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "1"
import numpy as np
import dask as dsk
from xgcm.duck_array_ops import concatenate


def test_concatenate():
    a = np.array([1, 2, 3])
    b = np.array([10])
    a_dask = dsk.array.from_array(a, chunks=1)
    b_dask = dsk.array.from_array(b, chunks=1)
    concat = concatenate([a, b], axis=0)
    concat_dask = concatenate([a_dask, b_dask], axis=0)
    concat_mixed = concatenate([a, b_dask], axis=0)
    assert isinstance(concat, np.ndarray)
    assert isinstance(concat_dask, dsk.array.Array)
    # assert isinstance(concat_mixed, np.ndarray)
    # # the resulting mixed array is a dask array, not np. due to changes in numpy?
    assert isinstance(concat_mixed, dsk.array.Array)
