import dask as dsk
import numpy as np

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
    assert isinstance(concat_mixed, dsk.array.Array)
