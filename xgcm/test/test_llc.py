"""
Integration tests for MITgcm LLC grid.
"""

import xarray as xr
import numpy as np
import pytest

from xgcm.grid import Grid


@pytest.fixture(scope="module",)
def llc_ds(request):
    shape = (13, 90, 90)
    nface, ny, nx = shape
    ds = xr.Dataset({'tracer': (('face', 'j', 'i'), np.random.rand(*shape)),
                     'u': (('face', 'j', 'i_g'), np.random.rand(*shape)),
                     'v': (('face', 'j_g', 'i'), np.random.rand(*shape))},
                    {'face': ('face', np.arange(nface)),
                     'j': ('j', np.arange(nx)),
                     'j_g': ('j_g', np.arange(nx)),
                     'i': ('i', np.arange(nx)),
                     'i_g': ('i_g', np.arange(nx))})
    return ds


LLC_COORDS = {'X': {'center': 'i', 'left': 'i_g'},
              'Y': {'center': 'j', 'left': 'j_g'}}

LLC_FACE_CONNECTIONS = {'face': {0: {'X': ((12, 'Y', False), (3, 'X', False)),
                                     'Y': (None, (1, 'Y', False))},
                                 1: {'X': ((11, 'Y', False), (4, 'X', False)),
                                     'Y': ((0, 'Y', False), (2, 'Y', False))},
                                 2: {'X': ((10, 'Y', False), (5, 'X', False)),
                                     'Y': ((1, 'Y', False), (6, 'X', False))},
                                 3: {'X': ((0, 'X', False), (9, 'Y', False)),
                                     'Y': (None, (4, 'Y', False))},
                                 4: {'X': ((1, 'X', False), (8, 'Y', False)),
                                     'Y': ((3, 'Y', False), (5, 'Y', False))},
                                 5: {'X': ((2, 'X', False), (7, 'Y', False)),
                                     'Y': ((4, 'Y', False), (6, 'Y', False))},
                                 6: {'X': ((2, 'Y', False), (7, 'X', False)),
                                     'Y': ((5, 'Y', False), (10, 'X', False))},
                                 7: {'X': ((6, 'X', False), (8, 'X', False)),
                                     'Y': ((5, 'X', False), (10, 'Y', False))},
                                 8: {'X': ((7, 'X', False), (9, 'X', False)),
                                     'Y': ((4, 'X', False), (11, 'Y', False))},
                                 9: {'X': ((8, 'X', False), None),
                                     'Y': ((3, 'X', False), (12, 'Y', False))},
                                 10: {'X': ((6, 'Y', False), (11, 'X', False)),
                                      'Y': ((7, 'Y', False), (2, 'X', False))},
                                 11: {'X': ((10, 'X', False), (12, 'X', False)),
                                      'Y': ((8, 'Y', False), (1, 'X', False))},
                                 12: {'X': ((11, 'X', False), None),
                                      'Y': ((9, 'Y', False), (0, 'X', False))}}}


@pytest.fixture(scope="module")
def llc_grid(llc_ds):
    grid = Grid(llc_ds, coords=LLC_COORDS,
                face_connections=LLC_FACE_CONNECTIONS)
    return grid


def test_llc_grid(llc_ds, llc_grid):
    assert llc_grid
    pass


def test_tracer_diff_interp(llc_ds, llc_grid):
    dt_x = llc_grid.diff(llc_ds.tracer, 'X', boundary='fill')
    it_x = llc_grid.interp(llc_ds.tracer, 'X', boundary='fill')
    assert dt_x.dims == ('face', 'j', 'i_g')
    assert it_x.dims == ('face', 'j', 'i_g')
    dt_y = llc_grid.diff(llc_ds.tracer, 'Y', boundary='fill')
    it_y = llc_grid.interp(llc_ds.tracer, 'Y', boundary='fill')
    assert dt_y.dims == ('face', 'j_g', 'i')
    assert it_y.dims == ('face', 'j_g', 'i')


def test_vector_normal_diff_interp(llc_ds, llc_grid):
    du = llc_grid.diff_2d_vector({'X': llc_ds.u, 'Y': llc_ds.v},
                                 boundary='fill')


def test_vector_tangent_diff_interp(llc_ds, llc_grid):
    du = llc_grid.diff_2d_vector({'X': llc_ds.v, 'Y': llc_ds.u},
                                 boundary='fill')
