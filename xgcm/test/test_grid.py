from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np

from xgcm.grid import Grid

from . datasets import all_datasets, nonperiodic_1d

def test_create_grid(all_datasets):
    grid = Grid(all_datasets)

def test_grid_repr(all_datasets):
    grid = Grid(all_datasets)
    r = repr(grid).split('\n')
    assert r[0] == "<xgcm.Grid>"
    # all datasets should have at least an X axis
    assert r[1].startswith('X-axis:')


def test_interp_c_to_g(periodic_1d):
    """Interpolate from c grid to g grid."""

    ds = periodic_1d

    # a linear gradient in the ni direction
    data_c = np.sin(ds['XC'])

    grid = Grid(ds)
    data_g = grid.interp(data_c, 'X')

    # check that the dimensions are right
    assert data_g.dims == ('XG',)
    xr.testing.assert_equal(data_g.XG, ds.XG)
    assert len(data_g.XG)==len(data_g)

    # check that the values are right
    np.testing.assert_allclose(
        data_u.values,
        0.5 * (data_c.values + np.roll(data_g.values, -1))
    )

def test_interp_c_to_g(nonperiodic_1d):
    """Interpolate from c grid to g grid."""

    ds = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.24
    data_c = grad * ds['ni']

    grid = Grid(ds, x_periodic=False)
    data_u = grid.interp(data_c, 'X')

    # check that the dimensions are right
    assert data_u.dims == ('ni_u',)
    xr.testing.assert_equal(data_u.ni_u, ds.ni_u[1:-1])
    assert len(data_u.ni_u)==len(data_u)

    # check that the values are right
    np.testing.assert_allclose(
        data_u.values,
        0.5 * (data_c.values[1:] + data_c.values[:-1])
    )

def test_interp_g_to_c(nonperiodic_1d):
    """Interpolate from g grid to c grid."""

    ds = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.43
    data_u = grad * ds['ni_u']

    grid = Grid(ds, x_periodic=False)
    data_c = grid.interp(data_u, 'X')

    # check that the dimensions are right
    assert data_c.dims == ('ni',)
    xr.testing.assert_equal(data_c.ni, ds.ni)
    assert len(data_c.ni)==len(data_c)

    # check that the values are right
    np.testing.assert_allclose(
        data_c.values,
        0.5 * (data_u.values[1:] + data_u.values[:-1])
    )
