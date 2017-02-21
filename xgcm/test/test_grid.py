from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np

from xgcm.grid import Grid, _replace_dim

from . datasets import all_datasets, nonperiodic_1d, periodic_1d

def test_create_grid(all_datasets):
    grid = Grid(all_datasets)

def test_grid_repr(all_datasets):
    grid = Grid(all_datasets)
    print(grid)
    r = repr(grid).split('\n')
    assert r[0] == "<xgcm.Grid>"
    # all datasets should have at least an X axis
    assert r[1].startswith('X-axis:')

def test_replace_dim():
    orig = xr.DataArray(np.random.rand(10),
                        coords={'x': (['x'], np.arange(10))},
                        dims=['x'])
    new_ds = xr.Dataset(coords={'xnew': (['xnew'], 5*np.arange(10))})
    new = _replace_dim(orig, 'x', new_ds.xnew, drop=True)
    assert new.dims == ('xnew',)
    assert new_ds.xnew.equals(new.xnew)

def test_interp_c_to_g_periodic(periodic_1d):
    """Interpolate from c grid to g grid."""
    ds = periodic_1d

    data_c = np.sin(ds['XC'])
    # np.roll(np.arange(5), 1) --> [4, 0, 1, 2, 3]
    # positive roll shifts left
    data_expected = 0.5 * (data_c.values + np.roll(data_c.values, 1))

    grid = Grid(ds)
    data_g = grid.interp(data_c, 'X')

    # check that the dimensions are right
    assert data_g.dims == ('XG',)
    xr.testing.assert_equal(data_g.XG, ds.XG)
    assert len(data_g.XG)==len(data_g)

    # check that the values are right
    np.testing.assert_allclose(data_g.values, data_expected)

    # try the same with chunks
    data_c = np.sin(ds['XC'])
    data_c = data_c.chunk(10)
    data_g = grid.interp(data_c, 'X')
    np.testing.assert_allclose(data_g.values, data_expected)

def test_diff_c_to_g_periodic(periodic_1d):
    ds = periodic_1d

    # a linear gradient in the ni direction
    data_c = np.sin(ds['XC'])
    data_expected = data_c.values - np.roll(data_c.values, 1)

    grid = Grid(ds)
    data_g = grid.diff(data_c, 'X')

    # check that the dimensions are right
    assert data_g.dims == ('XG',)
    xr.testing.assert_equal(data_g.XG, ds.XG)
    assert len(data_g.XG)==len(data_g)

    # check that the values are right
    np.testing.assert_allclose(data_g.values, data_expected)

    # try the same with chunks
    data_c = np.sin(ds['XC'])
    data_c = data_c.chunk(10)
    data_g = grid.diff(data_c, 'X')
    np.testing.assert_allclose(data_g.values, data_expected)

def test_interp_g_to_c_periodic(periodic_1d):
    """Interpolate from c grid to g grid."""
    ds = periodic_1d

    # a linear gradient in the ni direction
    data_g = np.sin(ds['XG'])
    data_expected = 0.5 * (data_g.values + np.roll(data_g.values, -1))

    grid = Grid(ds)
    data_c = grid.interp(data_g, 'X')

    # check that the dimensions are right
    assert data_c.dims == ('XC',)
    xr.testing.assert_equal(data_c.XC, ds.XC)
    assert len(data_c.XC)==len(data_c)

    # check that the values are right
    np.testing.assert_allclose(data_c.values, data_expected)

def test_diff_g_to_c_periodic(periodic_1d):
    ds = periodic_1d

    # a linear gradient in the ni direction
    data_g = np.sin(ds['XG'])
    # np.roll(np.arange(5), -1) --> [1, 2, 3, 4, 0]
    # negative roll shifts right
    data_expected = np.roll(data_g.values, -1) - data_g.values
    #data_expected = np.cos(ds['XC']).values * (2*np.pi) / 100.

    grid = Grid(ds)
    data_c = grid.diff(data_g, 'X')

    # check that the dimensions are right
    assert data_c.dims == ('XC',)
    xr.testing.assert_equal(data_c.XC, ds.XC)
    assert len(data_c.XC)==len(data_c)

    # check that the values are right
    np.testing.assert_allclose(data_c.values, data_expected)

def test_interp_c_to_g_nonperiodic(nonperiodic_1d):
    """Interpolate from c grid to g grid."""

    ds = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.24
    data_c = grad * ds['ni']
    data_expected = 0.5 * (data_c.values[1:] + data_c.values[:-1])

    grid = Grid(ds, x_periodic=False)
    data_u = grid.interp(data_c, 'X')

    # check that the dimensions are right
    assert data_u.dims == ('ni_u',)
    xr.testing.assert_equal(data_u.ni_u, ds.ni_u[1:-1])
    assert len(data_u.ni_u)==len(data_u)

    # check that the values are right
    np.testing.assert_allclose(data_u.values, data_expected)


def test_diff_c_to_g_nonperiodic(nonperiodic_1d):
    ds = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.24
    data_c = grad * ds['ni']
    data_expected = data_c.values[1:] - data_c.values[:-1]

    grid = Grid(ds, x_periodic=False)
    data_u = grid.diff(data_c, 'X')

    # check that the dimensions are right
    assert data_u.dims == ('ni_u',)
    xr.testing.assert_equal(data_u.ni_u, ds.ni_u[1:-1])
    assert len(data_u.ni_u)==len(data_u)

    # check that the values are right
    np.testing.assert_allclose(data_u.values, data_expected)
    np.testing.assert_allclose(data_u.values, grad)

def test_interp_g_to_c_nonperiodic(nonperiodic_1d):
    """Interpolate from g grid to c grid."""

    ds = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.43
    data_u = grad * ds['ni_u']
    data_expected = 0.5 * (data_u.values[1:] + data_u.values[:-1])

    grid = Grid(ds, x_periodic=False)
    data_c = grid.interp(data_u, 'X')

    # check that the dimensions are right
    assert data_c.dims == ('ni',)
    xr.testing.assert_equal(data_c.ni, ds.ni)
    assert len(data_c.ni)==len(data_c)

    # check that the values are right
    np.testing.assert_allclose(data_c.values, data_expected)


def test_diff_g_to_c_nonperiodic(nonperiodic_1d):
    """Interpolate from g grid to c grid."""

    ds = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.43
    data_u = grad * ds['ni_u']
    data_expected = data_u.values[1:] - data_u.values[:-1]

    grid = Grid(ds, x_periodic=False)
    data_c = grid.diff(data_u, 'X')

    # check that the dimensions are right
    assert data_c.dims == ('ni',)
    xr.testing.assert_equal(data_c.ni, ds.ni)
    assert len(data_c.ni)==len(data_c)

    # check that the values are right
    np.testing.assert_allclose(data_c.values, data_expected)
    np.testing.assert_allclose(data_c.values, grad)
