from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np

from xgcm import Grid
from xgcm.grid import _replace_dim, Axis

from . datasets import all_datasets, nonperiodic_1d, periodic_1d, periodic_2d

# helper function to produce axes from datasets
def _get_axes(ds):
    all_axes = {ds[c].attrs['axis'] for c in ds.dims if 'axis' in ds[c].attrs}
    axis_objs = {ax: Axis(ds, ax) for ax in all_axes}
    return axis_objs


def test_create_axis(all_datasets):
    ds, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_expected, coords_expected in expected['axes'].items():
        assert ax_expected in axis_objs
        this_axis = axis_objs[ax_expected]
        for axis_name, coord_name in coords_expected.items():
            assert axis_name in this_axis.coords
            assert this_axis.coords[axis_name].name == coord_name


def test_get_axis_coord(all_datasets):
    ds, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_name, axis in axis_objs.items():
        # create a dataarray with each axis coordinate
        for position, coord in axis.coords.items():
            da = 1 * ds[coord.name]
            assert axis._get_axis_coord(da) == (position, coord.name)


def test_axis_wrap_and_replace_2d(periodic_2d):
    ds, expected = periodic_2d
    axis_objs = _get_axes(ds)

    da_xc_yc = 0 * ds.XC * ds.YC + 1
    da_xc_yg = 0 * ds.XC * ds.YG + 1
    da_xg_yc = 0 * ds.XG * ds.YC + 1

    da_xc_yg_test = axis_objs['Y']._wrap_and_replace_coords(
                        da_xc_yc, da_xc_yc.data, 'left')
    assert da_xc_yg.equals(da_xc_yg_test)

    da_xg_yc_test = axis_objs['X']._wrap_and_replace_coords(
                        da_xc_yc, da_xc_yc.data, 'left')
    assert da_xg_yc.equals(da_xg_yc_test)


def test_axis_wrap_and_replace_nonperiodic(nonperiodic_1d):
    ds, expected = nonperiodic_1d
    axis = Axis(ds, 'X')

    da_ni = 0 * ds.ni + 1
    da_ni_u = 0 * ds.ni_u + 1

    da_ni_u_test = axis._wrap_and_replace_coords(da_ni, da_ni_u.data, 'face')
    assert da_ni_u.equals(da_ni_u_test)

    da_ni_test = axis._wrap_and_replace_coords(da_ni_u, da_ni.data, 'center')
    assert da_ni.equals(da_ni_test)


def test_axis_neighbor_pairs(nonperiodic_1d):
    ds, expected = nonperiodic_1d
    axis = Axis(ds, 'X')

    data_left, data_right = axis._get_neighbor_data_pairs(ds.data_ni_u, 'center')
    np.testing.assert_allclose(data_left, ds.data_ni_u.data[:-1])
    np.testing.assert_allclose(data_right, ds.data_ni_u.data[1:])

    with pytest.raises(NotImplementedError):
        _, _ = axis._get_neighbor_data_pairs(ds.data_ni, 'face')


def test_axis_neighbor_pairs_2d(periodic_2d):
    ds, expected = periodic_2d

    x_axis = Axis(ds, 'X')
    y_axis = Axis(ds, 'Y')

    cases = [
        # varname   #axis    #to   #roll args # order
        ('data_cc', x_axis, 'left', 1, 1, False),
        ('data_cc', y_axis, 'left', 1, 0, False),
        ('data_gg', x_axis, 'center', -1, 1, True),
        ('data_gg', y_axis, 'center', -1, 0, True)
    ]

    for varname, axis, to, roll, roll_axis, swap_order in cases:
        data = ds[varname]
        data_left, data_right = axis._get_neighbor_data_pairs(data, to)
        if swap_order:
            data_left, data_right = data_right, data_left
        np.testing.assert_allclose(data_left, np.roll(data.data,
                                                      roll, axis=roll_axis))
        np.testing.assert_allclose(data_right, data.data)


def test_axis_diff_and_interp(nonperiodic_1d):
    ds, expected = nonperiodic_1d
    axis = Axis(ds, 'X')

    data_left, data_right = axis._get_neighbor_data_pairs(ds.data_ni_u, 'center')
    data_left = ds.data_ni_u.data[:-1]
    data_right = ds.data_ni_u.data[1:]

    # interpolate
    data_interp_expected = xr.DataArray(0.5 * (data_left + data_right),
                                        dims=['ni'], coords={'ni': ds.ni})
    data_interp = axis.interp(ds.data_ni_u, 'center')
    assert data_interp_expected.equals(data_interp)

    # difference
    data_diff_expected = xr.DataArray(data_right - data_left,
                                      dims=['ni'], coords={'ni': ds.ni})
    data_diff = axis.diff(ds.data_ni_u, 'center')
    assert data_diff_expected.equals(data_diff)

def test_create_grid(all_datasets):
    ds, expected = all_datasets
    grid = Grid(ds)

def test_grid_repr(all_datasets):
    ds, expected = all_datasets
    grid = Grid(ds)
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
    ds, expected = periodic_1d

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
    ds, expected = periodic_1d

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
    ds, expected = periodic_1d

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
    ds, expected = periodic_1d

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

    ds, expected = nonperiodic_1d

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
    ds, expected = nonperiodic_1d

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

    ds, expected = nonperiodic_1d

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

    ds, expected = nonperiodic_1d

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
