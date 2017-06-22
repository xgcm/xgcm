from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np

from xgcm.grid import Grid, Axis

from . datasets import (all_datasets, nonperiodic_1d, periodic_1d, periodic_2d,
                        nonperiodic_2d)

# helper function to produce axes from datasets
def _get_axes(ds):
    all_axes = {ds[c].attrs['axis'] for c in ds.dims if 'axis' in ds[c].attrs}
    axis_objs = {ax: Axis(ds, ax) for ax in all_axes}
    return axis_objs


def test_create_axis(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_expected, coords_expected in expected['axes'].items():
        assert ax_expected in axis_objs
        this_axis = axis_objs[ax_expected]
        for axis_name, coord_name in coords_expected.items():
            assert axis_name in this_axis.coords
            assert this_axis.coords[axis_name].name == coord_name


def test_get_axis_coord(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_name, axis in axis_objs.items():
        # create a dataarray with each axis coordinate
        for position, coord in axis.coords.items():
            da = 1 * ds[coord.name]
            assert axis._get_axis_coord(da) == (position, coord.name)


def test_axis_wrap_and_replace_2d(periodic_2d):
    ds, periodic, expected = periodic_2d
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
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, 'X')

    da_c = 0 * ds.XC + 1
    da_g = 0 * ds.XG + 1

    to = (set(expected['axes']['X'].keys()) - {'center'}).pop()

    da_g_test = axis._wrap_and_replace_coords(da_c, da_g.data, to)
    assert da_g.equals(da_g_test)

    da_c_test = axis._wrap_and_replace_coords(da_g, da_c.data, 'center')
    assert da_c.equals(da_c_test)


# helper functions for padding arrays
def _pad_left(data, boundary, fill_value=0.):
    pad_val = data[0] if boundary=='extend' else fill_value
    return np.hstack([pad_val, data])


def _pad_right(data, boundary, fill_value=0.):
    pad_val = data[-1] if boundary=='extend' else fill_value
    return np.hstack([data, pad_val])


@pytest.mark.parametrize('boundary', [None, 'extend', 'fill'])
def test_axis_neighbor_pairs_face_to_center(nonperiodic_1d, boundary):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, 'X', periodic=periodic)

    padded = len(ds.XG) > len(ds.XC)
    to = (set(expected['axes']['X'].keys()) - {'center'}).pop()
    shift = expected.get('shift') or False

    if (boundary is None) and not padded:
        with pytest.raises(ValueError):
            data_left, data_right = axis._get_neighbor_data_pairs(ds.data_g,
                                        'center', boundary=boundary)
    else:
        data_left, data_right = axis._get_neighbor_data_pairs(ds.data_g,
                                    'center', boundary=boundary)
        if padded:
            expected_left = ds.data_g.data[:-1]
            expected_right = ds.data_g.data[1:]
        else:
            if shift:
                expected_right = ds.data_g.data
                expected_left = _pad_left(ds.data_g.data, boundary)[:-1]
            else:
                expected_left = ds.data_g.data
                expected_right = _pad_right(ds.data_g.data, boundary)[1:]

        np.testing.assert_allclose(data_left, expected_left)
        np.testing.assert_allclose(data_right, expected_right)

    # data_left, data_right = axis._get_neighbor_data_pairs(ds.data_c, to,
    #                                                         boundary='extend')
    # np.testing.assert_allclose(data_left[0], data_left[1])
    # np.testing.assert_allclose(data_right[-2], data_left[-1])
    #
    # data_left, data_right = axis._get_neighbor_data_pairs(ds.data_c, to,
    #                                                         boundary='fill')
    # np.testing.assert_allclose(data_left[0], 0.)
    # np.testing.assert_allclose(data_right[-1], 0.)
    #
    # data_left, data_right = axis._get_neighbor_data_pairs(ds.data_c, to,
    #                                                         boundary='fill',
    #                                                         fill_value=1.0)
    # np.testing.assert_allclose(data_left[0], 1.0)
    # np.testing.assert_allclose(data_right[-1], 1.0)


def test_axis_neighbor_pairs_2d(periodic_2d):
    ds, periodic, expected = periodic_2d

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


@pytest.mark.parametrize('boundary', ['extend', 'fill'])
def test_axis_diff_and_interp_nonperiodic_face_to_center(nonperiodic_1d, boundary):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, 'X', periodic=periodic)

    padded = len(ds.XG) > len(ds.XC)
    to = (set(expected['axes']['X'].keys()) - {'center'}).pop()
    shift = expected.get('shift') or False

    data = ds.data_g.data
    if padded:
        data_left = data[:-1]
        data_right = data[1:]
    elif shift:
        data_left = _pad_left(data[:-1], boundary)
        data_right = data
    else:
        data_left = data
        data_right = _pad_right(data[1:], boundary)

    # interpolate
    data_interp_expected = xr.DataArray(0.5 * (data_left + data_right),
                                        dims=['XC'], coords={'XC': ds.XC})
    data_interp = axis.interp(ds.data_g, 'center', boundary=boundary)
    print(data_interp_expected)
    print(data_interp)
    assert data_interp_expected.equals(data_interp)
    # check without "to" specified
    assert data_interp.equals(axis.interp(ds.data_g, boundary=boundary))

    # difference
    data_diff_expected = xr.DataArray(data_right - data_left,
                                      dims=['XC'], coords={'XC': ds.XC})
    data_diff = axis.diff(ds.data_g, 'center', boundary=boundary)
    assert data_diff_expected.equals(data_diff)
    # check without "to" specified
    assert data_diff.equals(axis.diff(ds.data_g, boundary=boundary))


@pytest.mark.parametrize('boundary', ['extend', 'fill'])
def test_axis_diff_and_interp_nonperiodic_center_to_face(nonperiodic_1d,
                boundary):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, 'X', periodic=periodic)

    padded = len(ds.XG) > len(ds.XC)
    to = (set(expected['axes']['X'].keys()) - {'center'}).pop()
    shift = expected.get('shift') or False

    data = ds.data_c.data
    if padded:
        data_left = _pad_left(data, boundary)
        data_right = _pad_right(data, boundary)
    elif shift:
        data_left = data
        data_right = _pad_right(data[1:], boundary)
    else:
        data_left = _pad_left(data[:-1], boundary)
        data_right = data

    # interpolate
    data_interp_expected = xr.DataArray(0.5 * (data_left + data_right),
                                        dims=['XG'], coords={'XG': ds.XG})
    data_interp = axis.interp(ds.data_c, to, boundary=boundary)
    assert data_interp_expected.equals(data_interp)
    # # check without "to" specified
    assert data_interp.equals(axis.interp(ds.data_c, boundary=boundary))

    # difference
    data_diff_expected = xr.DataArray(data_right - data_left,
                                       dims=['XG'], coords={'XG': ds.XG})
    data_diff = axis.diff(ds.data_c, to, boundary=boundary)
    assert data_diff_expected.equals(data_diff)
    # # check without "to" specified
    assert data_diff.equals(axis.diff(ds.data_c, boundary=boundary))


@pytest.mark.parametrize("varname, axis_name, to, roll, roll_axis, swap_order",
            [('data_cc', 'X', 'left', 1, 1, False),
            ('data_cc', 'Y', 'left', 1, 0, False),
            ('data_gg', 'X', 'center', -1, 1, True),
            ('data_gg', 'Y', 'center', -1, 0, True)]
)
def test_axis_diff_and_interp_periodic_2d(periodic_2d, varname, axis_name, to, roll,
                                          roll_axis, swap_order):
    ds, periodic, expected = periodic_2d

    axis = Axis(ds, axis_name)

    axis_lookups = {'XC': 'XG', 'XG': 'XC', 'YC': 'YG', 'YG': 'YC'}

    da = ds[varname]
    data = da.data
    data_roll = np.roll(data, roll, axis=roll_axis)
    if swap_order:
        data, data_roll = data_roll, data
    data_interp = 0.5 * (data + data_roll)
    data_diff = data - data_roll

    # determine new dims
    dims = list(da.dims)
    dims[roll_axis] = axis_lookups[dims[roll_axis]]
    coords = {dim: ds[dim] for dim in dims}

    da_interp_expected = xr.DataArray(data_interp, dims=dims, coords=coords)
    da_diff_expected = xr.DataArray(data_diff, dims=dims, coords=coords)

    da_interp = axis.interp(da, to)
    da_diff = axis.diff(da, to)

    assert da_interp_expected.equals(da_interp)
    assert da_diff_expected.equals(da_diff)

    # now make sure the defaults work (don't specify to)
    assert da_interp.equals(axis.interp(da))
    assert da_diff.equals(axis.diff(da))



@pytest.mark.parametrize('boundary', ['extend', 'fill'])
@pytest.mark.parametrize('axis_name', ['X', 'Y'])
@pytest.mark.parametrize('varname, this, to',
                         [('data_cc', 'center', 'left'),
                          ('data_gg', 'left', 'center')])
def test_axis_diff_and_interp_nonperiodic_2d(nonperiodic_2d, boundary, axis_name,
                                             varname, this, to,):
    ds, periodic, expected = nonperiodic_2d

    try:
        ax_periodic = axis_name in periodic
    except TypeError:
        ax_periodic = periodic

    axis = Axis(ds, axis_name, periodic=ax_periodic)
    da = ds[varname]

    # everything is left shift
    data = ds[varname].data

    axis_num = da.get_axis_num(axis.coords[this].name)
    print(axis_num)

    # lookups for numpy.pad
    numpy_pad_arg = {'extend': 'edge', 'fill': 'constant'}
    # args for numpy.pad
    pad_left = (1,0)
    pad_right = (0,1)
    pad_none = (0,0)
    if this=='center':
        pad_width = [pad_left if i==axis_num else pad_none
                     for i in range(data.ndim)]
        the_slice = [slice(0,-1) if i==axis_num else slice(None)
                     for i in range(data.ndim)]
        data_left = np.pad(data, pad_width, numpy_pad_arg[boundary])[the_slice]
        data_right = data
    elif this=='left':
        pad_width = [pad_right if i==axis_num else pad_none
                     for i in range(data.ndim)]
        the_slice = [slice(1,None) if i==axis_num else slice(None)
                     for i in range(data.ndim)]
        print(the_slice)
        data_right = np.pad(data, pad_width, numpy_pad_arg[boundary])[the_slice]
        print(data_right.shape)
        data_left = data

    data_interp = 0.5 * (data_left + data_right)
    data_diff = data_right - data_left

    # determine new dims
    dims = list(da.dims)
    dims[axis_num] = axis.coords[to].name
    coords = {dim: ds[dim] for dim in dims}

    da_interp_expected = xr.DataArray(data_interp, dims=dims, coords=coords)
    da_diff_expected = xr.DataArray(data_diff, dims=dims, coords=coords)

    da_interp = axis.interp(da, to, boundary=boundary)
    da_diff = axis.diff(da, to, boundary=boundary)

    assert da_interp_expected.equals(da_interp)
    assert da_diff_expected.equals(da_diff)


def test_create_grid(all_datasets):
    ds, periodic, expected = all_datasets
    grid = Grid(ds)


def test_grid_repr(all_datasets):
    ds, periodic, expected = all_datasets
    grid = Grid(ds)
    print(grid)
    r = repr(grid).split('\n')
    assert r[0] == "<xgcm.Grid>"


@pytest.mark.skip(reason="Depracting grid tests")
def test_interp_c_to_g_periodic(periodic_1d):
    """Interpolate from c grid to g grid."""
    ds, periodic, expected = periodic_1d
    reverse_shift = expected.get('shift') or False
    roll = -1 if reverse_shift else 1

    data_c = np.sin(ds['XC'])
    # np.roll(np.arange(5), 1) --> [4, 0, 1, 2, 3]
    # positive roll shifts left
    data_expected = 0.5 * (data_c.values + np.roll(data_c.values, roll))

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


@pytest.mark.skip(reason="Depracting grid tests")
def test_diff_c_to_g_periodic(periodic_1d):
    ds, periodic, expected = periodic_1d
    reverse_shift = expected.get('shift') or False
    roll = -1 if reverse_shift else 1

    # a linear gradient in the ni direction
    data_c = np.sin(ds['XC'])
    data_expected = roll*(data_c.values - np.roll(data_c.values, roll))

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


@pytest.mark.skip(reason="Depracting grid tests")
def test_interp_g_to_c_periodic(periodic_1d):
    """Interpolate from c grid to g grid."""
    ds, periodic, expected = periodic_1d

    reverse_shift = expected.get('shift') or False
    roll = 1 if reverse_shift else -1

    # a linear gradient in the ni direction
    data_g = np.sin(ds['XG'])
    data_expected = 0.5 * (data_g.values + np.roll(data_g.values, roll))

    grid = Grid(ds)
    data_c = grid.interp(data_g, 'X')

    # check that the dimensions are right
    assert data_c.dims == ('XC',)
    xr.testing.assert_equal(data_c.XC, ds.XC)
    assert len(data_c.XC)==len(data_c)

    # check that the values are right
    np.testing.assert_allclose(data_c.values, data_expected)


@pytest.mark.skip(reason="Depracting grid tests")
def test_diff_g_to_c_periodic(periodic_1d):
    ds, periodic, expected = periodic_1d

    reverse_shift = expected.get('shift') or False
    # a linear gradient in the ni direction
    data_g = np.sin(ds['XG'])
    # np.roll(np.arange(5), -1) --> [1, 2, 3, 4, 0]
    # negative roll shifts right
    roll = 1 if reverse_shift else -1
    data_expected = (-roll)*(np.roll(data_g.values, roll) - data_g.values)
    #data_expected = np.cos(ds['XC']).values * (2*np.pi) / 100.

    grid = Grid(ds)
    data_c = grid.diff(data_g, 'X')

    # check that the dimensions are right
    assert data_c.dims == ('XC',)
    xr.testing.assert_equal(data_c.XC, ds.XC)
    assert len(data_c.XC)==len(data_c)

    # check that the values are right
    np.testing.assert_allclose(data_c.values, data_expected)


@pytest.mark.skip(reason="Depracting grid tests")
@pytest.mark.parametrize('boundary', ['extend', 'fill'])
def test_interp_c_to_g_nonperiodic(nonperiodic_1d, boundary):
    """Interpolate from c grid to g grid."""

    ds, periodic, expected = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.24
    data_c = grad * ds['XC']

    data = data_c.data
    if boundary=='extend':
        pad_left, pad_right = data[0], data[-1]
    elif boundary=='fill':
        pad_left, pad_right = 0, 0
    data_left = np.hstack([pad_left, data])
    data_right = np.hstack([data, pad_right])
    data_expected = 0.5*(data_right + data_left)

    grid = Grid(ds, periodic=periodic)
    data_u = grid.interp(data_c, 'X', boundary=boundary)

    # check that the dimensions are right
    assert data_u.dims == ('XG',)
    xr.testing.assert_equal(data_u.XG, ds.XG)
    assert len(data_u.XG)==len(data_u)

    # check that the values are right
    np.testing.assert_allclose(data_u.values, data_expected)


@pytest.mark.skip(reason="Depracting grid tests")
@pytest.mark.parametrize('boundary', ['extend', 'fill'])
def test_diff_c_to_g_nonperiodic(nonperiodic_1d, boundary):
    ds, periodic, expected = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.24
    data_c = grad * ds['XC']

    data = data_c.data
    if boundary=='extend':
        pad_left, pad_right = data[0], data[-1]
    elif boundary=='fill':
        pad_left, pad_right = 0, 0
    data_left = np.hstack([pad_left, data])
    data_right = np.hstack([data, pad_right])
    data_expected = data_right - data_left

    grid = Grid(ds, periodic=periodic)
    data_u = grid.diff(data_c, 'X', boundary=boundary)

    # check that the dimensions are right
    assert data_u.dims == ('XG',)
    xr.testing.assert_equal(data_u.XG, ds.XG)
    assert len(data_u.XG)==len(data_u)

    # check that the values are right
    np.testing.assert_allclose(data_u.values, data_expected)
    np.testing.assert_allclose(data_u.values[1:-1], grad)

@pytest.mark.skip(reason="Depracting grid tests")
def test_interp_g_to_c_nonperiodic(nonperiodic_1d):
    """Interpolate from g grid to c grid."""

    ds, periodic, expected = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.43
    data_u = grad * ds['XG']
    data_expected = 0.5 * (data_u.values[1:] + data_u.values[:-1])

    grid = Grid(ds, periodic=periodic)
    data_c = grid.interp(data_u, 'X')

    # check that the dimensions are right
    assert data_c.dims == ('XC',)
    xr.testing.assert_equal(data_c.XC, ds.XC)
    assert len(data_c.XC)==len(data_c)

    # check that the values are right
    np.testing.assert_allclose(data_c.values, data_expected)

@pytest.mark.skip(reason="Depracting grid tests")
def test_diff_g_to_c_nonperiodic(nonperiodic_1d):
    """Interpolate from g grid to c grid."""

    ds, periodic, expected = nonperiodic_1d

    # a linear gradient in the ni direction
    grad = 0.43
    data_u = grad * ds['XG']
    data_expected = data_u.values[1:] - data_u.values[:-1]

    grid = Grid(ds, periodic=periodic)
    data_c = grid.diff(data_u, 'X')

    # check that the dimensions are right
    assert data_c.dims == ('XC',)
    xr.testing.assert_equal(data_c.XC, ds.XC)
    assert len(data_c.XC)==len(data_c)

    # check that the values are right
    np.testing.assert_allclose(data_c.values, data_expected)
    np.testing.assert_allclose(data_c.values, grad)
