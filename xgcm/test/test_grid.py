from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np

from xgcm.grid import Grid, Axis

from . datasets import (all_datasets, nonperiodic_1d, periodic_1d, periodic_2d,
                        nonperiodic_2d, all_2d, datasets)

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


def test_axis_repr(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_name, axis in axis_objs.items():
        r = repr(axis).split('\n')
        assert r[0].startswith("<xgcm.Axis")
    # TODO: make this more complete


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
@pytest.mark.parametrize('from_center', [True, False])
def test_axis_neighbor_pairs_nonperiodic_1d(nonperiodic_1d, boundary, from_center):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, 'X', periodic=periodic)

    # detect whether this is an outer or inner case
    # outer --> dim_line_diff = 1
    # inner --> dim_line_diff = -1
    dim_len_diff = len(ds.XG) - len(ds.XC)

    if from_center:
        to = (set(expected['axes']['X'].keys()) - {'center'}).pop()
        da = ds.data_c
    else:
        to = 'center'
        da = ds.data_g

    shift = expected.get('shift') or False

    # need boundary condition for everything but outer to center
    if (boundary is None) and (dim_len_diff == 0 or
        (dim_len_diff == 1 and from_center) or
        (dim_len_diff == -1 and not from_center)):
        with pytest.raises(ValueError):
            data_left, data_right = axis._get_neighbor_data_pairs(da, to,
                                                boundary=boundary)
    else:
        data_left, data_right = axis._get_neighbor_data_pairs(da, to,
                                                boundary=boundary)
        if (((dim_len_diff == 1) and not from_center) or
            ((dim_len_diff == -1) and from_center)):
            expected_left = da.data[:-1]
            expected_right = da.data[1:]
        elif (((dim_len_diff == 1) and from_center) or
              ((dim_len_diff == -1) and not from_center)):
            expected_left = _pad_left(da.data, boundary)
            expected_right = _pad_right(da.data, boundary)
        elif (shift and not from_center) or (not shift and from_center):
            expected_right = da.data
            expected_left = _pad_left(da.data, boundary)[:-1]
        else:
            expected_left = da.data
            expected_right = _pad_right(da.data, boundary)[1:]

        np.testing.assert_allclose(data_left, expected_left)
        np.testing.assert_allclose(data_right, expected_right)

@pytest.mark.parametrize('varname, axis_name, to, roll, roll_axis, swap_order',
    [('data_c', 'X', 'left', 1, 1, False),
    ('data_c', 'Y', 'left', 1, 0, False),
    ('data_g', 'X', 'center', -1, 1, True),
    ('data_g', 'Y', 'center', -1, 0, True)]
)
def test_axis_neighbor_pairs_2d(periodic_2d, varname, axis_name, to, roll,
                                roll_axis, swap_order):
    ds, periodic, expected = periodic_2d

    axis = Axis(ds, axis_name)

    data = ds[varname]
    data_left, data_right = axis._get_neighbor_data_pairs(data, to)
    if swap_order:
        data_left, data_right = data_right, data_left
    np.testing.assert_allclose(data_left, np.roll(data.data,
                                                  roll, axis=roll_axis))
    np.testing.assert_allclose(data_right, data.data)


@pytest.mark.parametrize('boundary', ['extend', 'fill'])
@pytest.mark.parametrize('from_center', [True, False])
def test_axis_diff_and_interp_nonperiodic_1d(nonperiodic_1d, boundary, from_center):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, 'X', periodic=periodic)

    dim_len_diff = len(ds.XG) - len(ds.XC)

    if from_center:
        to = (set(expected['axes']['X'].keys()) - {'center'}).pop()
        coord_to = 'XG'
        da = ds.data_c
    else:
        to = 'center'
        coord_to = 'XC'
        da = ds.data_g

    shift = expected.get('shift') or False

    data = da.data
    if ((dim_len_diff==1 and not from_center) or
        (dim_len_diff==-1 and from_center)):
        data_left = data[:-1]
        data_right = data[1:]
    elif ((dim_len_diff==1 and from_center) or
          (dim_len_diff==-1 and not from_center)):
        data_left = _pad_left(data, boundary)
        data_right = _pad_right(data, boundary)
    elif (shift and not from_center) or (not shift and from_center):
        data_left = _pad_left(data[:-1], boundary)
        data_right = data
    else:
        data_left = data
        data_right = _pad_right(data[1:], boundary)

    # interpolate
    data_interp_expected = xr.DataArray(0.5 * (data_left + data_right),
                                        dims=[coord_to],
                                        coords={coord_to: ds[coord_to]})
    data_interp = axis.interp(da, to, boundary=boundary)
    print(data_interp_expected)
    print(data_interp)
    assert data_interp_expected.equals(data_interp)
    # check without "to" specified
    assert data_interp.equals(axis.interp(da, boundary=boundary))

    # difference
    data_diff_expected = xr.DataArray(data_right - data_left,
                                      dims=[coord_to],
                                      coords={coord_to: ds[coord_to]})
    data_diff = axis.diff(da, to, boundary=boundary)
    assert data_diff_expected.equals(data_diff)
    # check without "to" specified
    assert data_diff.equals(axis.diff(da, boundary=boundary))


# this mega test covers all options for 2D data

@pytest.mark.parametrize('boundary', ['extend', 'fill'])
@pytest.mark.parametrize('axis_name', ['X', 'Y'])
@pytest.mark.parametrize('varname, this, to',
                         [('data_c', 'center', 'left'),
                          ('data_g', 'left', 'center')])
def test_axis_diff_and_interp_nonperiodic_2d(all_2d, boundary, axis_name,
                                             varname, this, to,):
    ds, periodic, expected = all_2d

    try:
        ax_periodic = axis_name in periodic
    except TypeError:
        ax_periodic = periodic

    axis = Axis(ds, axis_name, periodic=ax_periodic)
    da = ds[varname]

    # everything is left shift
    data = ds[varname].data

    axis_num = da.get_axis_num(axis.coords[this].name)
    print(axis_num, ax_periodic)

    # lookups for numpy.pad
    numpy_pad_arg = {'extend': 'edge', 'fill': 'constant'}
    # args for numpy.pad
    pad_left = (1,0)
    pad_right = (0,1)
    pad_none = (0,0)

    if this=='center':
        if ax_periodic:
            data_left = np.roll(data, 1, axis=axis_num)
            data_right = data
        else:
            pad_width = [pad_left if i==axis_num else pad_none
                         for i in range(data.ndim)]
            the_slice = [slice(0,-1) if i==axis_num else slice(None)
                         for i in range(data.ndim)]
            data_left = np.pad(data, pad_width, numpy_pad_arg[boundary])[the_slice]
            data_right = data
    elif this=='left':
        if ax_periodic:
            data_left = data
            data_right = np.roll(data, -1, axis=axis_num)
        else:
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

    boundary_arg = boundary if not ax_periodic else None
    da_interp = axis.interp(da, to, boundary=boundary_arg)
    da_diff = axis.diff(da, to, boundary=boundary_arg)

    assert da_interp_expected.equals(da_interp)
    assert da_diff_expected.equals(da_diff)


def test_axis_errors():
    ds = datasets['1d_left']

    ds_noattr = ds.copy()
    del ds_noattr.XC.attrs['axis']
    with pytest.raises(ValueError,
                       message="Couldn't find a center coordinate for axis X"):
        x_axis = Axis(ds_noattr, 'X', periodic=True)

    del ds_noattr.XG.attrs['axis']
    with pytest.raises(ValueError,
                       message="Couldn't find any coordinates for axis X"):
        x_axis = Axis(ds_noattr, 'X', periodic=True)

    ds_chopped = ds.copy()
    del ds_chopped['data_g']
    ds_chopped['XG'] = ds_chopped['XG'][:-3]
    with pytest.raises(ValueError, message="Left coordinate XG has"
                                    "incompatible length 7 (axis_len=9)"):
        x_axis = Axis(ds_chopped, 'X', periodic=True)

    ds_chopped.XG.attrs['c_grid_axis_shift'] = -0.5
    with pytest.raises(ValueError, message="Right coordinate XG has"
                                    "incompatible length 7 (axis_len=9)"):
        x_axis = Axis(ds_chopped, 'X', periodic=True)

    del ds_chopped.XG.attrs['c_grid_axis_shift']
    with pytest.raises(ValueError, message="Coordinate XC has invalid or "
                                "missing c_grid_axis_shift attribute `None`"):
        x_axis = Axis(ds_chopped, 'X', periodic=True)

    ax = Axis(ds, 'X', periodic=True)

    with pytest.raises(ValueError, message="Can't get neighbor pairs for"
                                   "the same position."):
        ax.interp(ds.data_c, 'center')

    with pytest.raises(KeyError,
                    message="Position 'right' was not found in axis.coords."):
        ax.interp(ds.data_c, 'right')

    with pytest.raises(ValueError, message="`boundary=fill` is not allowed "
                                    "with periodic axis X."):
        ax.interp(ds.data_c, 'right', boundary='fill')



def test_grid_create(all_datasets):
    ds, periodic, expected = all_datasets
    grid = Grid(ds, periodic=periodic)


def test_grid_repr(all_datasets):
    ds, periodic, expected = all_datasets
    grid = Grid(ds, periodic=periodic)
    print(grid)
    r = repr(grid).split('\n')
    assert r[0] == "<xgcm.Grid>"


def test_grid_ops(all_datasets):
    """
    Check that we get the same answer using Axis or Grid objects
    """
    ds, periodic, expected = all_datasets
    grid = Grid(ds, periodic=periodic)

    for axis_name in grid.axes.keys():
        try:
            ax_periodic = axis_name in periodic
        except TypeError:
            ax_periodic = periodic
        axis = Axis(ds, axis_name, periodic=ax_periodic)

        bcs = [None] if ax_periodic else ['fill', 'extend']
        for varname in ['data_c', 'data_g']:
            for boundary in bcs:
                da_interp = grid.interp(ds[varname], axis_name,
                    boundary=boundary)
                da_interp_ax = axis.interp(ds[varname], boundary=boundary)
                assert da_interp.equals(da_interp_ax)
                da_diff = grid.diff(ds[varname], axis_name,
                    boundary=boundary)
                da_diff_ax = axis.diff(ds[varname], boundary=boundary)
                assert da_diff.equals(da_diff_ax)
