from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np
import dask as dsk
from dask.array import from_array

from xgcm.grid import Grid, Axis, add_to_slice
from xgcm.duck_array_ops import concatenate

from . datasets import (all_datasets, nonperiodic_1d, periodic_1d, periodic_2d,
                        nonperiodic_2d, all_2d, datasets)

# helper function to produce axes from datasets
def _get_axes(ds):
    all_axes = {ds[c].attrs['axis'] for c in ds.dims if 'axis' in ds[c].attrs}
    axis_objs = {ax: Axis(ds, ax) for ax in all_axes}
    return axis_objs


# duck array ops should maybe get its own test module?
def test_concatenate():
    a = np.array([1, 2, 3])
    b = np.array([10])
    a_dask = from_array(a, chunks=1)
    b_dask = from_array(b, chunks=1)
    concat = concatenate([a, b], axis=0)
    concat_dask = concatenate([a_dask, b_dask], axis=0)
    concat_mixed = concatenate([a, b_dask], axis=0)
    assert isinstance(concat, np.ndarray)
    assert isinstance(concat_dask, dsk.array.Array)
    assert isinstance(concat_mixed,  np.ndarray)


@pytest.mark.parametrize('discontinuity', [None, 10, 360])
@pytest.mark.parametrize('right', [True, False])
def test_extend_right_left(discontinuity, right):
    ds = datasets['1d_left']
    ds_check = ds.copy()
    axis = Axis(ds, 'X')
    if discontinuity is None:
        ref = 0
    else:
        ref = discontinuity

    kw = {'boundary_discontinuity': discontinuity}
    if right:
        extended_raw = axis._extend_right(ds.XC, **kw)
        extended = extended_raw[-1]
        expected = ds.XC.data[0] + ref
        with pytest.raises(RuntimeError):
            axis._extend_right(ds.XC.data, **kw)
    else:
        extended_raw = axis._extend_left(ds.XC, **kw)
        extended = extended_raw[0]
        expected = ds.XC.data[-1] - ref
        with pytest.raises(RuntimeError):
            axis._extend_left(ds.XC.data, **kw)

    assert isinstance(extended_raw, np.ndarray)
    assert extended == expected


@pytest.mark.parametrize('fill_value', [0, 10, 20])
@pytest.mark.parametrize('boundary', ['fill', 'extend', 'extrapolate'])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('is_left_edge', [True, False])
@pytest.mark.parametrize('boundary_discontinuity', [None, 360])
def test_get_edge_data(periodic, fill_value,
                       boundary, is_left_edge,
                       boundary_discontinuity):
    ds = datasets['1d_left']
    axis = Axis(ds, 'X', periodic=periodic)
    edge = axis._get_edge_data(ds.XC, boundary=boundary,
                               fill_value=fill_value,
                               is_left_edge=is_left_edge,
                               boundary_discontinuity=boundary_discontinuity
                               )
    if is_left_edge:
        edge_periodic = ds.XC.data[-1]
        if boundary_discontinuity is not None:
            edge_periodic = edge_periodic - boundary_discontinuity
        edge_extend = ds.XC.data[0]
        edge_extra = ds.XC.data[0] - np.diff(ds.XC.data[0:2])
    else:
        edge_periodic = ds.XC.data[0]
        if boundary_discontinuity is not None:
            edge_periodic = edge_periodic + boundary_discontinuity
        edge_extend = ds.XC.data[-1]
        edge_extra = ds.XC.data[-1] + np.diff(ds.XC.data[-2:])
    edge_fill = fill_value

    if periodic:
        assert edge_periodic == edge
    else:
        if boundary == 'fill':
            assert edge_fill == edge
        elif boundary == 'extend':
            assert edge_extend == edge
        elif boundary == 'extrapolate':
            assert edge_extra == edge
        else:
            assert 0


def test_create_axis(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_expected, coords_expected in expected['axes'].items():
        assert ax_expected in axis_objs
        this_axis = axis_objs[ax_expected]
        for axis_name, coord_name in coords_expected.items():
            assert axis_name in this_axis.coords
            assert this_axis.coords[axis_name].name == coord_name


def _assert_axes_equal(ax1, ax2):
    assert ax1.name == ax2.name
    for pos, coord in ax1.coords.items():
        assert pos in ax2.coords
        this_coord = ax2.coords[pos]
        assert coord.equals(this_coord)
    assert ax1._periodic == ax2._periodic
    assert ax1._default_shifts == ax2._default_shifts
    assert ax1._facedim == ax2._facedim
    # TODO: make this work...
    #assert ax1._connections == ax2._connections


def test_create_axis_no_comodo(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)
    print(axis_objs)

    # now strip out the metadata
    ds_noattr = ds.copy()
    for var in ds.variables:
        ds_noattr[var].attrs.clear()

    for axis_name, axis_coords in expected['axes'].items():
        # now create the axis from scratch with no attributes
        this_axis = Axis(ds_noattr, axis_name, coords=axis_coords)
        axis_expected = axis_objs[axis_name]
        # make sure all the same stuff is present in both all_axes
        # TODO: come up with a more general way to assert axis equality
        _assert_axes_equal(axis_expected, this_axis)


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
# this feels silly...I'm basically just re-coding the function in order to
# test it
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


@pytest.mark.parametrize('boundary', ['extend', 'fill'])
def test_axis_cumsum(nonperiodic_1d, boundary):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, 'X', periodic=periodic)

    axis_expected = expected['axes']['X']

    cumsum_g = axis.cumsum(ds.data_g, to='center', boundary=boundary)
    assert cumsum_g.dims == ds.data_c.dims
    # check default "to"
    assert cumsum_g.equals(axis.cumsum(ds.data_g, boundary=boundary))

    to = set(axis_expected).difference({'center'}).pop()
    cumsum_c = axis.cumsum(ds.data_c, to=to, boundary=boundary)
    assert cumsum_c.dims == ds.data_g.dims
    # check default "to"
    assert cumsum_c.equals(axis.cumsum(ds.data_c, boundary=boundary))

    cumsum_c_raw = np.cumsum(ds.data_c.data)
    cumsum_g_raw = np.cumsum(ds.data_g.data)

    if to == 'right':
        np.testing.assert_allclose(cumsum_c.data, cumsum_c_raw)
        fill_value = 0. if boundary=='fill' else cumsum_g_raw[0]
        np.testing.assert_allclose(cumsum_g.data,
            np.hstack([fill_value, cumsum_g_raw[:-1]]))
    elif to == 'left':
        np.testing.assert_allclose(cumsum_g.data, cumsum_g_raw)
        fill_value = 0. if boundary=='fill' else cumsum_c_raw[0]
        np.testing.assert_allclose(cumsum_c.data,
            np.hstack([fill_value, cumsum_c_raw[:-1]]))
    elif to == 'inner':
        np.testing.assert_allclose(cumsum_c.data, cumsum_c_raw[:-1])
        fill_value = 0. if boundary=='fill' else cumsum_g_raw[0]
        np.testing.assert_allclose(cumsum_g.data,
            np.hstack([fill_value, cumsum_g_raw]))
    elif to == 'outer':
        np.testing.assert_allclose(cumsum_g.data, cumsum_g_raw[:-1])
        fill_value = 0. if boundary=='fill' else cumsum_c_raw[0]
        np.testing.assert_allclose(cumsum_c.data,
            np.hstack([fill_value, cumsum_c_raw]))

    ## not much point doing this...we don't have the right test datasets
    ## to really test the errors
    # other_positions = {'left', 'right', 'inner', 'outer'}.difference({to})
    # for pos in other_positions:
    #     with pytest.raises(KeyError):
    #         axis.cumsum(ds.data_c, to=pos, boundary=boundary)


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

    # max
    data_max_expected = xr.DataArray(xr.ufuncs.maximum(data_right, data_left),
                                     dims=[coord_to],
                                     coords={coord_to: ds[coord_to]})
    data_max = axis.max(da, to, boundary=boundary)
    assert data_max_expected.equals(data_max)
    # check without "to" specified
    assert data_max.equals(axis.max(da, boundary=boundary))

    # min
    data_min_expected = xr.DataArray(xr.ufuncs.minimum(data_right, data_left),
                                     dims=[coord_to],
                                     coords={coord_to: ds[coord_to]})
    data_min = axis.min(da, to, boundary=boundary)
    assert data_min_expected.equals(data_min)
    # check without "to" specified
    assert data_min.equals(axis.min(da, boundary=boundary))


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

    with pytest.raises(ValueError,
                    message="This axis doesn't contain a `right` position"):
        ax.interp(ds.data_c, 'right')

    with pytest.raises(ValueError, message="`boundary=fill` is not allowed "
                                    "with periodic axis X."):
        ax.interp(ds.data_c, 'right', boundary='fill')


def test_grid_create(all_datasets):
    ds, periodic, expected = all_datasets
    grid = Grid(ds, periodic=periodic)
    assert grid is not None


def test_create_grid_no_comodo(all_datasets):
    ds, periodic, expected = all_datasets
    grid_expected = Grid(ds, periodic=periodic)

    ds_noattr = ds.copy()
    for var in ds.variables:
        ds_noattr[var].attrs.clear()

    coords = expected['axes']
    grid = Grid(ds_noattr, periodic=periodic, coords=coords)

    for axis_name_expected in grid_expected.axes:
        axis_expected = grid_expected.axes[axis_name_expected]
        axis_actual = grid.axes[axis_name_expected]
        _assert_axes_equal(axis_expected, axis_actual)


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
                if boundary is not None:
                    da_cumsum = grid.cumsum(ds[varname], axis_name,
                        boundary=boundary)
                    da_cumsum_ax = axis.cumsum(ds[varname], boundary=boundary)
                    assert da_cumsum.equals(da_cumsum_ax)


def test_add_to_slice():
    np_ar = xr.DataArray(np.ones([2, 2, 3]),
                         dims=['lat', 'z', 'lon'])

    da_ar = xr.DataArray(from_array(np.ones([2, 2, 3]), chunks=1),
                         dims=['lat', 'z', 'lon'])

    np_new = add_to_slice(np_ar, 'lon', 1, 3.0)
    da_new = add_to_slice(da_ar, 'lon', 1, 3.0)
    da_new_last = add_to_slice(da_ar, 'lon', -1, 3.0)

    ref_last = np.array([[[1., 1., 4.],
                    [1., 1., 4.]],
                    [[1., 1., 4.],
                    [1., 1., 4.]]])

    ref = np.array([[[1., 4., 1.],
                    [1., 4., 1.]],
                    [[1., 4., 1.],
                    [1., 4., 1.]]])

    ref_ar = xr.DataArray(ref, dims=['lat', 'z', 'lon'])
    ref_ar_last = xr.DataArray(ref_last, dims=['lat', 'z', 'lon'])

    xr.testing.assert_equal(ref_ar, np_new)
    xr.testing.assert_equal(ref_ar, da_new.compute())
    xr.testing.assert_equal(ref_ar_last, da_new_last.compute())

# Needs test for _extend_right, _extend_left and the boundary_discontinuity input...not sure how to do that.
