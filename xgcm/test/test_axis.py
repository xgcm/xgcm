import numpy as np
import pytest
import xarray as xr

from xgcm.axis import Axis

from . import datasets


def periodic_1d():
    # TODO get this from datasets.py

    N = 5
    ds = xr.Dataset(
        {"data_g": (["XG"], np.random.rand(N)), "data_c": (["XC"], np.random.rand(N))},
        coords={
            "XG": (
                ["XG"],
                2 * np.pi / N * np.arange(0, N),
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "XC": (["XC"], 2 * np.pi / N * (np.arange(0, N) + 0.5), {"axis": "X"}),
        },
    )
    return ds


class TestInit:
    def test_default_init(self):

        # test initialisation
        axis = Axis(name="X", ds=periodic_1d(), coords={"center": "XC", "left": "XG"})

        # test attributes
        assert axis.name == "X"
        assert axis.coords == {"center": "XC", "left": "XG"}

        # test default values of attributes
        assert axis.default_shifts == {"left": "center", "center": "left"}
        assert axis.boundary == "periodic"

    def test_override_defaults(self):
        # test initialisation
        axis = Axis(
            name="foo",
            ds=periodic_1d(),
            coords={"center": "XC", "left": "XG"},
            # TODO does this make sense as default shifts?
            default_shifts={"left": "inner", "center": "outer"},
            boundary="fill",
        )

        # test attributes
        assert axis.name == "foo"
        assert axis.coords == {"center": "XC", "left": "XG"}

        # test default values of attributes
        # TODO (these deafult shift values make no physical sense)
        assert axis.default_shifts == {"left": "inner", "center": "outer"}
        assert axis.boundary == "fill"

    def test_inconsistent_dims(self):
        """Test when xgcm coord names are not present in dataset dims"""
        with pytest.raises(ValueError, match="Could not find dimension"):
            Axis(
                name="X",
                ds=periodic_1d(),
                coords={"center": "lat", "left": "lon"},
            )

    def test_invalid_args(self):

        # invalid defaults
        with pytest.raises(ValueError, match="Can't set the default"):
            Axis(
                name="foo",
                ds=periodic_1d(),
                coords={"center": "XC", "left": "XG"},
                default_shifts={"left": "left", "center": "center"},
            )

        with pytest.raises(ValueError, match="boundary must be one of"):
            Axis(
                name="foo",
                ds=periodic_1d(),
                coords={"center": "XC", "left": "XG"},
                boundary="blargh",
            )

    def test_repr(self):
        axis = Axis(name="X", ds=periodic_1d(), coords={"center": "XC", "left": "XG"})
        repr = axis.__repr__()

        assert repr.startswith("<xgcm.Axis 'X'")


def test_get_position_name():
    ds = periodic_1d()
    axis = Axis(name="X", ds=ds, coords={"center": "XC", "left": "XG"})

    da = ds["data_g"]
    pos, name = axis._get_position_name(da)
    assert pos == "left"
    assert name == "XG"


def test_get_axis_dim_num():
    ds = periodic_1d()
    axis = Axis(name="X", ds=ds, coords={"center": "XC", "left": "XG"})

    da = ds["data_g"]
    num = axis._get_axis_dim_num(da)
    assert num == da.get_axis_num("XG")


def test_axis_repr(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_name, axis in axis_objs.items():
        r = repr(axis).split("\n")
        assert r[0].startswith("<xgcm.Axis")
    # TODO: make this more complete


def test_assert_axes_equal():
    ...


# TODO raise similar errors
def test_axis_errors():
    ds = datasets["1d_left"]

    ds_noattr = ds.copy()
    del ds_noattr.XC.attrs["axis"]
    with pytest.raises(
        ValueError, match="Couldn't find a center coordinate for axis X"
    ):
        _ = Axis(ds_noattr, "X", periodic=True)

    del ds_noattr.XG.attrs["axis"]
    with pytest.raises(ValueError, match="Couldn't find any coordinates for axis X"):
        _ = Axis(ds_noattr, "X", periodic=True)

    ds_chopped = ds.copy().isel(XG=slice(None, 3))
    del ds_chopped["data_g"]
    with pytest.raises(ValueError, match="coordinate XG has incompatible length"):
        _ = Axis(ds_chopped, "X", periodic=True)

    ds_chopped.XG.attrs["c_grid_axis_shift"] = -0.5
    with pytest.raises(ValueError, match="coordinate XG has incompatible length"):
        _ = Axis(ds_chopped, "X", periodic=True)

    del ds_chopped.XG.attrs["c_grid_axis_shift"]
    with pytest.raises(
        ValueError,
        match="Found two coordinates without `c_grid_axis_shift` attribute for axis X",
    ):
        _ = Axis(ds_chopped, "X", periodic=True)

    ax = Axis(ds, "X", periodic=True)

    with pytest.raises(
        ValueError, match="Can't get neighbor pairs for the same position."
    ):
        ax.interp(ds.data_c, "center")

    with pytest.raises(
        ValueError, match="This axis doesn't contain a `right` position"
    ):
        ax.interp(ds.data_c, "right")

    # This case is broken, need to fix!
    # with pytest.raises(
    #    ValueError, match="`boundary=fill` is not allowed " "with periodic axis X."
    # ):
    #    ax.interp(ds.data_c, "left", boundary="fill")


def test_create_axis_no_coords(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)

    ds_drop = ds.drop_vars(list(ds.coords))

    for axis_name, axis_coords in expected["axes"].items():
        # now create the axis from scratch with no attributes OR coords
        ax2 = Axis(ds_drop, axis_name, coords=axis_coords)
        # and compare to the one created with attributes
        ax1 = axis_objs[axis_name]

        assert ax1.name == ax2.name
        for pos, coord in ax1.coords.items():
            assert pos in ax2.coords
        assert ax1._periodic == ax2._periodic
        assert ax1._default_shifts == ax2._default_shifts
        assert ax1._facedim == ax2._facedim


def test_axis_wrap_and_replace_2d(periodic_2d):
    ds, periodic, expected = periodic_2d
    axis_objs = _get_axes(ds)

    da_xc_yc = 0 * ds.XC * ds.YC + 1
    da_xc_yg = 0 * ds.XC * ds.YG + 1
    da_xg_yc = 0 * ds.XG * ds.YC + 1

    da_xc_yg_test = axis_objs["Y"]._wrap_and_replace_coords(
        da_xc_yc, da_xc_yc.data, "left"
    )
    assert da_xc_yg.equals(da_xc_yg_test)

    da_xg_yc_test = axis_objs["X"]._wrap_and_replace_coords(
        da_xc_yc, da_xc_yc.data, "left"
    )
    assert da_xg_yc.equals(da_xg_yc_test)


def test_axis_wrap_and_replace_nonperiodic(nonperiodic_1d):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, "X")

    da_c = 0 * ds.XC + 1
    da_g = 0 * ds.XG + 1

    to = (set(expected["axes"]["X"].keys()) - {"center"}).pop()

    da_g_test = axis._wrap_and_replace_coords(da_c, da_g.data, to)
    assert da_g.equals(da_g_test)

    da_c_test = axis._wrap_and_replace_coords(da_g, da_c.data, "center")
    assert da_c.equals(da_c_test)


# TODO move to wherever cumsum is tested
@pytest.mark.parametrize(
    "boundary", ["extend", "fill", pytest.param("extrapolate", marks=pytest.mark.xfail)]
)
def test_axis_cumsum(nonperiodic_1d, boundary):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, "X", periodic=periodic)

    axis_expected = expected["axes"]["X"]

    cumsum_g = axis.cumsum(ds.data_g, to="center", boundary=boundary)
    assert cumsum_g.dims == ds.data_c.dims
    # check default "to"
    assert cumsum_g.equals(axis.cumsum(ds.data_g, boundary=boundary))

    to = set(axis_expected).difference({"center"}).pop()
    cumsum_c = axis.cumsum(ds.data_c, to=to, boundary=boundary)
    assert cumsum_c.dims == ds.data_g.dims
    # check default "to"
    assert cumsum_c.equals(axis.cumsum(ds.data_c, boundary=boundary))

    cumsum_c_raw = np.cumsum(ds.data_c.data)
    cumsum_g_raw = np.cumsum(ds.data_g.data)

    if to == "right":
        np.testing.assert_allclose(cumsum_c.data, cumsum_c_raw)
        fill_value = 0.0 if boundary == "fill" else cumsum_g_raw[0]
        np.testing.assert_allclose(
            cumsum_g.data, np.hstack([fill_value, cumsum_g_raw[:-1]])
        )
    elif to == "left":
        np.testing.assert_allclose(cumsum_g.data, cumsum_g_raw)
        fill_value = 0.0 if boundary == "fill" else cumsum_c_raw[0]
        np.testing.assert_allclose(
            cumsum_c.data, np.hstack([fill_value, cumsum_c_raw[:-1]])
        )
    elif to == "inner":
        np.testing.assert_allclose(cumsum_c.data, cumsum_c_raw[:-1])
        fill_value = 0.0 if boundary == "fill" else cumsum_g_raw[0]
        np.testing.assert_allclose(cumsum_g.data, np.hstack([fill_value, cumsum_g_raw]))
    elif to == "outer":
        np.testing.assert_allclose(cumsum_g.data, cumsum_g_raw[:-1])
        fill_value = 0.0 if boundary == "fill" else cumsum_c_raw[0]
        np.testing.assert_allclose(cumsum_c.data, np.hstack([fill_value, cumsum_c_raw]))

    # not much point doing this...we don't have the right test datasets
    # to really test the errors
    # other_positions = {'left', 'right', 'inner', 'outer'}.difference({to})
    # for pos in other_positions:
    #     with pytest.raises(KeyError):
    #         axis.cumsum(ds.data_c, to=pos, boundary=boundary)


@pytest.mark.parametrize(
    "boundary", ["extend", "fill", pytest.param("extrapolate", marks=pytest.mark.xfail)]
)
@pytest.mark.parametrize("from_center", [True, False])
def test_axis_diff_and_interp_nonperiodic_1d(nonperiodic_1d, boundary, from_center):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, "X", periodic=periodic)

    dim_len_diff = len(ds.XG) - len(ds.XC)

    if from_center:
        to = (set(expected["axes"]["X"].keys()) - {"center"}).pop()
        coord_to = "XG"
        da = ds.data_c
    else:
        to = "center"
        coord_to = "XC"
        da = ds.data_g

    shift = expected.get("shift") or False

    data = da.data
    if (dim_len_diff == 1 and not from_center) or (dim_len_diff == -1 and from_center):
        data_left = data[:-1]
        data_right = data[1:]
    elif (dim_len_diff == 1 and from_center) or (
        dim_len_diff == -1 and not from_center
    ):
        data_left = _pad_left(data, boundary)
        data_right = _pad_right(data, boundary)
    elif (shift and not from_center) or (not shift and from_center):
        data_left = _pad_left(data[:-1], boundary)
        data_right = data
    else:
        data_left = data
        data_right = _pad_right(data[1:], boundary)

    # interpolate
    data_interp_expected = xr.DataArray(
        0.5 * (data_left + data_right), dims=[coord_to], coords={coord_to: ds[coord_to]}
    )
    data_interp = axis.interp(da, to, boundary=boundary)
    assert data_interp_expected.equals(data_interp)
    # check without "to" specified
    assert data_interp.equals(axis.interp(da, boundary=boundary))

    # difference
    data_diff_expected = xr.DataArray(
        data_right - data_left, dims=[coord_to], coords={coord_to: ds[coord_to]}
    )
    data_diff = axis.diff(da, to, boundary=boundary)
    assert data_diff_expected.equals(data_diff)
    # check without "to" specified
    assert data_diff.equals(axis.diff(da, boundary=boundary))

    # max
    data_max_expected = xr.DataArray(
        np.maximum(data_right, data_left),
        dims=[coord_to],
        coords={coord_to: ds[coord_to]},
    )
    data_max = axis.max(da, to, boundary=boundary)
    assert data_max_expected.equals(data_max)
    # check without "to" specified
    assert data_max.equals(axis.max(da, boundary=boundary))

    # min
    data_min_expected = xr.DataArray(
        np.minimum(data_right, data_left),
        dims=[coord_to],
        coords={coord_to: ds[coord_to]},
    )
    data_min = axis.min(da, to, boundary=boundary)
    assert data_min_expected.equals(data_min)
    # check without "to" specified
    assert data_min.equals(axis.min(da, boundary=boundary))


# this mega test covers all options for 2D data


@pytest.mark.parametrize(
    "boundary", ["extend", "fill", pytest.param("extrapolate", marks=pytest.mark.xfail)]
)
@pytest.mark.parametrize("axis_name", ["X", "Y"])
@pytest.mark.parametrize(
    "varname, this, to", [("data_c", "center", "left"), ("data_g", "left", "center")]
)
def test_axis_diff_and_interp_nonperiodic_2d(
    all_2d, boundary, axis_name, varname, this, to
):
    ds, periodic, _ = all_2d

    try:
        ax_periodic = axis_name in periodic
    except TypeError:
        ax_periodic = periodic

    boundary_arg = boundary if not ax_periodic else None
    axis = Axis(ds, axis_name, periodic=ax_periodic, boundary=boundary_arg)
    da = ds[varname]

    # everything is left shift
    data = ds[varname].data

    axis_num = da.get_axis_num(axis.coords[this])

    # lookups for numpy.pad
    numpy_pad_arg = {"extend": "edge", "fill": "constant"}
    # args for numpy.pad
    pad_left = (1, 0)
    pad_right = (0, 1)
    pad_none = (0, 0)

    if this == "center":
        if ax_periodic:
            data_left = np.roll(data, 1, axis=axis_num)
        else:
            pad_width = [
                pad_left if i == axis_num else pad_none for i in range(data.ndim)
            ]
            the_slice = tuple(
                [
                    slice(0, -1) if i == axis_num else slice(None)
                    for i in range(data.ndim)
                ]
            )
            data_left = np.pad(data, pad_width, numpy_pad_arg[boundary])[the_slice]
        data_right = data
    elif this == "left":
        if ax_periodic:
            data_left = data
            data_right = np.roll(data, -1, axis=axis_num)
        else:
            pad_width = [
                pad_right if i == axis_num else pad_none for i in range(data.ndim)
            ]
            the_slice = tuple(
                [
                    slice(1, None) if i == axis_num else slice(None)
                    for i in range(data.ndim)
                ]
            )
            data_right = np.pad(data, pad_width, numpy_pad_arg[boundary])[the_slice]
            data_left = data

    data_interp = 0.5 * (data_left + data_right)
    data_diff = data_right - data_left

    # determine new dims
    dims = list(da.dims)
    dims[axis_num] = axis.coords[to]
    coords = {dim: ds[dim] for dim in dims}

    da_interp_expected = xr.DataArray(data_interp, dims=dims, coords=coords)
    da_diff_expected = xr.DataArray(data_diff, dims=dims, coords=coords)

    da_interp = axis.interp(da, to)
    da_diff = axis.diff(da, to)

    assert da_interp_expected.equals(da_interp)
    assert da_diff_expected.equals(da_diff)

    if boundary_arg is not None:
        if boundary == "extend":
            bad_boundary = "fill"
        elif boundary == "fill":
            bad_boundary = "extend"

        da_interp_wrong = axis.interp(da, to, boundary=bad_boundary)
        assert not da_interp_expected.equals(da_interp_wrong)
        da_diff_wrong = axis.diff(da, to, boundary=bad_boundary)
        assert not da_diff_expected.equals(da_diff_wrong)


# helper function to produce axes from datasets
def _get_axes(ds):
    all_axes = {ds[c].attrs["axis"] for c in ds.dims if "axis" in ds[c].attrs}
    axis_objs = {ax: Axis(ds, ax) for ax in all_axes}
    return axis_objs


def test_create_axis_no_comodo(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)

    # now strip out the metadata
    ds_noattr = ds.copy()
    for var in ds.variables:
        ds_noattr[var].attrs.clear()

    for axis_name, axis_coords in expected["axes"].items():
        # now create the axis from scratch with no attributes
        ax2 = Axis(ds_noattr, axis_name, coords=axis_coords)
        # and compare to the one created with attributes
        ax1 = axis_objs[axis_name]

        assert ax1.name == ax2.name
        for pos, coord_name in ax1.coords.items():
            assert pos in ax2.coords
            assert coord_name == ax2.coords[pos]
        assert ax1._periodic == ax2._periodic
        assert ax1._default_shifts == ax2._default_shifts
        assert ax1._facedim == ax2._facedim
