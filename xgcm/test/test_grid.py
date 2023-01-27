import numpy as np
import pytest
import xarray as xr

from xgcm.grid import Grid

from .datasets import all_2d  # noqa: F401
from .datasets import all_datasets  # noqa: F401
from .datasets import datasets  # noqa: F401
from .datasets import datasets_grid_metric  # noqa: F401
from .datasets import nonperiodic_1d  # noqa: F401
from .datasets import nonperiodic_2d  # noqa: F401
from .datasets import periodic_1d  # noqa: F401
from .datasets import periodic_2d  # noqa: F401


def _assert_axes_equal(ax1, ax2):
    assert ax1.name == ax2.name
    for pos, coord in ax1.coords.items():
        assert pos in ax2.coords
        assert coord == ax2.coords[pos]
    assert ax1._periodic == ax2._periodic
    assert ax1._default_shifts == ax2._default_shifts
    # TODO: make this work...
    # assert ax1._facedim == ax2._facedim
    # assert ax1._connections == ax2._connections


class TestInvalidGrid:
    def test_raise_non_str_axis_name(self, periodic_2d):
        with pytest.raises(TypeError, match="name argument must be of type str"):
            ds, *_ = datasets_grid_metric("C")
            Grid(ds, coords={1: {"left": "XG"}})

    def test_non_ds_type(self):
        with pytest.raises(
            TypeError, match="ds argument must be of type xarray.Dataset"
        ):
            Grid(4, coords={"ax1": {"left": "XG"}})

    def test_invalid_position_name(self):
        with pytest.raises(ValueError):
            ds, *_ = datasets_grid_metric("C")
            Grid(ds, coords={"ax1": {"outer space": "XG"}})

    def test_nonexistent_dimension(self):
        with pytest.raises(ValueError):
            ds, *_ = datasets_grid_metric("C")
            Grid(ds, coords={"ax1": {"center": "XGEEEEEEEE"}})

    @pytest.mark.xfail(reason="Not yet implemented")
    def test_duplicate_values(self):
        with pytest.raises(ValueError):
            ds, *_ = datasets_grid_metric("C")
            Grid(ds, coords={"ax1": {"left": "xt", "right": "xt"}})

        with pytest.raises(ValueError):
            ds, *_ = datasets_grid_metric("C")
            Grid(ds, coords={"ax1": {"left": "xt"}, "ax2": {"right": "xt"}})

    def test_inconsistent_lengths(self):
        # TODO incompatible lengths (e.g. outer dim not 1 element longer than center dim)
        ...


class TestGrid:
    def test_init(self):
        ...

    def test_kwargs_mapped_over_multiple_axes(self):
        ...

    def test_repr(self):
        ...

    def test_properties(self):
        # test boundaries

        # test face connections

        # test metrics
        ...


def periodic_1d_single_pos():
    N = 5
    ds = xr.Dataset(
        {"data_c": (["XC"], np.random.rand(N))},
        coords={
            "XC": (["XC"], 2 * np.pi / N * (np.arange(0, N) + 0.5), {"axis": "X"}),
        },
    )
    return ds


def test_raise_on_operation_not_valid_for_same_position():
    ds = periodic_1d_single_pos()
    grid = Grid(ds, coords={"X": {"center": "XC"}})
    with pytest.raises(
        NotImplementedError, match="Could not find any pre-defined diff grid ufuncs"
    ):
        grid.diff(ds.data_c, "X", to="center")


@pytest.mark.parametrize(
    "padding",
    [
        "fill",
        "extend",
        {"X": "fill", "Y": "extend"},
    ],
)
@pytest.mark.parametrize("fill_value", [0, 1.0])
def test_grid_create(all_datasets, padding, fill_value):
    ds, periodic, expected = all_datasets
    grid = Grid(ds, periodic=periodic)

    assert grid is not None

    for ax in grid.axes.values():
        assert ax.padding is "periodic" if periodic else "fill"
        assert ax.fill_value == 0.0

    grid = Grid(ds, periodic=periodic, padding=padding, fill_value=fill_value)

    for name, ax in grid.axes.items():
        if isinstance(padding, dict):
            expected = padding.get(name)
        else:
            expected = padding
        assert ax.padding == expected

        if isinstance(fill_value, dict):
            expected = fill_value.get(name)
        else:
            expected = fill_value
        assert ax.fill_value == expected


def test_create_grid_no_comodo(all_datasets):
    ds, periodic, expected = all_datasets
    grid_expected = Grid(ds, periodic=periodic)

    ds_noattr = ds.copy()
    for var in ds.variables:
        ds_noattr[var].attrs.clear()

    coords = expected["axes"]
    grid = Grid(ds_noattr, periodic=periodic, coords=coords)

    for axis_name_expected in grid_expected.axes:
        axis_expected = grid_expected.axes[axis_name_expected]
        axis_actual = grid.axes[axis_name_expected]
        _assert_axes_equal(axis_expected, axis_actual)


def test_grid_no_coords(periodic_1d):
    """Ensure that you can use xgcm with Xarray datasets that don't have dimension coordinates."""
    ds, periodic, expected = periodic_1d
    ds_nocoords = ds.drop_vars(list(ds.dims.keys()))

    coords = expected["axes"]
    grid = Grid(ds_nocoords, periodic=periodic, coords=coords)

    diff = grid.diff(ds["data_c"], "X")
    assert len(diff.coords) == 0
    interp = grid.interp(ds["data_c"], "X")
    assert len(interp.coords) == 0


def test_grid_repr(all_datasets):
    ds, periodic, _ = all_datasets
    grid = Grid(ds, periodic=periodic)
    r = repr(grid).split("\n")
    assert r[0] == "<xgcm.Grid>"


@pytest.mark.parametrize(
    "padding", ["extend", "fill", pytest.param("extrapolate", marks=pytest.mark.xfail)]
)
def test_cumsum(nonperiodic_1d, padding):
    ds, periodic, expected = nonperiodic_1d
    grid = Grid(ds, padding="periodic")

    cumsum_g = grid.cumsum(ds.data_g, axis="X", to="center", padding=padding)

    to = grid.axes["X"].default_shifts["center"]
    cumsum_c = grid.cumsum(ds.data_c, axis="X", to=to, padding=padding)

    cumsum_c_raw = np.cumsum(ds.data_c.data)
    cumsum_g_raw = np.cumsum(ds.data_g.data)

    if to == "right":
        np.testing.assert_allclose(cumsum_c.data, cumsum_c_raw)
        fill_value = 0.0 if padding == "fill" else cumsum_g_raw[0]
        np.testing.assert_allclose(
            cumsum_g.data, np.hstack([fill_value, cumsum_g_raw[:-1]])
        )
    elif to == "left":
        np.testing.assert_allclose(cumsum_g.data, cumsum_g_raw)
        fill_value = 0.0 if padding == "fill" else cumsum_c_raw[0]
        np.testing.assert_allclose(
            cumsum_c.data, np.hstack([fill_value, cumsum_c_raw[:-1]])
        )
    elif to == "inner":
        np.testing.assert_allclose(cumsum_c.data, cumsum_c_raw[:-1])
        fill_value = 0.0 if padding == "fill" else cumsum_g_raw[0]
        np.testing.assert_allclose(cumsum_g.data, np.hstack([fill_value, cumsum_g_raw]))
    elif to == "outer":
        np.testing.assert_allclose(cumsum_g.data, cumsum_g_raw[:-1])
        fill_value = 0.0 if padding == "fill" else cumsum_c_raw[0]
        np.testing.assert_allclose(cumsum_c.data, np.hstack([fill_value, cumsum_c_raw]))

    # not much point doing this...we don't have the right test datasets
    # to really test the errors
    # other_positions = {'left', 'right', 'inner', 'outer'}.difference({to})
    # for pos in other_positions:
    #     with pytest.raises(KeyError):
    #         axis.cumsum(ds.data_c, to=pos, padding=padding)


@pytest.mark.parametrize(
    "func",
    ["interp", "max", "min", "diff", "cumsum"],
)
@pytest.mark.parametrize(
    "padding",
    [
        "fill",
        pytest.param("extrapolate", marks=pytest.mark.xfail),
        "extend",
        {"X": "fill", "Y": "extend"},
        {"X": "extend", "Y": "fill"},
    ],
)
def test_dask_vs_eager(all_datasets, func, padding):
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords)
    grid_method = getattr(grid, func)
    eager_result = grid_method(ds.tracer, "X", padding=padding)

    ds = ds.chunk({"xt": 1, "yt": 1, "time": 1, "zt": 1})
    grid = Grid(ds, coords=coords)
    grid_method = getattr(grid, func)
    dask_result = grid_method(ds.tracer, "X", padding=padding).compute()

    xr.testing.assert_allclose(dask_result, eager_result)


def test_grid_dict_input_padding_fill(nonperiodic_1d):
    """Test axis kwarg input functionality using dict input"""
    ds, _, _ = nonperiodic_1d
    grid_direct = Grid(ds, periodic=False, padding="fill", fill_value=5)
    grid_dict = Grid(ds, periodic=False, padding={"X": "fill"}, fill_value={"X": 5})
    assert grid_direct.axes["X"].fill_value == grid_dict.axes["X"].fill_value
    assert grid_direct.axes["X"].padding == grid_dict.axes["X"].padding


def test_invalid_padding_error():
    ds = datasets["1d_left"]
    with pytest.raises(ValueError):
        Grid(ds, padding="bad")
    with pytest.raises(ValueError):
        Grid(ds, padding={"X": "bad"})
    with pytest.raises(ValueError):
        Grid(ds, padding={"X": 0})
    with pytest.raises(ValueError):
        Grid(ds, padding=0)


def test_invalid_fill_value_error():
    ds = datasets["1d_left"]
    with pytest.raises(TypeError):
        Grid(ds, fill_value="bad")
    with pytest.raises(TypeError):
        Grid(ds, fill_value={"X": "bad"})


@pytest.mark.parametrize(
    "funcname",
    [
        "diff",
        "interp",
        "min",
        "max",
        "integrate",
        "average",
        "cumsum",
        "cumint",
        "derivative",
        # TODO: we can get rid of many of these after the release. With the grid_ufunc logic many of these go through the same codepath
        # e.g. diff/interp/min/max all are the same, so we can probably reduce this to diff, cumsum, integrate, derivative, cumint
    ],
)
@pytest.mark.parametrize("gridtype", ["B", "C"])
def test_keep_coords(funcname, gridtype):
    ds, coords, metrics = datasets_grid_metric(gridtype)
    ds = ds.assign_coords(yt_bis=ds["yt"], xt_bis=ds["xt"])
    grid = Grid(ds, coords=coords, metrics=metrics)

    func = getattr(grid, funcname)
    for axis_name in grid.axes.keys():
        result = func(ds.tracer, axis_name)
        base_coords = list(result.dims)
        augmented_coords = [
            c
            for c in ds.coords
            if set(ds[c].dims).issubset(result.dims) and c not in result.dims
        ]

        if funcname in ["integrate", "average"]:
            assert set(result.coords) == set(base_coords + augmented_coords)
        else:
            assert set(result.coords) == set(base_coords)

        # TODO: why is the behavior different for integrate and average?
        if funcname not in ["integrate", "average"]:
            result = func(ds.tracer, axis_name, keep_coords=False)
            assert set(result.coords) == set(base_coords)

            result = func(ds.tracer, axis_name, keep_coords=True)
            assert set(result.coords) == set(base_coords + augmented_coords)


def test_keep_coords_deprecation():
    ds, coords, metrics = datasets_grid_metric("B")
    ds = ds.assign_coords(yt_bis=ds["yt"], xt_bis=ds["xt"])
    grid = Grid(ds, coords=coords, metrics=metrics)
    for axis_name in grid.axes.keys():
        with pytest.warns(DeprecationWarning):
            grid.diff(ds.tracer, axis_name, keep_coords=False)


def test_padding_kwarg_same_as_grid_constructor_kwarg():
    ds = datasets["2d_left"]
    grid1 = Grid(ds, periodic=False)
    grid2 = Grid(ds, periodic=False, padding={"X": "fill", "Y": "fill"})

    actual1 = grid1.interp(ds.data_g, ("X", "Y"), padding={"X": "fill", "Y": "fill"})
    actual2 = grid2.interp(ds.data_g, ("X", "Y"))

    xr.testing.assert_identical(actual1, actual2)


@pytest.mark.parametrize(
    "metric_axes,metric_name",
    [
        (["Y", "X"], "area_n"),
        ("X", "dx_t"),
        ("Y", "dy_ne"),
        (["Y", "X"], "dy_n"),
        (["X"], "tracer"),
    ],
)
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize(
    "padding, padding_expected",
    [
        ({"X": "fill", "Y": "fill"}, {"X": "fill", "Y": "fill"}),
        ({"X": "extend", "Y": "extend"}, {"X": "extend", "Y": "extend"}),
        pytest.param(
            {"X": "extrapolate", "Y": "extrapolate"},
            {"X": "extrapolate", "Y": "extrapolate"},
            marks=pytest.mark.xfail(
                reason="padding via extrapolation not yet supported in grid_ufunc refactor"
            ),
        ),
        ("fill", {"X": "fill", "Y": "fill"}),
        ("extend", {"X": "extend", "Y": "extend"}),
        pytest.param(
            "extrapolate",
            {"X": "extrapolate", "Y": "extrapolate"},
            marks=pytest.mark.xfail(
                reason="padding via extrapolation not yet supported in grid_ufunc refactor"
            ),
        ),
        ({"X": "extend", "Y": "fill"}, {"X": "extend", "Y": "fill"}),
        pytest.param(
            {"X": "extrapolate", "Y": "fill"},
            {"X": "extrapolate", "Y": "fill"},
            marks=pytest.mark.xfail(
                reason="padding via extrapolation not yet supported in grid_ufunc refactor"
            ),
        ),
        pytest.param(
            "fill",
            {"X": "fill", "Y": "extend"},
            marks=pytest.mark.xfail,
            id="padding not equal to padding_expected",
        ),
    ],
)
@pytest.mark.parametrize("fill_value", [None, 0.1])
def test_interp_like(
    metric_axes, metric_name, periodic, padding, padding_expected, fill_value
):

    ds, coords, _ = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, periodic=periodic)
    grid.set_metrics(metric_axes, metric_name)
    metric_available = grid._metrics.get(frozenset(metric_axes), None)
    metric_available = metric_available[0]
    interp_metric = grid.interp_like(
        metric_available, ds.u, padding=padding, fill_value=fill_value
    )
    expected_metric = grid.interp(
        ds[metric_name], metric_axes, padding=padding_expected, fill_value=fill_value
    )

    xr.testing.assert_allclose(interp_metric, expected_metric)


def test_input_not_dims():
    data = np.random.rand(4, 5)
    coord = np.random.rand(4, 5)
    ds = xr.DataArray(
        data, dims=["x", "y"], coords={"c": (["x", "y"], coord)}
    ).to_dataset(name="data")
    with pytest.raises(ValueError, match="Could not find dimension"):
        Grid(ds, coords={"X": {"center": "c"}})


def test_input_dim_notfound():
    data = np.random.rand(4, 5)
    coord = np.random.rand(4, 5)
    ds = xr.DataArray(
        data, dims=["x", "y"], coords={"c": (["x", "y"], coord)}
    ).to_dataset(name="data")
    msg = r"Could not find dimension `other` \(for the `center` position on axis `X`\) in input dataset."
    with pytest.raises(ValueError, match=msg):
        Grid(ds, coords={"X": {"center": "other"}})


@pytest.mark.parametrize(
    "funcname",
    [
        "interp",
        "diff",
        "min",
        "max",
        "cumsum",
        "derivative",
        "cumint",
    ],
)
@pytest.mark.parametrize(
    "padding",
    ["fill", "extend"],
)
@pytest.mark.parametrize(
    "fill_value",
    [0, 10, None],
)
def test_padding_global_input(funcname, padding, fill_value):
    """Test that globally defined padding values result in
    the same output as when the parameters are defined the grid methods
    """
    ds, coords, metrics = datasets_grid_metric("C")
    axis = "X"
    # Test results by globally specifying fill value/padding on grid object
    grid_global = Grid(
        ds,
        coords=coords,
        metrics=metrics,
        periodic=False,
        padding=padding,
        fill_value=fill_value,
    )
    func_global = getattr(grid_global, funcname)
    global_result = func_global(ds.tracer, axis)

    # Test results by manually specifying fill value/padding on grid method
    grid_manual = Grid(
        ds, coords=coords, metrics=metrics, periodic=False, padding=padding
    )

    func_manual = getattr(grid_manual, funcname)
    manual_result = func_manual(ds.tracer, axis, padding=padding, fill_value=fill_value)
    xr.testing.assert_allclose(global_result, manual_result)


class TestInputErrorGridMethods:
    def test_multiple_keys_vector_input(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector components provided as dictionaries should contain exactly one key/value pair. .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            grid.diff({"X": xr.DataArray(), "Y": xr.DataArray()}, "X")

    def test_wrong_input_type_scalar(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "All data arguments must be either a DataArray or Dictionary .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            grid.diff("not_a_dataarray", "X")

    def test_wrong_input_type_vector(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Dictionary inputs must have a DataArray as value. Got .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            grid.diff({"X": "not_a_dataarray"}, "X")

    def test_wrong_axis_vector_input_axis(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector component with unknown axis provided. Grid has axes .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            grid.diff({"wrong": xr.DataArray()}, "X")


class TestInputErrorApplyAsGridUfunc:
    def test_multiple_keys_vector_input(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector components provided as dictionaries should contain exactly one key/value pair. .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            grid.apply_as_grid_ufunc(
                lambda x: x, {"X": xr.DataArray(), "Y": xr.DataArray()}, "X"
            )

    def test_wrong_input_type_scalar(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "All data arguments must be either a DataArray or Dictionary .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            grid.apply_as_grid_ufunc(lambda x: x, "not_a_dataarray", "X")

    def test_wrong_input_type_vector(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Dictionary inputs must have a DataArray as value. Got .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            grid.apply_as_grid_ufunc(lambda x: x, {"X": "not_a_dataarray"}, "X")

    def test_wrong_axis_vector_input_axis(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector component with unknown axis provided. Grid has axes .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            grid.apply_as_grid_ufunc(lambda x: x, {"wrong": xr.DataArray()}, "X")

    def test_vector_input_data_other_mismatch(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = (
            "When providing multiple input arguments, `other_component`"
            " needs to provide one dictionary per input"
        )
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            # Passing 3 args and 2 other components should fail.
            grid.apply_as_grid_ufunc(
                lambda x: x,
                {"X": xr.DataArray()},
                {"Y": xr.DataArray()},
                {"Z": xr.DataArray()},
                axis="X",
                other_component=[{"X": xr.DataArray()}, {"Y": xr.DataArray()}],
            )

    def test_wrong_input_type_vector_multi_input(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Dictionary inputs must have a DataArray as value. Got .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            # Passing 3 args and 2 other components should fail.
            grid.apply_as_grid_ufunc(
                lambda x: x,
                {"X": xr.DataArray()},
                {"Y": "not_a_data_array"},
                axis="X",
                other_component=[{"X": xr.DataArray()}, {"Y": xr.DataArray()}],
            )

    def test_wrong_axis_vector_input_axis_multi_input(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector component with unknown axis provided. Grid has axes .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            # Passing 3 args and 2 other components should fail.
            grid.apply_as_grid_ufunc(
                lambda x: x,
                {"X": xr.DataArray()},
                {"Y": xr.DataArray()},
                axis="X",
                other_component=[{"wrong": xr.DataArray()}, {"Y": xr.DataArray()}],
            )
