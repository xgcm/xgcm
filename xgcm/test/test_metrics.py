import functools
import operator

import numpy as np
import pytest
import xarray as xr

from xgcm.grid import Grid
from xgcm.metrics import iterate_axis_combinations
from xgcm.test.datasets import datasets_grid_metric


def test_multiple_metrics_per_axis():
    # copied from test_derivatives.py - should refactor
    dx = 10.0
    ds = xr.Dataset(
        {
            "foo": (("XC",), [1.0, 2.0, 4.0, 3.0]),
            "bar": (("XG",), [10.0, 20.0, 30.0, 40.0]),
        },
        coords={
            "XC": (("XC",), [0.5, 1.5, 2.5, 3.5]),
            "XG": (("XG",), [0, 1.0, 2.0, 3.0]),
            "dXC": (("XC",), [dx, dx, dx, dx]),
            "dXG": (("XG",), [dx, dx, dx, dx]),
        },
    )

    grid = Grid(
        ds,
        coords={"X": {"center": "XC", "left": "XG"}},
        metrics={("X",): ["dXC", "dXG"]},
        periodic=True,
    )

    assert grid.get_metric(ds.foo, ("X",)).equals(ds.dXC.reset_coords(drop=True))
    assert grid.get_metric(ds.bar, ("X",)).equals(ds.dXG.reset_coords(drop=True))


def test_metrics_2d_grid():
    # this is a uniform grid
    # a non-uniform grid would provide a more rigorous test
    dx = 10.0
    dy = 11.0
    area = 120.0
    ny, nx = 7, 9
    ds = xr.Dataset(
        {"foo": (("YC", "XC"), np.ones((ny, nx)))},
        coords={
            "XC": (("XC",), np.arange(nx)),
            "dX": (("XC",), np.full(nx, dx)),
            "YC": (("YC",), np.arange(ny)),
            "dY": (("YC",), np.full(ny, dy)),
            "area": (("YC", "XC"), np.full((ny, nx), area)),
        },
    )

    grid = Grid(
        ds,
        coords={"X": {"center": "XC"}, "Y": {"center": "YC"}},
        metrics={("X",): ["dX"], ("Y",): ["dY"], ("X", "Y"): ["area"]},
    )

    assert ds.dX.reset_coords(drop=True).equals(grid.get_metric(ds.foo, ("X",)))
    assert ds.dY.reset_coords(drop=True).equals(grid.get_metric(ds.foo, ("Y",)))
    assert ds.area.reset_coords(drop=True).equals(grid.get_metric(ds.foo, ("X", "Y")))
    assert ds.area.reset_coords(drop=True).equals(grid.get_metric(ds.foo, ("Y", "X")))

    # try with no area metric:
    grid = Grid(
        ds,
        coords={"X": {"center": "XC"}, "Y": {"center": "YC"}},
        metrics={("X",): ["dX"], ("Y",): ["dY"]},
    )

    dxdy = (ds.dX * ds.dY).reset_coords(drop=True).transpose("YC", "XC")
    actual = grid.get_metric(ds.foo, ("Y", "X")).transpose("YC", "XC")
    assert dxdy.equals(actual)


@pytest.mark.parametrize(
    "key, metric_vars",
    [
        (("X",), ["dx_t"]),  # recommended way
        ("X", "dx_t"),
        (("X", "Y"), ["area_t"]),
        (
            ("X", "Y"),
            ["area_t", "area_t"],
        ),  # this should also trigger an error, it does not
        (("X", "Y"), ["area_t", "area_e", "area_n", "area_ne"]),
        (("X", "Y", "Z"), ["volume_t"]),
    ],
)
def test_assign_metric(key, metric_vars):
    ds, coords, _ = datasets_grid_metric("C")
    _ = Grid(ds, coords=coords, metrics={key: metric_vars})


@pytest.mark.parametrize(
    "axes, expected",
    [
        (
            ("X", "Y"),
            (
                (frozenset({"X", "Y"}),),
                (frozenset({"X"}), frozenset({"Y"})),
                (frozenset({"Y"}), frozenset({"X"})),
            ),
        ),
        (
            ("X", "Y", "Z"),
            (
                (frozenset({"Y", "X", "Z"}),),
                (frozenset({"Z"}), frozenset({"X"}), frozenset({"Y"})),
                (frozenset({"X"}), frozenset({"Y"}), frozenset({"Z"})),
                (frozenset({"Y"}), frozenset({"X"}), frozenset({"Z"})),
                (frozenset({"Y", "Z"}), frozenset({"X"})),
                (frozenset({"Y", "X"}), frozenset({"Z"})),
                (frozenset({"X", "Z"}), frozenset({"Y"})),
            ),
        ),
    ],
)
def test_iterate_axis_combinations(axes, expected):

    actual = list(iterate_axis_combinations(axes))

    expected = [set(e) for e in expected]
    actual = [set(a) for a in actual]

    # comparing these is a pain since output order is not guaranteed.
    # instead we ensure actual and expected have the same number of elements
    # and that every element in actual is present in expected and vice-versa
    assert len(actual) == len(expected)
    for a in actual:
        assert set(a) in expected
    for e in expected:
        assert set(e) in actual


@pytest.mark.parametrize(
    "axes, data_var, drop_vars, metric_expected_list",
    [
        ("X", "tracer", None, ["dx_t"]),
        (["X", "Y"], "tracer", None, ["area_t"]),
        (
            ("X", "Y"),
            "tracer",
            None,
            ["area_t"],
        ),
        # should we be able to pass a tuple as well as a list?
        (["X", "Y", "Z"], "tracer", None, ["volume_t"]),
        (["X"], "u", None, ["dx_e"]),
        (["X", "Y"], "u", None, ["area_e"]),
    ],
)
def test_get_metric_orig(axes, data_var, drop_vars, metric_expected_list):
    ds, coords, metrics = datasets_grid_metric("C")
    # drop metrics according to drop_vars input, and remove from metrics input
    if drop_vars:
        print(drop_vars)
        ds = ds.drop_vars(drop_vars)
        metrics = {k: [a for a in v if a not in drop_vars] for k, v in metrics.items()}

    grid = Grid(ds, coords=coords, metrics=metrics)
    metric = grid.get_metric(ds[data_var], axes)
    expected_metrics = [ds[me].reset_coords(drop=True) for me in metric_expected_list]
    expected = functools.reduce(operator.mul, expected_metrics, 1)
    assert metric.equals(expected)


def test_get_metric_with_conditions_01():
    # Condition 1: metric with matching axes and dimensions exist
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, metrics=metrics)
    get_metric = grid.get_metric(ds.v, ("X", "Y"))

    expected_metric = ds["area_n"].reset_coords(drop=True)

    xr.testing.assert_allclose(get_metric, expected_metric)


@pytest.mark.parametrize("periodic", [True, False])
def test_get_metric_with_conditions_02a(periodic):
    # Condition 2, case a: interpolate metric with matching axis to desired dimensions
    ds, coords, _ = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, periodic=periodic, boundary="extend")
    grid.set_metrics(("X", "Y"), "area_e")

    get_metric = grid.get_metric(ds.v, ("X", "Y"))

    expected_metric = grid.interp(ds["area_e"], ("X", "Y"))

    xr.testing.assert_allclose(get_metric, expected_metric)


def test_get_metric_with_conditions_02b():
    # Condition 2, case b: get_metric should select for the metric with matching axes and interpolate from there,
    # even if other metrics in the desired positions are available
    ds, coords, _ = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords)
    grid.set_metrics(("X", "Y"), "area_e")
    grid.set_metrics(("X"), "dx_n")
    grid.set_metrics(("Y"), "dx_n")

    get_metric = grid.get_metric(ds.v, ("X", "Y"))

    expected_metric = grid.interp(ds["area_e"], ("X", "Y"))

    xr.testing.assert_allclose(get_metric, expected_metric)


def test_get_metric_with_conditions_03a():
    # Condition 3: use provided metrics with matching dimensions to calculate for required metric
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords)

    grid.set_metrics(("X"), "dx_n")
    grid.set_metrics(("Y"), "dy_n")

    get_metric = grid.get_metric(ds.v, ("X", "Y"))

    metric_var_1 = ds.dx_n
    metric_var_2 = ds.dy_n
    expected_metric = (metric_var_1 * metric_var_2).reset_coords(drop=True)

    xr.testing.assert_allclose(get_metric, expected_metric)


def test_get_metric_with_conditions_03b():
    # Condition 3: use provided metrics with matching dimensions to calculate for required metric
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords)
    grid.set_metrics(("X", "Y"), "area_t")
    grid.set_metrics(("Z"), "dz_t")

    get_metric = grid.get_metric(ds.tracer, ("X", "Y", "Z"))

    metric_var_1 = ds.area_t
    metric_var_2 = ds.dz_t
    expected_metric = (metric_var_1 * metric_var_2).reset_coords(drop=True)

    xr.testing.assert_allclose(get_metric, expected_metric)


def test_get_metric_with_conditions_04a():
    # Condition 4, case a: 1 metric on the wrong position (must interpolate before multiplying)
    ds, coords, _ = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords)
    grid.set_metrics(("X"), "dx_t")
    grid.set_metrics(("Y"), "dy_n")

    get_metric = grid.get_metric(ds.v, ("X", "Y"))

    interp_metric = grid.interp(ds.dx_t, "Y")
    expected_metric = (interp_metric * ds.dy_n).reset_coords(drop=True)

    xr.testing.assert_allclose(get_metric, expected_metric)


def test_get_metric_with_conditions_04b():
    # Condition 4, case b: 2 metrics in the wrong position (must interpolate both before multiplying)
    ds, coords, _ = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords)
    grid.set_metrics(("X"), "dx_t")
    grid.set_metrics(("Y"), "dy_t")

    get_metric = grid.get_metric(ds.v, ("X", "Y"))

    interp_metric_1 = grid.interp(ds.dx_t, "Y")
    interp_metric_2 = grid.interp(ds.dy_t, "Y")
    expected_metric = (interp_metric_1 * interp_metric_2).reset_coords(drop=True)

    xr.testing.assert_allclose(get_metric, expected_metric)


def test_set_metric():

    ds, coords, metrics = datasets_grid_metric("C")
    expected_metrics = {k: [ds[va] for va in v] for k, v in metrics.items()}

    grid = Grid(ds, coords=coords, metrics=metrics)

    grid_manual = Grid(ds, coords=coords)

    for key, value in metrics.items():
        grid_manual.set_metrics(key, value)

    assert len(grid._metrics) > 0

    for k, v in expected_metrics.items():

        k = frozenset(k)
        assert k in grid._metrics.keys()
        assert k in grid_manual._metrics.keys()

        for metric_expected, metric in zip(v, grid_manual._metrics[k]):
            xr.testing.assert_allclose(metric_expected.reset_coords(drop=True), metric)

        for metric_expected, metric in zip(v, grid._metrics[k]):
            xr.testing.assert_allclose(metric_expected.reset_coords(drop=True), metric)


@pytest.mark.parametrize(
    "metric_axes, exist_metric_varname, add_metric_varname, expected_varname",
    [
        (
            "X",
            ["dx_t", "dx_n", "dx_e", "dx_ne"],
            ["dx_n_overwrite"],
            ["dx_t", "dx_n_overwrite", "dx_e", "dx_ne"],
        ),
        (
            ("Y", "X"),
            ["area_t", "area_n", "area_e", "area_ne"],
            ["area_n_overwrite"],
            ["area_t", "area_n_overwrite", "area_e", "area_ne"],
        ),
        # overwrite 1 existing metric, append 1 new metric
        (
            "X",
            ["dx_t", "dx_n", "dx_e"],
            ["dx_n_overwrite", "dx_ne"],
            ["dx_t", "dx_n_overwrite", "dx_e", "dx_ne"],
        ),
    ],
)
def test_set_metric_overwrite_true(
    metric_axes, exist_metric_varname, add_metric_varname, expected_varname
):
    ds, coords, metrics = datasets_grid_metric("C")

    ds = ds.assign_coords({add_metric_varname[0]: ds[exist_metric_varname[1]] * 10})

    metrics = {
        k: [m for m in v if m in exist_metric_varname] for k, v in metrics.items()
    }
    grid = Grid(ds, coords=coords, metrics=metrics)

    for av in add_metric_varname:
        grid.set_metrics(metric_axes, av, overwrite=True)

    key = frozenset(list(metric_axes))
    set_metric = grid._metrics.get(key)

    expected_metric = []
    for ev in expected_varname:
        metric_var = ds[ev].reset_coords(drop=True)
        expected_metric.append(metric_var)

    assert len(set_metric) == len(expected_metric)

    for i in range(len(set_metric)):
        assert set_metric[i].equals(expected_metric[i])


@pytest.mark.parametrize(
    "metric_axes,overwrite_metric,add_metric",
    [("X", "dx_t_overwrite", "dx_t"), ("X", "dx_e", None)],
)
def test_set_metric_value_errors(metric_axes, overwrite_metric, add_metric):
    ds, coords, metrics = datasets_grid_metric("C")

    if add_metric is not None:
        ds = ds.assign_coords({overwrite_metric: ds[add_metric] * 10})

    grid = Grid(ds, coords=coords, metrics=metrics)

    with pytest.raises(ValueError, match="setting overwrite=True."):
        grid.set_metrics(metric_axes, overwrite_metric)


@pytest.mark.parametrize(
    "metric_axes,add_metric",
    [("X", "foo"), (("U", "V"), "area_n")],
)
def test_set_metric_key_errors(metric_axes, add_metric):
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, metrics=metrics)

    if len(metric_axes) == 1:
        with pytest.raises(KeyError, match="not found in dataset."):
            grid.set_metrics(metric_axes, add_metric)
    else:
        with pytest.raises(KeyError, match="not compatible with grid axes"):
            grid.set_metrics(metric_axes, add_metric)
