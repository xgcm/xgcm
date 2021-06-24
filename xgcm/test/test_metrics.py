import functools
import operator

import numpy as np
import pytest
import xarray as xr

from xgcm.grid import Grid
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
    "axes, data_var, drop_vars, metric_expected_list, expected_error",
    [
        ("X", "tracer", None, ["dx_t"], None),
        (["X", "Y"], "tracer", None, ["area_t"], None),
        (
            ("X", "Y"),
            "tracer",
            None,
            ["area_t"],
            None,
        ),  # should we be able to pass a tuple as well as a list?
        (["X", "Y", "Z"], "tracer", None, ["volume_t"], None),
        (["X"], "u", None, ["dx_e"], None),
        (["X", "Y"], "u", None, ["area_e"], None),
    ],
)
def test_get_metric_orig(
    axes, data_var, drop_vars, metric_expected_list, expected_error
):
    ds, coords, metrics = datasets_grid_metric("C")
    # drop metrics according to drop_vars input, and remove from metrics input
    if drop_vars:
        print(drop_vars)
        ds = ds.drop_vars(drop_vars)
        metrics = {k: [a for a in v if a not in drop_vars] for k, v in metrics.items()}

    grid = Grid(ds, coords=coords, metrics=metrics)
    if expected_error:
        with pytest.raises(expected_error):
            metric = grid.get_metric(ds[data_var], axes)
    else:
        metric = grid.get_metric(ds[data_var], axes)
        expected_metrics = [
            ds[me].reset_coords(drop=True) for me in metric_expected_list
        ]
        expected = functools.reduce(operator.mul, expected_metrics, 1)
        assert metric.equals(expected)


@pytest.mark.parametrize(
    "set_metrics,metric_axes,ds_array,ds_expected,set_metric_vars,expected_metric",
    [
        (
            "metrics",
            ("X", "Y"),
            "v",
            "area_n",
            None,
            "cond1",
        ),  # Condition 1: metric with matching axes and dimensions exist
        (
            "set_metrics",
            ("X", "Y"),
            "v",
            "area_e",
            {("X", "Y"): "area_e"},
            "cond2",
        ),  # Condition 2, case a: interpolate metric with matching axis to desired dimensions
        (
            "set_metrics",
            ("X", "Y"),
            "v",
            "area_e",
            {("X", "Y"): "area_e", "X": "dx_n", "Y": "dy_n"},
            "cond2",
        ),  # Condition 2, case b: get_metric should select for the metric with matching axes and interpolate from there,
        # even if other metrics in the desired positions are available
        (
            "set_metrics",
            ("X", "Y"),
            "v",
            None,
            {"X": "dx_n", "Y": "dy_n"},
            "cond3",
        ),  # Condition 3: use provided metrics to calculate for required metric
        (
            "set_metrics",
            ("X", "Y"),
            "v",
            None,
            {"X": "dx_t", "Y": "dy_n"},
            "cond4a",
        ),  # Condition 4, case a: 1 metric on the wrong position (must interpolate before multiplying)
        (
            "set_metrics",
            ("X", "Y"),
            "v",
            None,
            {"X": "dx_t", "Y": "dy_t"},
            "cond4b",
        ),  # Condition 4, case b: 2 metrics in the wrong position (must interpolate before multiplying)
        (
            "set_metrics",
            ["X", "Y", "Z"],
            "tracer",
            None,
            {("X", "Y"): "area_t", "Z": "dz_t"},
            "cond3",
        ),  # Adapt from original version of test_get_metric
    ],
)
def test_get_metric_with_conditions(
    set_metrics, metric_axes, ds_array, ds_expected, set_metric_vars, expected_metric
):

    ds, coords, metrics = datasets_grid_metric("C")
    if set_metrics == "metrics":
        grid = Grid(ds, coords=coords, metrics=metrics)
    else:
        grid = Grid(ds, coords=coords)
        for k, v in set_metric_vars.items():
            grid.set_metrics(k, v)

    get_metric = grid.get_metric(ds[ds_array], metric_axes)

    if expected_metric == "cond1":
        expected_metric = ds[ds_expected].reset_coords(drop=True)
    elif expected_metric == "cond2":
        expected_metric = grid.interp(ds[ds_expected], ("X", "Y"))
    elif expected_metric == "cond3":
        metric_values = [ds[v] for k, v in set_metric_vars.items()]
        expected_metric = (
            functools.reduce(operator.mul, metric_values, 1)
        ).reset_coords(drop=True)
    elif expected_metric == "cond4a":
        metric_values = [ds[v] for k, v in set_metric_vars.items()]
        metric_interp = grid.interp(metric_values[0], "Y")
        expected_metric = (metric_interp * metric_values[1]).reset_coords(drop=True)
    elif expected_metric == "cond4b":
        metric_values = [ds[v] for k, v in set_metric_vars.items()]
        metric_interp_1 = grid.interp(metric_values[0], "Y")
        metric_interp_2 = grid.interp(metric_values[1], "Y")
        expected_metric = (metric_interp_1 * metric_interp_2).reset_coords(drop=True)

    xr.testing.assert_equal(get_metric, expected_metric)
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
            xr.testing.assert_equal(metric_expected.reset_coords(drop=True), metric)

        for metric_expected, metric in zip(v, grid._metrics[k]):
            xr.testing.assert_equal(metric_expected.reset_coords(drop=True), metric)


@pytest.mark.parametrize(
    "metric_axes,metric_name",
    [
        ("X", "dx_t"),
        ("Y", "dy_ne"),
    ],
)
def test_interp_metrics(metric_axes, metric_name):

    ds, coords, _ = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords)
    grid.set_metrics(metric_axes, metric_name)
    interp_metric = grid._interp_metric(ds.u, metric_axes)

    test_metric = grid.interp(ds[metric_name], metric_axes)

    xr.testing.assert_equal(interp_metric, test_metric)
    xr.testing.assert_allclose(interp_metric, test_metric)
