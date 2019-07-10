from __future__ import print_function
import pytest

from xgcm.grid import Grid
from xgcm.test.datasets import datasets_grid_metric


@pytest.mark.parametrize("gridtype", ["B", "C"])
def test_integrate(gridtype):
    ds_full = datasets_grid_metric()
    ds = ds_full[gridtype]
    grid = Grid(
        ds,
        coords=ds_full["coords"],
        metrics=ds_full["metrics"],
        # periodic=True,
    )
    # test tracer position
    for axis, metric_name, dim in zip(
        ["X", "Y", "Z", ["X", "Y"], ["X", "Y", "Z"]],
        ["dx_t", "dy_t", "dz_t", "area_t", "volume_t"],
        ["xt", "yt", "zt", ["xt", "yt"], ["xt", "yt", "zt"]],
    ):
        integrated = grid.integrate(ds.tracer, axis)
        expected = (ds.tracer * ds[metric_name]).sum(dim)
        assert integrated.equals(expected)
        # test with tuple input if list is provided
        if isinstance(axis, list):
            integrated = grid.integrate(ds.tracer, tuple(axis))
            expected = (ds.tracer * ds[metric_name]).sum(dim)
            assert integrated.equals(expected)

    # test u positon
    if gridtype == "B":
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
            ["xu", "yu", ["xu", "yu"]],
        ):
            integrated = grid.integrate(ds.u, axis)
            expected = (ds.u * ds[metric_name]).sum(dim)
            assert integrated.equals(expected)
    elif gridtype == "C":
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_e", "dy_e", "area_e"],  # need more metrics?
            ["xu", "yt", ["xu", "yt"]],
        ):
            integrated = grid.integrate(ds.u, axis)
            expected = (ds.u * ds[metric_name]).sum(dim)
            assert integrated.equals(expected)

    # test v positon
    if gridtype == "B":
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
            ["xu", "yu", ["xu", "yu"]],
        ):
            integrated = grid.integrate(ds.v, axis)
            expected = (ds.v * ds[metric_name]).sum(dim)
            assert integrated.equals(expected)
    elif gridtype == "C":
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_n", "dy_n", "area_n"],  # need more metrics?
            ["xt", "yu", ["xt", "yu"]],
        ):
            integrated = grid.integrate(ds.v, axis)
            expected = (ds.v * ds[metric_name]).sum(dim)
            assert integrated.equals(expected)
