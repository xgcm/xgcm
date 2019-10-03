from __future__ import print_function
import pytest

from xgcm.grid import Grid
from xgcm.test.datasets import datasets_grid_metric


def test_integrate_bgrid():
    ds, coords, metrics = datasets_grid_metric("B")
    grid = Grid(ds, coords=coords, metrics=metrics)
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
    # test u position
    for axis, metric_name, dim in zip(
        ["X", "Y", ["X", "Y"]],
        ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
        ["xu", "yu", ["xu", "yu"]],
    ):
        integrated = grid.integrate(ds.u, axis)
        expected = (ds.u * ds[metric_name]).sum(dim)
        assert integrated.equals(expected)

    # test v position
    for axis, metric_name, dim in zip(
        ["X", "Y", ["X", "Y"]],
        ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
        ["xu", "yu", ["xu", "yu"]],
    ):
        integrated = grid.integrate(ds.v, axis)
        expected = (ds.v * ds[metric_name]).sum(dim)
        assert integrated.equals(expected)


def test_integrate_cgrid():
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, metrics=metrics)
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
    for axis, metric_name, dim in zip(
        ["X", "Y", ["X", "Y"]],
        ["dx_e", "dy_e", "area_e"],  # need more metrics?
        ["xu", "yt", ["xu", "yt"]],
    ):
        integrated = grid.integrate(ds.u, axis)
        expected = (ds.u * ds[metric_name]).sum(dim)
        assert integrated.equals(expected)

    # test v positon
    for axis, metric_name, dim in zip(
        ["X", "Y", ["X", "Y"]],
        ["dx_n", "dy_n", "area_n"],  # need more metrics?
        ["xt", "yu", ["xt", "yu"]],
    ):
        integrated = grid.integrate(ds.v, axis)
        expected = (ds.v * ds[metric_name]).sum(dim)
        assert integrated.equals(expected)


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_integrate_missingaxis(axis):
    # Error should be raised if integration axes include dimension not in datasets
    ds, coords, metrics = datasets_grid_metric("C")

    del coords[axis]

    del_metrics = [k for k in metrics.keys() if axis in k]
    for dm in del_metrics:
        del metrics[dm]

    grid = Grid(ds, coords=coords, metrics=metrics)
    match_message = "Axis " + axis + " not found"

    with pytest.raises(ValueError, match=match_message):
        grid.integrate(ds.tracer, ["X", "Y", "Z"])

    with pytest.raises(ValueError, match=match_message):
        grid.integrate(ds, axis)

    if axis == "Y":
        # test two missing axes at the same time
        del coords["X"]
        del_metrics = [k for k in metrics.keys() if "X" in k]
        for dm in del_metrics:
            del metrics[dm]
        grid = Grid(ds, coords=coords, metrics=metrics)
        match_message = "Axis X,Y not found"
        with pytest.raises(ValueError, match=match_message):
            grid.integrate(ds, ["X", "Y", "Z"])


def test_integrate_missingdim():
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, metrics=metrics)

    with pytest.raises(ValueError, match="matching dimension corresponding to axis X"):
        grid.integrate(ds.tracer.mean("xt"), "X")

    with pytest.raises(ValueError, match="matching dimension corresponding to axis X"):
        grid.integrate(ds.tracer.mean("xt"), ["X", "Y", "Z"])
