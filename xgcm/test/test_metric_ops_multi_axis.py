from __future__ import print_function
import pytest

from xgcm.grid import Grid
from xgcm.test.datasets import datasets_grid_metric


def _expected_result(da, metric, dim, funcname):
    if funcname == "integrate":
        expected = (da * metric).sum(dim)
    elif funcname == "average":
        expected = (da * metric).sum(dim) / metric.sum(dim)
    else:
        raise ValueError("funcname (`%s`) not recognized" % funcname)
    return expected


@pytest.mark.parametrize("funcname", ["integrate", "average"])
# @pytest.mark.parametrize("funcname", ["integrate"])
def test_bgrid(funcname):
    ds, coords, metrics = datasets_grid_metric("B")
    grid = Grid(ds, coords=coords, metrics=metrics)

    func = getattr(grid, funcname)

    # test tracer position
    for axis, metric_name, dim in zip(
        ["X", "Y", "Z", ["X", "Y"], ["X", "Y", "Z"]],
        ["dx_t", "dy_t", "dz_t", "area_t", "volume_t"],
        ["xt", "yt", "zt", ["xt", "yt"], ["xt", "yt", "zt"]],
    ):
        new = func(ds.tracer, axis)
        expected = _expected_result(ds.tracer, ds[metric_name], dim, funcname)
        assert new.equals(expected)

        # test with tuple input if list is provided
        if isinstance(axis, list):
            new = func(ds.tracer, tuple(axis))
            assert new.equals(expected)

        # test u position
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
            ["xu", "yu", ["xu", "yu"]],
        ):
            new = func(ds.u, axis)
            expected = _expected_result(ds.u, ds[metric_name], dim, funcname)
            assert new.equals(expected)

        # test v position
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
            ["xu", "yu", ["xu", "yu"]],
        ):
            new = func(ds.v, axis)
            expected = _expected_result(ds.v, ds[metric_name], dim, funcname)
            assert new.equals(expected)


@pytest.mark.parametrize("funcname", ["integrate", "average"])
def test_cgrid(funcname):
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, metrics=metrics)

    func = getattr(grid, funcname)

    # test tracer position
    for axis, metric_name, dim in zip(
        ["X", "Y", "Z", ["X", "Y"], ["X", "Y", "Z"]],
        ["dx_t", "dy_t", "dz_t", "area_t", "volume_t"],
        ["xt", "yt", "zt", ["xt", "yt"], ["xt", "yt", "zt"]],
    ):
        new = func(ds.tracer, axis)
        expected = _expected_result(ds.tracer, ds[metric_name], dim, funcname)
        assert new.equals(expected)
        # test with tuple input if list is provided
        if isinstance(axis, list):
            new = func(ds.tracer, tuple(axis))
            assert new.equals(expected)

    # test u positon
    for axis, metric_name, dim in zip(
        ["X", "Y", ["X", "Y"]],
        ["dx_e", "dy_e", "area_e"],  # need more metrics?
        ["xu", "yt", ["xu", "yt"]],
    ):
        new = func(ds.u, axis)
        expected = _expected_result(ds.u, ds[metric_name], dim, funcname)
        assert new.equals(expected)

    # test v positon
    for axis, metric_name, dim in zip(
        ["X", "Y", ["X", "Y"]],
        ["dx_n", "dy_n", "area_n"],  # need more metrics?
        ["xt", "yu", ["xt", "yu"]],
    ):
        new = func(ds.v, axis)
        expected = _expected_result(ds.v, ds[metric_name], dim, funcname)
        assert new.equals(expected)


@pytest.mark.parametrize(
    "funcname", ["integrate", "average"]
)  # there is probably a way to define this as a fixture for the module?
@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_missingaxis(axis, funcname):
    # Error should be raised if application axes
    # include dimension not in datasets

    ds, coords, metrics = datasets_grid_metric("C")

    del coords[axis]

    del_metrics = [k for k in metrics.keys() if axis in k]
    for dm in del_metrics:
        del metrics[dm]

    grid = Grid(ds, coords=coords, metrics=metrics)

    func = getattr(grid, funcname)

    match_message = (
        "Unable to find any combinations of metrics for array dims.*%s.*" % axis
    )

    with pytest.raises(KeyError, match=match_message):
        func(ds.tracer, ["X", "Y", "Z"])

    with pytest.raises(KeyError, match=match_message):
        func(ds, axis)

    if axis == "Y":
        # test two missing axes at the same time
        del coords["X"]
        del_metrics = [k for k in metrics.keys() if "X" in k]
        for dm in del_metrics:
            del metrics[dm]

        grid = Grid(ds, coords=coords, metrics=metrics)

        func = getattr(grid, funcname)

        match_message = (
            "Unable to find any combinations of metrics for array dims.*X.*Y.*Z.*"
        )
        with pytest.raises(KeyError, match=match_message):
            func(ds, ["X", "Y", "Z"])

        match_message = (
            "Unable to find any combinations of metrics for array dims.*X.*Y.*"
        )
        with pytest.raises(KeyError, match=match_message):
            func(ds, ("X", "Y"))


@pytest.mark.parametrize("funcname", ["integrate", "average"])
def test_missingdim(funcname):
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, metrics=metrics)

    func = getattr(grid, funcname)

    match_message = "Unable to find any combinations of metrics for array dims.*X.*"
    with pytest.raises(KeyError, match=match_message):
        func(ds.tracer.mean("xt"), "X")

    match_message = (
        "Unable to find any combinations of metrics for array dims.*X.*Y.*Z.*"
    )
    with pytest.raises(KeyError, match=match_message):
        func(ds.tracer.mean("xt"), ["X", "Y", "Z"])
