import pytest
from xarray.testing import assert_allclose

from xgcm.grid import Grid
from xgcm.test.datasets import datasets_grid_metric


def _expected_result(da, metric, grid, dim, axes, funcname, boundary=None):
    """this is factoring out the expected output of metric aware operations"""
    if funcname == "integrate":
        expected = (da * metric).sum(dim)
    elif funcname == "average":
        expected = (da * metric).sum(dim) / metric.sum(dim)
    elif funcname == "cumint":
        expected = grid.cumsum(da * metric, axes, boundary=boundary)
    else:
        raise ValueError(f"funcname {funcname} not recognized")
    return expected


@pytest.mark.parametrize("funcname", ["integrate", "average", "cumint"])
@pytest.mark.parametrize(
    "boundary", ["fill", "extend"]
)  # we do not support extrapolate for cumsum?
@pytest.mark.parametrize(
    "periodic",
    [None, "True", "False", {"X": True, "Y": False}, {"X": False, "Y": True}],
)
class TestParametrized:
    def test_bgrid(self, funcname, boundary, periodic):
        ds, coords, metrics = datasets_grid_metric("B")
        grid = Grid(ds, coords=coords, metrics=metrics, periodic=periodic)

        if funcname == "cumint":
            # cumint needs a boundary
            kwargs = dict(boundary=boundary)
        else:
            # integrate and average don't use the boundary input
            kwargs = dict()

        func = getattr(grid, funcname)

        # test tracer position
        for axis, metric_name, dim in zip(
            ["X", "Y", "Z", ["X", "Y"], ["X", "Y", "Z"]],
            ["dx_t", "dy_t", "dz_t", "area_t", "volume_t"],
            ["xt", "yt", "zt", ["xt", "yt"], ["xt", "yt", "zt"]],
        ):
            new = func(ds.tracer, axis, **kwargs)
            expected = _expected_result(
                ds.tracer, ds[metric_name], grid, dim, axis, funcname, **kwargs
            )
            assert_allclose(new, expected)

            # test with tuple input if list is provided
            if isinstance(axis, list):
                new = func(ds.tracer, tuple(axis), **kwargs)
                assert_allclose(new, expected)

        # test u position
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
            ["xu", "yu", ["xu", "yu"]],
        ):
            new = func(ds.u, axis, **kwargs)
            expected = _expected_result(
                ds.u, ds[metric_name], grid, dim, axis, funcname, **kwargs
            )
            assert_allclose(new, expected)

        # test v position
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
            ["xu", "yu", ["xu", "yu"]],
        ):
            new = func(ds.v, axis, **kwargs)
            expected = _expected_result(
                ds.v, ds[metric_name], grid, dim, axis, funcname, **kwargs
            )
            assert_allclose(new, expected)

    def test_cgrid(self, funcname, boundary, periodic):
        ds, coords, metrics = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords, metrics=metrics, periodic=periodic)

        if funcname == "cumint":
            # cumint needs a boundary
            kwargs = dict(boundary=boundary)
        else:
            # integrate and average don't use the boundary input
            kwargs = dict()

        func = getattr(grid, funcname)

        # test tracer position
        for axis, metric_name, dim in zip(
            ["X", "Y", "Z", ["X", "Y"], ["X", "Y", "Z"]],
            ["dx_t", "dy_t", "dz_t", "area_t", "volume_t"],
            ["xt", "yt", "zt", ["xt", "yt"], ["xt", "yt", "zt"]],
        ):

            new = func(ds.tracer, axis, **kwargs)
            expected = _expected_result(
                ds.tracer, ds[metric_name], grid, dim, axis, funcname, **kwargs
            )
            assert_allclose(new, expected)
            # test with tuple input if list is provided
            if isinstance(axis, list):
                new = func(ds.tracer, tuple(axis), **kwargs)
                assert_allclose(new, expected)

        # test u positon
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_e", "dy_e", "area_e"],  # need more metrics?
            ["xu", "yt", ["xu", "yt"]],
        ):
            new = func(ds.u, axis, **kwargs)
            expected = _expected_result(
                ds.u, ds[metric_name], grid, dim, axis, funcname, **kwargs
            )
            assert_allclose(new, expected)

        # test v positon
        for axis, metric_name, dim in zip(
            ["X", "Y", ["X", "Y"]],
            ["dx_n", "dy_n", "area_n"],  # need more metrics?
            ["xt", "yu", ["xt", "yu"]],
        ):
            new = func(ds.v, axis, **kwargs)
            expected = _expected_result(
                ds.v, ds[metric_name], grid, dim, axis, funcname, **kwargs
            )
            assert_allclose(new, expected)

    @pytest.mark.parametrize("axis", ["X", "Y", "Z"])
    def test_missingaxis(self, axis, funcname, periodic, boundary):
        # Error should be raised if application axes include dimension not in datasets

        ds, coords, metrics = datasets_grid_metric("C")

        del coords[axis]

        del_metrics = [k for k in metrics.keys() if axis in k]
        for dm in del_metrics:
            del metrics[dm]

        grid = Grid(ds, coords=coords, metrics=metrics, periodic=periodic)

        func = getattr(grid, funcname)

        if funcname == "cumint":
            # cumint needs a boundary
            kwargs = dict(boundary=boundary)
        else:
            kwargs = dict()

        with pytest.raises(KeyError, match="Did not find axis"):
            func(ds.tracer, ["X", "Y", "Z"], **kwargs)

        if axis == "Y":
            # test two missing axes at the same time
            del coords["X"]
            del_metrics = [k for k in metrics.keys() if "X" in k]
            for dm in del_metrics:
                del metrics[dm]

            grid = Grid(ds, coords=coords, metrics=metrics)

            func = getattr(grid, funcname)

            if funcname == "cumint":
                # cumint needs a boundary
                kwargs = dict(boundary="fill")
            else:
                kwargs = dict()

            with pytest.raises(KeyError, match="Did not find axis"):
                func(ds.tracer, ["X", "Y", "Z"], **kwargs)

            with pytest.raises(KeyError, match="Did not find axis"):
                func(ds.tracer, ("X", "Y"), **kwargs)

    def test_metric_axes_missing_from_array(self, funcname, periodic, boundary):
        ds, coords, metrics = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords, metrics=metrics, periodic=periodic)

        if funcname == "cumint":
            # cumint needs a boundary
            kwargs = dict(boundary="fill")
        else:
            kwargs = dict()

        func = getattr(grid, funcname)

        with pytest.raises(ValueError, match="Did not find single matching dimension"):
            func(ds.tracer.mean("xt"), "X", **kwargs)

        with pytest.raises(ValueError, match="Did not find single matching dimension"):
            func(ds.tracer.mean("xt"), ["X", "Y", "Z"], **kwargs)
