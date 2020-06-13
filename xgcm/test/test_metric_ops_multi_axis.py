from __future__ import print_function
import pytest
import xarray as xr
import numpy as np

from xgcm.grid import Grid
from xgcm.test.datasets import datasets_grid_metric


def _expected_result(da, metric, grid, dim, axes, funcname, boundary=None):
    """this is factoring out the expected output of metric aware operations"""

    # make sure nans in the data are reflected in the metric
    nanmask = np.isnan(da)
    metric = metric.where(~nanmask)

    if funcname == "integrate":
        expected = (da * metric).sum(dim)
    elif funcname == "average":
        expected = (da * metric).sum(dim) / metric.sum(dim)
    elif funcname == "cumint":
        expected = grid.cumsum(da * metric, axes, boundary=boundary)
    else:
        raise ValueError("funcname (`%s`) not recognized" % funcname)
    return expected


def _parse_metrics_for_grid(grid_config, grid_position):
    # parses the appropriate axes, metrics and dims for the different grid configs and grid positions
    if grid_config == "B":
        if grid_position == "tracer":
            return zip(
                ["X", "Y", "Z", ["X", "Y"], ["X", "Y", "Z"]],
                ["dx_t", "dy_t", "dz_t", "area_t", "volume_t"],
                ["xt", "yt", "zt", ["xt", "yt"], ["xt", "yt", "zt"]],
            )
        elif grid_position == "u":
            return zip(
                ["X", "Y", ["X", "Y"]],
                ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
                ["xu", "yu", ["xu", "yu"]],
            )
        elif grid_position == "v":
            return zip(
                ["X", "Y", ["X", "Y"]],
                ["dx_ne", "dy_ne", "area_ne"],  # need more metrics?
                ["xu", "yu", ["xu", "yu"]],
            )
    elif grid_config == "C":
        if grid_position == "tracer":
            return zip(
                ["X", "Y", "Z", ["X", "Y"], ["X", "Y", "Z"]],
                ["dx_t", "dy_t", "dz_t", "area_t", "volume_t"],
                ["xt", "yt", "zt", ["xt", "yt"], ["xt", "yt", "zt"]],
            )
        elif grid_position == "u":
            return zip(
                ["X", "Y", ["X", "Y"]],
                ["dx_e", "dy_e", "area_e"],  # need more metrics?
                ["xu", "yt", ["xu", "yt"]],
            )
        elif grid_position == "v":
            return zip(
                ["X", "Y", ["X", "Y"]],
                ["dx_n", "dy_n", "area_n"],  # need more metrics?
                ["xt", "yu", ["xt", "yu"]],
            )


@pytest.mark.parametrize("funcname", ["integrate", "average", "cumint"])
@pytest.mark.parametrize(
    "boundary", ["fill", "extend"]
)  # we do not support extrapolate for cumsum?
@pytest.mark.parametrize(
    "periodic",
    [None, "True", "False", {"X": True, "Y": False}, {"X": False, "Y": True}],
)
class TestParametrized:
    @pytest.mark.parametrize("grid_config", ["B", "C"])
    @pytest.mark.parametrize("grid_position", ["tracer", "u", "v"])
    @pytest.mark.parametrize("missing_values", [False, True])
    def test_grids(
        self, funcname, boundary, periodic, missing_values, grid_config, grid_position
    ):
        ds, coords, metrics = datasets_grid_metric(grid_config)
        grid = Grid(ds, coords=coords, metrics=metrics, periodic=periodic)

        if funcname == "cumint":
            # cumint needs a boundary...
            kwargs = dict(boundary=boundary)
        else:
            # integrate and average do use the boundary input
            kwargs = dict()

        func = getattr(grid, funcname)

        da = ds[grid_position]
        if missing_values:
            mask = np.random.choice([True, False], size=ds.tracer.data.shape)
            da = da.where(mask)

        for axis, metric_name, dim in _parse_metrics_for_grid(
            grid_config, grid_position
        ):
            new = func(da, axis, **kwargs)
            expected = _expected_result(
                da, ds[metric_name], grid, dim, axis, funcname, **kwargs
            )

            xr.testing.assert_allclose(new, expected)

            # test with tuple input if list is provided
            if isinstance(axis, list):
                new = func(da, tuple(axis), **kwargs)
                xr.testing.assert_allclose(new, expected)

    @pytest.mark.parametrize("axis", ["X", "Y", "Z"])
    def test_missingaxis(self, axis, funcname, periodic, boundary):
        # Error should be raised if application axes
        # include dimension not in datasets

        ds, coords, metrics = datasets_grid_metric("C")

        del coords[axis]

        del_metrics = [k for k in metrics.keys() if axis in k]
        for dm in del_metrics:
            del metrics[dm]

        grid = Grid(ds, coords=coords, metrics=metrics, periodic=periodic)

        func = getattr(grid, funcname)

        if funcname == "cumint":
            # cumint needs a boundary...
            kwargs = dict(boundary=boundary)
        else:
            kwargs = dict()

        match_message = (
            "Unable to find any combinations of metrics for array dims.*%s.*" % axis
        )

        with pytest.raises(KeyError, match=match_message):
            func(ds.tracer, ["X", "Y", "Z"], **kwargs)

        with pytest.raises(KeyError, match=match_message):
            func(ds, axis, **kwargs)

        if axis == "Y":
            # test two missing axes at the same time
            del coords["X"]
            del_metrics = [k for k in metrics.keys() if "X" in k]
            for dm in del_metrics:
                del metrics[dm]

            grid = Grid(ds, coords=coords, metrics=metrics)

            func = getattr(grid, funcname)

            if funcname == "cumint":
                # cumint needs a boundary...
                kwargs = dict(boundary="fill")
            else:
                kwargs = dict()

            match_message = (
                "Unable to find any combinations of metrics for array dims.*X.*Y.*Z.*"
            )
            with pytest.raises(KeyError, match=match_message):
                func(ds, ["X", "Y", "Z"], **kwargs)

            match_message = (
                "Unable to find any combinations of metrics for array dims.*X.*Y.*"
            )
            with pytest.raises(KeyError, match=match_message):
                func(ds, ("X", "Y"), **kwargs)

    def test_missingdim(self, funcname, periodic, boundary):
        ds, coords, metrics = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords, metrics=metrics, periodic=periodic)

        func = getattr(grid, funcname)

        if funcname == "cumint":
            # cumint needs a boundary...
            kwargs = dict(boundary=boundary)
        else:
            kwargs = dict()

        match_message = "Unable to find any combinations of metrics for array dims.*X.*"
        with pytest.raises(KeyError, match=match_message):
            func(ds.tracer.mean("xt"), "X", **kwargs)

        match_message = (
            "Unable to find any combinations of metrics for array dims.*X.*Y.*Z.*"
        )
        with pytest.raises(KeyError, match=match_message):
            func(ds.tracer.mean("xt"), ["X", "Y", "Z"], **kwargs)
