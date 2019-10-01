from __future__ import print_function
import pytest

from xgcm.grid import Grid
from xgcm.test.datasets import datasets_grid_metric


<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Refactored b and c grid testing
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
=======
def datasets():
    """Uniform grid test dataset.
    Should eventually be extended to nonuniform grid"""
    xt = np.arange(4)
    xu = xt + 0.5
    yt = np.arange(5)
    yu = yt + 0.5
    zt = np.arange(6)
    zw = zt + 0.5
    t = np.arange(10)

    def data_generator():
        return np.random.rand(len(xt), len(yt), len(t), len(zt))

    # Need to add a tracer here to get the tracer dimsuffix
    tr = xr.DataArray(
        data_generator(), coords=[("xt", xt), ("yt", yt), ("time", t), ("zt", zt)]
    )

    u_b = xr.DataArray(
        data_generator(), coords=[("xu", xu), ("yu", yu), ("time", t), ("zt", zt)]
    )

    v_b = xr.DataArray(
        data_generator(), coords=[("xu", xu), ("yu", yu), ("time", t), ("zt", zt)]
    )

    u_c = xr.DataArray(
        data_generator(), coords=[("xu", xu), ("yt", yt), ("time", t), ("zt", zt)]
    )

    v_c = xr.DataArray(
        data_generator(), coords=[("xt", xt), ("yu", yu), ("time", t), ("zt", zt)]
    )

    wt = xr.DataArray(
        data_generator(), coords=[("xt", xt), ("yt", yt), ("time", t), ("zw", zw)]
    )

    # maybe also add some other combo of x,t y,t arrays....
    timeseries = xr.DataArray(np.random.rand(len(t)), coords=[("time", t)])

    # northeast distance
    dx = 0.3
    dy = 2
    dz = 20

    dx_ne = xr.DataArray(
        np.ones([len(xt), len(yt)]) * dx - 0.1, coords=[("xu", xu), ("yu", yu)]
    )
    dx_n = xr.DataArray(
        np.ones([len(xt), len(yt)]) * dx - 0.2, coords=[("xt", xt), ("yu", yu)]
    )
    dx_e = xr.DataArray(
        np.ones([len(xt), len(yt)]) * dx - 0.3, coords=[("xu", xu), ("yt", yt)]
    )
    dx_t = xr.DataArray(
        np.ones([len(xt), len(yt)]) * dx - 0.4, coords=[("xt", xt), ("yt", yt)]
    )

    dy_ne = xr.DataArray(
        np.ones([len(xt), len(yt)]) * dy + 0.1, coords=[("xu", xu), ("yu", yu)]
    )
    dy_n = xr.DataArray(
        np.ones([len(xt), len(yt)]) * dy + 0.2, coords=[("xt", xt), ("yu", yu)]
    )
    dy_e = xr.DataArray(
        np.ones([len(xt), len(yt)]) * dy + 0.3, coords=[("xu", xu), ("yt", yt)]
    )
    dy_t = xr.DataArray(
        np.ones([len(xt), len(yt)]) * dy + 0.4, coords=[("xt", xt), ("yt", yt)]
    )

    dz_t = xr.DataArray(
        data_generator() * dz, coords=[("xt", xt), ("yt", yt), ("time", t), ("zt", zt)]
    )
    dz_w = xr.DataArray(
        data_generator() * dz, coords=[("xt", xt), ("yt", yt), ("time", t), ("zw", zw)]
    )

    # Make sure the areas are not just the product of x and y distances
    area_ne = (dx_ne * dy_ne) + 0.1
    area_n = (dx_n * dy_n) + 0.2
    area_e = (dx_e * dy_e) + 0.3
    area_t = (dx_t * dy_t) + 0.4

    # calculate volumes, but again add small differences.
    volume_t = (dx_t * dy_t * dz_t) + 0.25
>>>>>>> applied black locally to facilitate merge

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

<<<<<<< HEAD
    # test u positon
    for axis, metric_name, dim in zip(
        ["X", "Y", ["X", "Y"]],
        ["dx_e", "dy_e", "area_e"],  # need more metrics?
        ["xu", "yt", ["xu", "yt"]],
    ):
        integrated = grid.integrate(ds.u, axis)
        expected = (ds.u * ds[metric_name]).sum(dim)
        assert integrated.equals(expected)
=======
    # combine to different grid configurations (B and C grid)
    ds_b = _add_metrics(
        xr.Dataset(
            {"u": u_b, "v": v_b, "wt": wt, "tracer": tr, "timeseries": timeseries}
        )
    )
    ds_c = _add_metrics(
        xr.Dataset(
            {"u": u_c, "v": v_c, "wt": wt, "tracer": tr, "timeseries": timeseries}
        )
    )
>>>>>>> applied black locally to facilitate merge

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

<<<<<<< HEAD
    if axis == "Y":
        # test two missing axes at the same time
        del coords["X"]
        del_metrics = [k for k in metrics.keys() if "X" in k]
        for dm in del_metrics:
            del metrics[dm]
        grid = Grid(ds, coords=coords, metrics=metrics)
        match_message = "Axis X,Y not found"
=======
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
        ),  # should we be able to pass a tuple aswell as a list?
        (["X", "Y", "Z"], "tracer", None, ["volume_t"], None),
        (["X"], "u", None, ["dx_e"], None),
        (["X", "Y"], "u", None, ["area_e"], None),
        (("X", "Y", "Z"), "u", None, ["volume_t"], KeyError),  # This should error out
        # reconstructed cases
        (["X", "Y"], "tracer", ["area_t"], ["dx_t", "dy_t"], None),
        (["X", "Y", "Z"], "tracer", ["volume_t"], ["area_t", "dz_t"], None),
    ],
)
def test_get_metric(axes, data_var, drop_vars, metric_expected_list, expected_error):
    ds_full = datasets()
    ds = ds_full["C"]
    metrics = ds_full["metrics"]
    # drop metrics according to drop_vars input, and remove from metrics input
    if drop_vars:
        print(drop_vars)
        ds = ds.drop(drop_vars)
        metrics = {k: [a for a in v if a not in drop_vars] for k, v in metrics.items()}
>>>>>>> applied black locally to facilitate merge

        with pytest.raises(ValueError, match=match_message):
            grid.integrate(ds, ["X", "Y", "Z"])


def test_integrate_missingdim():
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, metrics=metrics)

    with pytest.raises(ValueError, match="matching dimension corresponding to axis X"):
        grid.integrate(ds.tracer.mean("xt"), "X")

    with pytest.raises(ValueError, match="matching dimension corresponding to axis X"):
        grid.integrate(ds.tracer.mean("xt"), ["X", "Y", "Z"])
