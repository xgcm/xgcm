from __future__ import print_function
import xarray as xr
import numpy as np
import pytest

from xgcm.grid import Grid, Axis


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
        data_generator(),
        coords=[("xt", xt), ("yt", yt), ("time", t), ("zt", zt)],
    )

    u_b = xr.DataArray(
        data_generator(),
        coords=[("xu", xu), ("yu", yu), ("time", t), ("zt", zt)],
    )

    v_b = xr.DataArray(
        data_generator(),
        coords=[("xu", xu), ("yu", yu), ("time", t), ("zt", zt)],
    )

    u_c = xr.DataArray(
        data_generator(),
        coords=[("xu", xu), ("yt", yt), ("time", t), ("zt", zt)],
    )

    v_c = xr.DataArray(
        data_generator(),
        coords=[("xt", xt), ("yu", yu), ("time", t), ("zt", zt)],
    )

    wt = xr.DataArray(
        data_generator(),
        coords=[("xt", xt), ("yt", yt), ("time", t), ("zw", zw)],
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
        data_generator() * dz,
        coords=[("xt", xt), ("yt", yt), ("time", t), ("zt", zt)],
    )
    dz_w = xr.DataArray(
        data_generator() * dz,
        coords=[("xt", xt), ("yt", yt), ("time", t), ("zw", zw)],
    )

    # Make sure the areas are not just the product of x and y distances
    area_ne = (dx_ne * dy_ne) + 0.1
    area_n = (dx_n * dy_n) + 0.2
    area_e = (dx_e * dy_e) + 0.3
    area_t = (dx_t * dy_t) + 0.4

    # calculate volumes, but again add small differences.
    volume_t = (dx_t * dy_t * dz_t) + 0.25

    def _add_metrics(obj):
        obj = obj.copy()
        for name, data in zip(
            [
                "dx_ne",
                "dx_n",
                "dx_e",
                "dx_t",
                "dy_ne",
                "dy_n",
                "dy_e",
                "dy_t",
                "dz_t",
                "dz_w",
                "area_ne",
                "area_n",
                "area_e",
                "area_t",
                "volume_t",
            ],
            [
                dx_ne,
                dx_n,
                dx_e,
                dx_t,
                dy_ne,
                dy_n,
                dy_e,
                dy_t,
                dz_t,
                dz_w,
                area_ne,
                area_n,
                area_e,
                area_t,
                volume_t,
            ],
        ):
            obj.coords[name] = data
        # add xgcm attrs
        for ii in ["xu", "xt"]:
            obj[ii].attrs["axis"] = "X"
        for ii in ["yu", "yt"]:
            obj[ii].attrs["axis"] = "Y"
        for ii in ["zt", "zw"]:
            obj[ii].attrs["axis"] = "Z"
        for ii in ["time"]:
            obj[ii].attrs["axis"] = "T"
        for ii in ["xu", "yu", "zw"]:
            obj[ii].attrs["c_grid_axis_shift"] = 0.5
        return obj

    coords = {
        "X": {"center": "xt", "right": "xu"},
        "Y": {"center": "yt", "right": "yu"},
        "Z": {"center": "zt", "right": "zw"},
    }

    metrics = {
        ("X",): ["dx_t", "dx_n", "dx_e", "dx_ne"],
        ("Y",): ["dy_t", "dy_n", "dy_e", "dy_ne"],
        ("Z",): ["dz_t", "dz_w"],
        ("X", "Y"): ["area_t", "area_n", "area_e", "area_ne"],
        ("X", "Y", "Z"): ["volume_t"],
    }
    # Used previously for some fail test. Retain for now.
    # coords_outer = {
    #     "X": {"center": "xt", "outer": "xu"},
    #     "Y": {"center": "yt", "outer": "yu"},
    #     "Z": {"center": "zt", "outer": "zw"},
    # }

    # combine to different grid configurations (B and C grid)
    ds_b = _add_metrics(
        xr.Dataset(
            {
                "u": u_b,
                "v": v_b,
                "wt": wt,
                "tracer": tr,
                "timeseries": timeseries,
            }
        )
    )
    ds_c = _add_metrics(
        xr.Dataset(
            {
                "u": u_c,
                "v": v_c,
                "wt": wt,
                "tracer": tr,
                "timeseries": timeseries,
            }
        )
    )

    # Some fail tests used in the weighted mean (retain for later)
    # ds_fail = _add_metrics(
    #     xr.Dataset(
    #         {"u": u_b, "v": v_c, "tracer": tr, "timeseries": timeseries}
    #     )
    # )
    # ds_fail2 = _add_metrics(
    #     xr.Dataset(
    #         {"u": u_b, "v": v_c, "tracer": tr, "timeseries": timeseries}
    #     )
    # )

    return {
        "B": ds_b,
        "C": ds_c,
        # "fail_gridtype": ds_fail,
        # "fail_dimtype": ds_fail2,
        "coords": coords,
        "metrics": metrics,
        # "fail_coords": coords_outer,
    }


###################
# Test metric internals (this should be moved I suppose)
##################
@pytest.mark.parametrize(
    "key, metric_vars",
    [
        # (("X"), ["dx_t"]), # this fails, can we raise a more useful error?
        (("X",), ["dx_t"]),
        (("X",), "dx_t"),
        (("X", "Y"), ["area_t"]),
        (
            ("X", "Y"),
            ["area_t", "area_t"],
        ),  # this should also trigger an error, it does not
        (("X", "Y"), ["area_t", "area_e", "area_n", "area_ne"]),
        (("X", "Y", "Z"), ["volume_t"]),
    ],
)
def test_assign_get_metric(key, metric_vars):
    ds_full = datasets()
    ds = ds_full["C"]
    grid = Grid(
        ds,
        coords=ds_full["coords"],
        # periodic=True,
    )
    grid._assign_metrics({key: metric_vars})

    # convert key
    key = frozenset(key)

    # if input is str convert to list
    if isinstance(metric_vars, str):
        metric_vars = [metric_vars]
    expected = [ds[mv].reset_coords(drop=True) for mv in metric_vars]

    assert all([a.equals(b) for a, b in zip(grid._metrics[key], expected)])


@pytest.mark.parametrize(
    "axes, data_var, metric_expected",
    [
        ("X", "tracer", "dx_t"),
        (["X", "Y"], "tracer", "area_t"),
        (["X", "Y", "Z"], "tracer", "volume_t"),
        (["X"], "u", "dx_e"),
        (["X", "Y"], "u", "area_e"),
        # (("X", "Y", "Z"), "u", "volume_t"), # This should error out
        # should we be able to pass a tuple aswell as a list?
    ],
)
def test_get_metric(axes, data_var, metric_expected):
    ds_full = datasets()
    ds = ds_full["C"]
    grid = Grid(
        ds,
        coords=ds_full["coords"],
        metrics=ds_full["metrics"],
        # periodic=True,
    )
    metric = grid.get_metric(ds[data_var], axes)
    expected = ds[metric_expected].reset_coords(drop=True)
    assert metric.equals(expected)
    # TODO integrate a more sleek test with B grid input


@pytest.mark.parametrize("gridtype", ["B", "C"])
def test_integrate(gridtype):
    ds_full = datasets()
    ds = ds_full[gridtype]
    grid = Grid(
        ds,
        coords=ds_full["coords"],
        metrics=ds_full["metrics"],
        # periodic=True,
    )
    for axis, metric_name, dim in zip(
        ["X", "Y", "Z"], ["dx_t", "dy_t", "dz_t"], ["xt", "yt", "zt"]
    ):
        print(ds.tracer)
        metric_test = grid.get_metric(ds.tracer, (axis,))
        print(metric_test)
        # expected = (ds.tracer * ds[metric_name]).sum(dim)
        # print(expected)
        # test = grid.integrate(ds.tracer, [axis])
        # print(test)
        # assert grid.integrate(ds.tracer, axis).equals(expected)
        assert 1 == 0
