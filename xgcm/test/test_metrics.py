from __future__ import print_function
import xarray as xr
import numpy as np
import functools
import operator
import pytest

from xgcm.grid import Grid, Axis

def test_multiple_metrics_per_axis():
    # copied from test_derivatives.py - should refactor
    dx = 10.0
    ds = xr.Dataset(
        {"foo": (("XC",), [1.0, 2.0, 4.0, 3.0]),
         "bar": (("XG",), [10.0, 20.0, 30.0, 40.0])},
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


# Julius's functions
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
            obj.coords[name].attrs["tracked_name"] = name
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
            {"u": u_b, "v": v_b, "wt": wt, "tracer": tr, "timeseries": timeseries}
        )
    )
    ds_c = _add_metrics(
        xr.Dataset(
            {"u": u_c, "v": v_c, "wt": wt, "tracer": tr, "timeseries": timeseries}
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


@pytest.mark.parametrize(
    "key, metric_vars",
    [
        (("X",), ["dx_t"]), # recommended way
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
    ds_full = datasets()
    ds = ds_full["C"]
    grid = Grid(
        ds,
        coords=ds_full["coords"],
        metrics={key: metric_vars},
    )


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

    grid = Grid(
        ds,
        coords=ds_full["coords"],
        metrics=metrics,
    )
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
