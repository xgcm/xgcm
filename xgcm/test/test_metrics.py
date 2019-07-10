from __future__ import print_function
import xarray as xr
import numpy as np

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
