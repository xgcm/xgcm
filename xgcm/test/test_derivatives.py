from __future__ import print_function
import xarray as xr

import numpy as np

from xgcm.grid import Grid, Axis
from xgcm.test.datasets import datasets_grid_metric


def test_derivative_uniform_grid():
    # this is a uniform grid
    # a non-uniform grid would provide a more rigorous test
    dx = 10.0
    dy = 10.0
    arr = [
        [1.0, 2.0, 4.0, 3.0],
        [4.0, 7.0, 1.0, 2.0],
        [3.0, 1.0, 0.0, 9.0],
        [8.0, 5.0, 2.0, 1.0],
    ]
    ds = xr.Dataset(
        {"foo": (("XC", "YC"), arr)},
        coords={
            "XC": (("XC",), [0.5, 1.5, 2.5, 3.5]),
            "XG": (("XG",), [0, 1.0, 2.0, 3.0]),
            "dXC": (("XC",), [dx, dx, dx, dx]),
            "dXG": (("XG",), [dx, dx, dx, dx]),
            "YC": (("YC",), [0.5, 1.5, 2.5, 3.5]),
            "YG": (("YG",), [0, 1.0, 2.0, 3.0]),
            "dYC": (("YC",), [dy, dy, dy, dy]),
            "dYG": (("YG",), [dy, dy, dy, dy]),
        },
    )

    grid = Grid(
        ds,
        coords={
            "X": {"center": "XC", "left": "XG"},
            "Y": {"center": "YC", "left": "YG"},
        },
        metrics={("X",): ["dXC", "dXG"], ("Y",): ["dYC", "dYG"]},
        periodic=True,
    )

    # Test x direction
    dfoo_dx = grid.derivative(ds.foo, "X")
    expected = grid.diff(ds.foo, "X") / dx
    assert dfoo_dx.equals(expected)

    # Test x direction
    dfoo_dy = grid.derivative(ds.foo, "Y")
    expected = grid.diff(ds.foo, "Y") / dy
    assert dfoo_dy.equals(expected)


def test_derivative_c_grid():
    # test derivatives with synthetic C grid data

    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, metrics=metrics)

    # run this for each axis and each field in dataset
    def test_single_derivative(axis, fld, dx):

        dvar_dx = grid.derivative(fld, axis)
        expected = grid.diff(fld, axis) / dx
        abs_diff = np.sum(np.abs(dvar_dx - expected))

        assert dvar_dx.equals(expected.reset_coords(drop=True))

    # tracer point
    var = "tracer"
    test_axes = ["X", "Y", "Z"]
    test_dx = ["dx_e", "dy_n", "dz_w"]
    for ax, dx in zip(test_axes, test_dx):
        test_single_derivative(ax, ds[var], ds[dx])

    # zonal velocity point
    var = "u"
    test_dx = ["dx_t", "dy_ne", "dz_w"]
    for ax, dx in zip(test_axes, test_dx):
        test_single_derivative(ax, ds[var], ds[dx])

    # meridional velocity point
    var = "v"
    test_dx = ["dx_ne", "dy_t", "dz_w"]
    for ax, dx in zip(test_axes, test_dx):
        test_single_derivative(ax, ds[var], ds[dx])

    # vertical velocity point
    var = "wt"
    test_dx = ["dx_e", "dy_n", "dz_t"]
    for ax, dx in zip(test_axes, test_dx):
        test_single_derivative(ax, ds[var], ds[dx])


def test_derivative_b_grid():
    # test derivatives with synthetic B grid data

    ds, coords, metrics = datasets_grid_metric("B")
    grid = Grid(ds, coords=coords, metrics=metrics)

    # run this for each axis and each field in dataset
    def test_single_derivative(axis, fld, dx):

        dvar_dx = grid.derivative(fld, axis)
        expected = grid.diff(fld, axis) / dx
        abs_diff = np.sum(np.abs(dvar_dx - expected))

        assert dvar_dx.equals(expected.reset_coords(drop=True))

    # tracer point
    var = "tracer"
    test_axes = ["X", "Y", "Z"]
    test_dx = ["dx_e", "dy_n", "dz_w"]
    for ax, dx in zip(test_axes, test_dx):
        test_single_derivative(ax, ds[var], ds[dx])

    # zonal velocity point
    var = "u"
    test_dx = ["dx_n", "dy_e", "dz_w"]
    for ax, dx in zip(test_axes, test_dx):
        test_single_derivative(ax, ds[var], ds[dx])

    # meridional velocity point
    var = "v"
    test_dx = ["dx_n", "dy_e", "dz_w"]
    for ax, dx in zip(test_axes, test_dx):
        test_single_derivative(ax, ds[var], ds[dx])

    # vertical velocity point
    var = "wt"
    test_dx = ["dx_e", "dy_n", "dz_t"]
    for ax, dx in zip(test_axes, test_dx):
        test_single_derivative(ax, ds[var], ds[dx])
