from __future__ import print_function
import xarray as xr

from xgcm.grid import Grid, Axis


def test_derivative_uniform_grid():
    # this is a uniform grid
    # a non-uniform grid would provide a more rigorous test
    dx = 10.0
    dy = 10.0
    arr = [[1.0, 2.0, 4.0, 3.0], \
           [4.0, 7.0, 1.0, 2.0], \
           [3.0, 1.0, 0.0, 9.0], \
           [8.0, 5.0, 2.0, 1.0]]
    ds = xr.Dataset(
        {"foo": (("XC","YC"), arr)},
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
        coords={"X": {"center": "XC", "left": "XG"},
                "Y": {"center": "YC", "left": "YG"}},
        metrics={("X",): ["dXC", "dXG"],
                 ("Y",): ["dYC", "dYG"]},
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

