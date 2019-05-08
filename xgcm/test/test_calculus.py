from __future__ import print_function
import xarray as xr
import numpy as np

from xgcm.grid import Grid, Axis


def test_derivative_uniform_grid():
    # this is a uniform grid
    # a non-uniform grid would provide a more rigorous test
    dx = 10.
    ds = xr.Dataset({'foo': (('XC',), [1., 2., 4., 3.])},
                    coords={'XC': (('XC',), [0.5, 1.5, 2.5, 3.5]),
                            'XG': (('XG',), [0, 1., 2., 3.]),
                            'dXC': (('XC',), [dx, dx, dx, dx]),
                            'dXG': (('XG',), [dx, dx, dx, dx])})

    grid = Grid(ds, coords={'X': {'center': 'XC', 'left': 'XG'}},
                metrics={('X',): ['dXC', 'dXG']},
                periodic=True)

    dfoo_dx = grid.derivative(ds.foo, 'X')
    expected = grid.diff(ds.foo, 'X') / dx
    assert dfoo_dx.equals(expected)

def test_metrics_2d_grid():
    # this is a uniform grid
    # a non-uniform grid would provide a more rigorous test
    dx = 10.
    dy = 11.
    area = 120.
    ny, nx = 7, 9
    ds = xr.Dataset({'foo': (('YC', 'XC'), np.ones((ny, nx)))},
                    coords={'XC': (('XC',), np.arange(nx)),
                            'dX': (('XC',), np.full(nx, dx)),
                            'YC': (('YC',), np.arange(ny)),
                            'dY': (('YC',), np.full(ny, dy)),
                            'area': (('YC', 'XC'), np.full((ny, nx), area))})

    grid = Grid(ds, coords={'X': {'center': 'XC'}, 'Y': {'center': 'YC'}},
                    metrics={('X',): ['dX'], ('Y',): ['dY'],
                             ('X', 'Y'): ['area']})

    assert ds.dX.reset_coords(drop=True).equals(grid.get_metric(ds.foo, ('X',)))
    assert ds.dY.reset_coords(drop=True).equals(grid.get_metric(ds.foo, ('Y',)))
    assert ds.area.reset_coords(drop=True).equals(grid.get_metric(ds.foo, ('X', 'Y',)))
    assert ds.area.reset_coords(drop=True).equals(grid.get_metric(ds.foo, ('Y', 'X')))

    # try with no area metric:
    grid = Grid(ds, coords={'X': {'center': 'XC'}, 'Y': {'center': 'YC'}},
                    metrics={('X',): ['dX'], ('Y',): ['dY']})

    dxdy = (ds.dX * ds.dY).reset_coords(drop=True).transpose('YC', 'XC')
    actual = grid.get_metric(ds. foo, ('Y', 'X')).transpose('YC', 'XC')
    assert dxdy.equals(actual)
