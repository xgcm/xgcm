from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np
from dask.array import from_array

from xgcm.grid import Grid, Axis, add_to_slice

@pytest.fixture(scope='module')
def ds():
    N = 25
    return xr.Dataset({'data_c': (['face', 'y', 'x'], np.random.rand(2, N, N))},
            coords={'x': (('x',), np.arange(N), {'axis': 'X'}),
                    'y': (('y',), np.arange(N), {'axis': 'Y'}),
                    'face' : (('face',), [0, 1])})

@pytest.fixture(scope='module')
def cs():
    # cubed-sphere
    N = 25
    return xr.Dataset({'data_c': (['face', 'y', 'x'], np.random.rand(6, N, N))},
            coords={'x': (('x',), np.arange(N), {'axis': 'X'}),
                    'y': (('y',), np.arange(N), {'axis': 'Y'}),
                    'face' : (('face',), [0, 1])})


def test_create_periodic_grid(ds):
    ds = ds.isel(face=0)
    grid = Grid(ds, periodic=True)
    for axis_name in grid.axes:
        axis = grid.axes[axis_name]
        assert axis._facedim is None
        connect_left, connect_right = axis._connections[None]
        assert connect_left[0] is None
        assert connect_right[0] is None
        assert connect_left[1] is axis
        assert connect_right[1] is axis
        assert connect_left[2] == False
        assert connect_right[2] == False


def test_get_periodic_grid_edge(ds):
    ds = ds.isel(face=0)
    grid = Grid(ds, periodic=True)

    xaxis = grid.axes['X']
    left_edge_data_x = xaxis._get_edge_data(ds.data_c)
    np.testing.assert_allclose(left_edge_data_x,
                               ds.data_c.isel(x=-1).data[:, None])
    right_edge_data_x = xaxis._get_edge_data(ds.data_c, is_left_edge=False)
    np.testing.assert_allclose(right_edge_data_x,
                               ds.data_c.isel(x=0).data[:, None])

    yaxis = grid.axes['Y']
    left_edge_data_y = yaxis._get_edge_data(ds.data_c)
    np.testing.assert_allclose(left_edge_data_y,
                               ds.data_c.isel(y=-1).data[None, :])
    right_edge_data_y = yaxis._get_edge_data(ds.data_c, is_left_edge=False)
    np.testing.assert_allclose(right_edge_data_y,
                               ds.data_c.isel(y=0).data[None, :])


def test_create_connected_grid(ds):
    # simplest scenario with one face connection
    grid = Grid(ds, face_connections=
                # name of the dimension to exchange over
                {'face':
                    # key: index of face
                    # value: another dictionary
                      # key: axis name
                      # value: a tuple of link specifiers
                      #      neighbor face index,
                      #        neighboring axis to connect to,
                      #          whether to reverse the connection
                    {0: {'X': (None, (1, 'X', False))},
                     1: {'X': ((0, 'X', False), None)}
                     }
                })

    xaxis = grid.axes['X']
    yaxis = grid.axes['Y']

    # make sure we have actual axis objects in the connection dict
    # this is a bad test because it tests the details of the implementation,
    # not the behavior. But it is useful for now
    assert xaxis._facedim == 'face'
    assert xaxis._connections[0][1][0] == 1
    assert xaxis._connections[0][1][1] is xaxis
    assert xaxis._connections[1][0][0] == 0
    assert xaxis._connections[1][0][1] is xaxis
