from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np
from dask.array import from_array

from xgcm.grid import Grid, Axis, add_to_slice

def test_create_connected_grid():

    # create a square grid
    N = 25

    ds = xr.Dataset({'data_c': (['face', 'y', 'x'], np.random.rand(2, N, N))},
                coords={'x': (('x',), np.arange(N), {'axis': 'X'}),
                        'y': (('y',), np.arange(N), {'axis': 'Y'}),
                        'face' : (('face',), [0, 1])})

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
