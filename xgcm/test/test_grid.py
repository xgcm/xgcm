from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np

from xgcm.grid import Grid

@pytest.fixture
def input_dataset():
    """
    example from comodo website
    http://pycomodo.forge.imag.fr/norm.html
    netcdf example {
            dimensions:
                    ni = 9 ;
                    ni_u = 10 ;
            variables:
                    float ni(ni) ;
                            ni:axis = "X" ;
                            ni:standard_name = "x_grid_index" ;
                            ni:long_name = "x-dimension of the grid" ;
                            ni:c_grid_dynamic_range = "2:8" ;
                    float ni_u(ni_u) ;
                            ni_u:axis = "X" ;
                            ni_u:standard_name = "x_grid_index_at_u_location" ;
                            ni_u:long_name = "x-dimension of the grid" ;
                            ni_u:c_grid_dynamic_range = "3:8" ;
                            ni_u:c_grid_axis_shift = -0.5 ;
            data:
                    ni = 1, 2, 3, 4, 5, 6, 7, 8, 9 ;
                    ni_u = 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5 ;
    }
    """

    return xr.Dataset(
        coords={'ni': (['ni',], np.arange(1,10),
                        {'axis': 'X',
                         'standard_name': 'x_grid_index',
                         'long_name': 'x-dimension of the grid',
                         'c_grid_dynamic_range': '2:8'}),
                'ni_u': (['ni_u',], np.arange(0.5,10),
                         {'axis': 'X',
                          'standard_name': 'x_grid_index_at_u_location',
                          'long_name': 'x-dimension of the grid',
                          'c_grid_dynamic_range': '3:8',
                          'c_grid_axis_shift': -0.5})
        })

def test_create_grid(input_dataset):
    grid = Grid(input_dataset)

def test_interp(input_dataset):

    ds = input_dataset

    # a linear gradient in the ni direction
    grad = 0.24
    data_c = grad * ds['ni']

    grid = Grid(ds, x_periodic=False)
    data_u = grid.interp(data_c, 'X')

    # check that the dimensions are right
    assert data_u.dims == ('ni_u',)
    xr.testing.assert_equal(data_u.ni_u, ds.ni_u[1:-1])
    assert len(data_u.ni_u)==len(data_u)

    # check that the values are right
    # what do we do about the boundary conditions?
    np.testing.assert_allclose(
        data_u.values,
        0.5 * (data_c.values[1:] + data_c.values[:-1])
    )
