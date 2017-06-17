from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np

# example from comodo website
# http://pycomodo.forge.imag.fr/norm.html
# netcdf example {
#         dimensions:
#                 ni = 9 ;
#                 ni_u = 10 ;
#         variables:
#                 float ni(ni) ;
#                         ni:axis = "X" ;
#                         ni:standard_name = "x_grid_index" ;
#                         ni:long_name = "x-dimension of the grid" ;
#                         ni:c_grid_dynamic_range = "2:8" ;
#                 float ni_u(ni_u) ;
#                         ni_u:axis = "X" ;
#                         ni_u:standard_name = "x_grid_index_at_u_location" ;
#                         ni_u:long_name = "x-dimension of the grid" ;
#                         ni_u:c_grid_dynamic_range = "3:8" ;
#                         ni_u:c_grid_axis_shift = -0.5 ;
#         data:
#                 ni = 1, 2, 3, 4, 5, 6, 7, 8, 9 ;
#                 ni_u = 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5 ;
# }

N = 100
datasets = {
    # the comodo example
    'nonperiodic_1d': xr.Dataset(
        {'data_ni': (['ni'], np.random.rand(9)),
         'data_ni_u': (['ni_u'], np.random.rand(10))},
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
        }),
    # my own invention
    'periodic_1d_left': xr.Dataset(
        {'data_g': (['XG'], np.random.rand(N)),
         'data_c': (['XC'], np.random.rand(N))},
        coords={'XG': (['XG',], 2*np.pi/N*np.arange(0,N),
                        {'axis': 'X',
                         'c_grid_axis_shift': -0.5}),
                'XC': (['XC',], 2*np.pi/N*(np.arange(0,N)+0.5),
                                {'axis': 'X'})

        }),
    'periodic_1d_right': xr.Dataset(
        {'data_g': (['XG'], np.random.rand(N)),
         'data_c': (['XC'], np.random.rand(N))},
        coords={'XG': (['XG',], 2*np.pi/N*np.arange(1,N+1),
                        {'axis': 'X',
                         'c_grid_axis_shift': 0.5}),
                'XC': (['XC',], 2*np.pi/N*(np.arange(0,N)-0.5),
                                {'axis': 'X'})

        }),
    'periodic_2d_left': xr.Dataset(
        {'data_gg': (['YG', 'XG'], np.random.rand(2*N, N)),
         'data_cc': (['YC', 'XC'], np.random.rand(2*N, N))},
        coords={'XG': (['XG',], 2*np.pi/N*np.arange(0,N),
                        {'axis': 'X',
                         'c_grid_axis_shift': -0.5}),
                'XC': (['XC',], 2*np.pi/N*(np.arange(0,N)+0.5),
                                {'axis': 'X'}),
                'YG': (['YG',], 2*np.pi/(2*N)*np.arange(0,2*N),
                                {'axis': 'Y',
                                 'c_grid_axis_shift': -0.5}),
                'YC': (['YC',], 2*np.pi/(2*N)*(np.arange(0,2*N)+0.5),
                                        {'axis': 'Y'})

        }),
}

expected_values = {
    'nonperiodic_1d': {'axes': {'X': {'center': 'ni', 'face': 'ni_u'}}},
    'periodic_1d_left': {'axes': {'X': {'center': 'XC', 'left': 'XG'}}},
    'periodic_1d_right': {'axes': {'X': {'center': 'XC', 'right': 'XG'}},
                          'shift': True},
    'periodic_2d_left': {'axes': {'X': {'center': 'XC', 'left': 'XG'},
                             'Y': {'center': 'YC', 'left': 'YG'}}}
}

@pytest.fixture(scope="module", params=datasets.keys())
def all_datasets(request):
    return datasets[request.param], expected_values[request.param]

@pytest.fixture(scope="module", params=['nonperiodic_1d'])
def nonperiodic_1d(request):
    return datasets[request.param], expected_values[request.param]

@pytest.fixture(scope="module", params=['periodic_1d_left', 'periodic_1d_right'])
def periodic_1d(request):
    return datasets[request.param], expected_values[request.param]

@pytest.fixture(scope="module", params=['periodic_2d_left'])
def periodic_2d(request):
    return datasets[request.param], expected_values[request.param]
