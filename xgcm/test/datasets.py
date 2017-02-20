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

datasets = {
    # the comodo example
    'nonperiodic_1d': xr.Dataset(
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
    'periodic_1d': xr.Dataset(
        coords={'XG': (['XG',], 2*np.pi/100*np.arange(0,100),
                        {'axis': 'X',
                         'c_grid_axis_shift': -0.5}),
                'XC': (['XC',], 2*np.pi/100*(np.arange(0,100)+0.005),
                                {'axis': 'X'})

        })
}


@pytest.fixture(scope="module", params=datasets.keys())
def all_datasets(request):
    return datasets[request.param]

@pytest.fixture(scope="module", params=['nonperiodic_1d'])
def nonperiodic_1d(request):
    return datasets[request.param]
