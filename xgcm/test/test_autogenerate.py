from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np


from xgcm.autogenerate import AutoGenerate, autogenerate_ds

# create test datasets
dx = 0.5
dy = 1.0
dz = 0.5
a = np.random.rand(180, int(360/dx), int(10/dz))

x = np.arange(-180, 180, dx)
y = np.arange(-90, 90, dy)
z = np.arange(0, 10, dz)

ds_original = xr.Dataset(
                         {'somedata': (['lat', 'lon', 'z'], a)},
                         coords={'lon': (['lon', ], x+(dx/2.0)),
                                 'lat': (['lat', ], y+(dy/2.0)),
                                 'z': (['z', ], z+(dx/2.0))}
                        )
ds_out_center = xr.Dataset(
                     {'somedata': (['lat', 'lon', 'z'],
                                   a)},
                     coords={'lon': (['lon', ], x+(dx/2.0),
                                     {'axis': 'X'}),
                             'lat': (['lat', ], y+(dy/2.0),
                                     {'axis': 'Y'}),
                             'z': (['z', ], z+(dz/2.0),
                                   {'axis': 'Z'}),
                             'lon_inferred': (['lon_inferred', ], x,
                                              {'axis': 'X',
                                              'c_grid_axis_shift': -0.5}),
                             'lat_inferred': (['lat_inferred', ], y,
                                              {'axis': 'Y',
                                              'c_grid_axis_shift': -0.5}),
                             'z_inferred': (['z_inferred', ], z,
                                            {'axis': 'Z',
                                            'c_grid_axis_shift': -0.5})}
                        )

ds_out_left = xr.Dataset(
                     {'somedata': (['lat', 'lon', 'z'],
                                   a)},
                     coords={'lon': (['lon', ], x+(dx/2.0),
                                     {'axis': 'X',
                                     'c_grid_axis_shift': -0.5}),
                             'lat': (['lat', ], y+(dy/2.0),
                                     {'axis': 'Y',
                                     'c_grid_axis_shift': -0.5}),
                             'z': (['z', ], z+(dz/2.0),
                                   {'axis': 'Z',
                                   'c_grid_axis_shift': -0.5}),
                             'lon_inferred': (['lon_inferred', ], x+dx,
                                              {'axis': 'X'}),
                             'lat_inferred': (['lat_inferred', ], y+dy,
                                              {'axis': 'Y'}),
                             'z_inferred': (['z_inferred', ], z+dz,
                                            {'axis': 'Z'})}
                        )

ds_out_right = xr.Dataset(
                     {'somedata': (['lat', 'lon', 'z'],
                                   a)},
                     coords={'lon': (['lon', ], x+(dx/2.0),
                                     {'axis': 'X',
                                     'c_grid_axis_shift': 0.5}),
                             'lat': (['lat', ], y+(dy/2.0),
                                     {'axis': 'Y',
                                     'c_grid_axis_shift': 0.5}),
                             'z': (['z', ], z+(dz/2.0),
                                   {'axis': 'Z',
                                   'c_grid_axis_shift': 0.5}),
                             'lon_inferred': (['lon_inferred', ], x,
                                              {'axis': 'X'}),
                             'lat_inferred': (['lat_inferred', ], y,
                                              {'axis': 'Y'}),
                             'z_inferred': (['z_inferred', ], z,
                                            {'axis': 'Z'})}
                        )


def test_autogenerate_ds():
    axes = {'X': 'lon', 'Y': 'lat', 'Z': 'z'}
    center = autogenerate_ds(ds_original, axes=axes, position='center')

    left = autogenerate_ds(ds_original, axes=axes, position='left')

    right = autogenerate_ds(ds_original, axes=axes, position='right')

    xr.testing.assert_identical(center, ds_out_center)
    for ke in left.coords.keys():
        print(left.coords[ke].attrs)
        print(ds_out_left.coords[ke].attrs)
    # xr.testing.assert_identical(left, ds_out_left)
    xr.testing.assert_identical(right, ds_out_right)
