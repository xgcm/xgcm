from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np
from xarray.testing import assert_identical, assert_allclose, assert_equal


from xgcm.autogenerate import auto_pad, fill_attrs, generate_axis, autogenerate_grid_ds

# create test datasets
dx = 0.5
dy = 1.0
dz = 0.5
a = np.random.rand(180, int(360/dx), int(10/dz))

x = np.arange(-180, 180, dx)
y = np.arange(-90, 90, dy)
z = np.arange(0, 10, dz)

xx, yy = np.meshgrid(x, y)
_, _, zz = np.meshgrid(x, y, z)

ds_original = xr.Dataset(
                         {'somedata': (['lat', 'lon', 'z'], a)},
                         coords={'lon': (['lon', ], x+(dx/2.0)),
                                 'lat': (['lat', ], y+(dy/2.0)),
                                 'z': (['z', ], z+(dx/2.0)),
                                 'llon': (['lat', 'lon'], xx+(dx/2.0)),
                                 'llat': (['lat', 'lon'], yy+(dy/2.0)),
                                 'zz': (['lat', 'lon', 'z'], zz+(dx/2.0))}
                        )
ds_out_left = xr.Dataset(
                     {'somedata': (['lat', 'lon', 'z'],
                                   a)},
                     coords={'lon': (['lon', ], x+(dx/2.0),
                                     {'axis': 'X'}),
                             'lat': (['lat', ], y+(dy/2.0),
                                     {'axis': 'Y'}),
                             'z': (['z', ], z+(dz/2.0),
                                   {'axis': 'Z'}),
                             'llon': (['lat', 'lon'], xx+(dx/2.0),
                                      {'axis': 'X'}),
                             'llat': (['lat', 'lon'], yy+(dy/2.0),
                                      {'axis': 'Y'}),
                             'zz': (['lat', 'lon', 'z'], zz+(dx/2.0),
                                    {'axis': 'Z'}),
                             'lon_inferred': (['lon_inferred', ], x,
                                              {'axis': 'X',
                                              'c_grid_axis_shift': -0.5}),
                             'lat_inferred': (['lat_inferred', ], y,
                                              {'axis': 'Y',
                                              'c_grid_axis_shift': -0.5}),
                             'z_inferred': (['z_inferred', ], z,
                                            {'axis': 'Z',
                                            'c_grid_axis_shift': -0.5}),
                             'llon_inferred': (['lat', 'lon_inferred', ], xx,
                                               {'axis': 'X',
                                               'c_grid_axis_shift': -0.5}),
                             'llat_inferred': (['lat_inferred', 'lon'], yy,
                                               {'axis': 'Y',
                                               'c_grid_axis_shift': -0.5}),
                             'zz_inferred': (['lat', 'lon', 'z_inferred'], zz,
                                             {'axis': 'Z',
                                             'c_grid_axis_shift': -0.5})}
                                             )

ds_out_right = xr.Dataset(
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
                             'llon': (['lat', 'lon'], xx+(dx/2.0),
                                      {'axis': 'X',
                                      'c_grid_axis_shift': -0.5}),
                             'llat': (['lat', 'lon'], yy+(dy/2.0),
                                      {'axis': 'Y',
                                      'c_grid_axis_shift': -0.5}),
                             'zz': (['lat', 'lon', 'z'], zz+(dx/2.0),
                                    {'axis': 'Z',
                                    'c_grid_axis_shift': -0.5}),
                             'lon_inferred': (['lon_inferred', ], x+dx,
                                              {'axis': 'X'}),
                             'lat_inferred': (['lat_inferred', ], y+dy,
                                              {'axis': 'Y'}),
                             'z_inferred': (['z_inferred', ], z+dz,
                                            {'axis': 'Z'}),
                             'llon_inferred': (['lat', 'lon_inferred', ], xx+dx,
                                               {'axis': 'X'}),
                             'llat_inferred': (['lat_inferred', 'lon'], yy+dy,
                                               {'axis': 'Y'}),
                             'zz_inferred': (['lat', 'lon', 'z_inferred'], zz+dz,
                                             {'axis': 'Z'})}
                        )

ds_out_center = xr.Dataset(
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
                             'llon': (['lat', 'lon'], xx+(dx/2.0),
                                      {'axis': 'X',
                                      'c_grid_axis_shift': 0.5}),
                             'llat': (['lat', 'lon'], yy+(dy/2.0),
                                      {'axis': 'Y',
                                      'c_grid_axis_shift': 0.5}),
                             'zz': (['lat', 'lon', 'z'], zz+(dx/2.0),
                                    {'axis': 'Z',
                                    'c_grid_axis_shift': 0.5}),
                             'lon_inferred': (['lon_inferred', ], x,
                                              {'axis': 'X'}),
                             'lat_inferred': (['lat_inferred', ], y,
                                              {'axis': 'Y'}),
                             'z_inferred': (['z_inferred', ], z,
                                            {'axis': 'Z'}),
                             'llon_inferred': (['lat', 'lon_inferred', ], xx,
                                               {'axis': 'X'}),
                             'llat_inferred': (['lat_inferred', 'lon'], yy,
                                               {'axis': 'Y'}),
                             'zz_inferred': (['lat', 'lon', 'z_inferred'], zz,
                                             {'axis': 'Z'})}
                        )


def test_generate_axis():
    # This case should raise the error:'Either "wrap" or "pad" have to be
    # specified'
    # a = generate_axis(ds_original, 'X', 'lon',
    #                   pos_from='center',
    #                   pos_to='right')
    a = generate_axis(ds_original.copy(), 'X', 'lon', 'lon',
                      pos_from='center',
                      pos_to='right',
                      wrap=360)
    b = generate_axis(ds_original.copy(), 'Y', 'lat', 'lat',
                      pos_from='center',
                      pos_to='left',
                      wrap=180)
    c = generate_axis(ds_original.copy(), 'Z', 'z', 'z',
                      pos_from='left',
                      pos_to='center',
                      pad='auto')
    assert_allclose(a['lon_inferred'], ds_out_right['lon_inferred'])
    assert_allclose(b['lat_inferred'], ds_out_left['lat_inferred'])
    assert_allclose(c['z_inferred'], ds_out_right['z_inferred'])

    # Mulitdim cases
    aa = generate_axis(a, 'X', 'llon', 'lon',
                       pos_from='center',
                       pos_to='right',
                       wrap=360,
                       raw_switch=False)
    bb = generate_axis(b, 'Y', 'llat', 'lat',
                       pos_from='center',
                       pos_to='left',
                       wrap=180,
                       raw_switch=False)
    cc = generate_axis(c, 'Z', 'zz', 'z',
                       pos_from='left',
                       pos_to='center',
                       pad='auto',
                       raw_switch=False)
    assert_allclose(aa['llon_inferred'], ds_out_right['llon_inferred'])
    assert_allclose(bb['llat_inferred'], ds_out_left['llat_inferred'])
    assert_allclose(cc['zz_inferred'], ds_out_right['zz_inferred'])


def test_autogenerate_grid_ds():
    # simple case...just the dims
    axis_dims = {'X': 'lon', 'Y': 'lat', 'Z': 'z'}
    axis_coords = {'X': 'llon', 'Y': 'llat', 'Z': 'zz'}
    ds_old = ds_original.copy()
    ds_new = autogenerate_grid_ds(ds_old, axis_dims,
                                  wrap={'lon': 360, 'lat': 180},
                                  pad={'z': 'auto'})
    assert_equal(ds_new, ds_out_left.drop(['llon_inferred',
                                           'llat_inferred',
                                           'zz_inferred']))
    # TODO why are they not identical ? assert identical fails
    ds_new = autogenerate_grid_ds(ds_original,
                                  axis_dims,
                                  axis_coords,
                                  wrap={'lon': 360, 'lat': 180,
                                        'llon': 360, 'llat': 180},
                                  pad={'z': 'auto', 'zz': 'auto'})
    assert_equal(ds_new, ds_out_left)


def test_auto_pad():
    a, b = auto_pad(ds_original['z'], 'z')
    aa, bb = auto_pad(ds_original['zz'], 'z')
    assert np.isclose(a, -(dx/2))
    assert np.isclose(aa, -(dx/2))


def test_fill_attrs():
    a = fill_attrs(ds_out_right['lon'], 'right', 'X')
    assert a.attrs['axis'] == 'X'
    assert a.attrs['c_grid_axis_shift'] == 0.5

    a = fill_attrs(ds_out_right['lon'], 'left', 'Z')
    assert a.attrs['axis'] == 'Z'
    assert a.attrs['c_grid_axis_shift'] == -0.5

    a = fill_attrs(ds_out_right['lon'], 'center', 'Y')
    assert a.attrs['axis'] == 'Y'
    assert ('c_grid_axis_shift' not in a.attrs.keys())
