from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np
from xarray.testing import assert_allclose, assert_equal


from xgcm.autogenerate import auto_pad, fill_attrs, generate_axis, \
    generate_grid_ds, parse_wrap_pad, parse_position, position_to_relative

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
ds_original_1D = xr.Dataset(
                            {'somedata': (['z', ], np.array([1, 2, 3]))},
                            coords={'z': (['z', ], z[0:3])}
)
ds_original_1D_padded = xr.Dataset(
                                   {'somedata': (['z', ],
                                                 np.array([1, 2, 3]))},
                                   coords={'z': (['z', ],
                                           z[0:3]),
                                           'test': (['test', ],
                                                   z[0:3]+(dx/2.0))}
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
                             'llon_inferred': (['lat', 'lon_inferred'],
                                               xx+dx,
                                               {'axis': 'X'}),
                             'llat_inferred': (['lat_inferred', 'lon'],
                                               yy+dy,
                                               {'axis': 'Y'}),
                             'zz_inferred': (['lat', 'lon', 'z_inferred'],
                                             zz+dz,
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
    a = generate_axis(ds_original, 'X', 'lon', 'lon',
                      pos_from='center',
                      pos_to='right',
                      wrap=360)
    b = generate_axis(ds_original, 'Y', 'lat', 'lat',
                      pos_from='center',
                      pos_to='left',
                      wrap=180)
    c = generate_axis(ds_original, 'Z', 'z', 'z',
                      pos_from='left',
                      pos_to='center',
                      pad='auto')
    d = generate_axis(ds_original_1D, 'Z', 'z', 'z',
                      pos_from='left',
                      pos_to='center',
                      pad=1.0+dz,
                      new_name='test')
    assert_allclose(a['lon_inferred'], ds_out_right['lon_inferred'])
    assert_allclose(b['lat_inferred'], ds_out_left['lat_inferred'])
    assert_allclose(c['z_inferred'], ds_out_right['z_inferred'])
    assert_allclose(d['test'], ds_original_1D_padded['test'])

    # Mulitdim cases
    aa = generate_axis(a, 'X', 'llon', 'lon',
                       pos_from='center',
                       pos_to='right',
                       wrap=360,
                       attrs_from_scratch=False)
    bb = generate_axis(b, 'Y', 'llat', 'lat',
                       pos_from='center',
                       pos_to='left',
                       wrap=180,
                       attrs_from_scratch=False)
    cc = generate_axis(c, 'Z', 'zz', 'z',
                       pos_from='left',
                       pos_to='center',
                       pad='auto',
                       attrs_from_scratch=False)
    assert_allclose(aa['llon_inferred'], ds_out_right['llon_inferred'])
    assert_allclose(bb['llat_inferred'], ds_out_left['llat_inferred'])
    assert_allclose(cc['zz_inferred'], ds_out_right['zz_inferred'])

    with pytest.raises(RuntimeError):
        generate_axis(c['somedata'], 'Z', 'zz', 'z',
                      pos_from='left',
                      pos_to='center',
                      pad='auto',
                      attrs_from_scratch=False)
    with pytest.raises(RuntimeError):
        generate_axis(c, 'Z', 'zz', 'z', pad='auto', wrap=360)
    with pytest.raises(RuntimeError):
        generate_axis(c, 'Z', 'zz', 'z', pad=None, wrap=None)


def test_generate_grid_ds():
    # simple case...just the dims
    axis_dims = {'X': 'lon', 'Y': 'lat', 'Z': 'z'}
    axis_coords = {'X': 'llon', 'Y': 'llat', 'Z': 'zz'}
    ds_old = ds_original.copy()
    ds_new = generate_grid_ds(ds_old, axis_dims,
                              wrap={'lon': 360, 'lat': 180},
                              pad={'z': 'auto'})
    assert_equal(ds_new, ds_out_left.drop(['llon_inferred',
                                           'llat_inferred',
                                           'zz_inferred']))
    # TODO why are they not identical ? assert identical fails
    ds_new = generate_grid_ds(ds_original,
                              axis_dims,
                              axis_coords,
                              wrap={'lon': 360, 'lat': 180,
                                    'llon': 360, 'llat': 180},
                              pad={'z': 'auto', 'zz': 'auto'})
    assert_equal(ds_new, ds_out_left)


def test_parse_wrap_pad():
    assert parse_wrap_pad(360, 'anything') == 360
    assert parse_wrap_pad({'something': 360}, 'something') == 360
    assert parse_wrap_pad({'something': 360}, 'something_else') is None


@pytest.mark.parametrize('p_f, p_t', [('left', 'center'),
                                      ('center', 'left'),
                                      ('center', 'right'),
                                      ('right', 'center')])
def test_parse_position(p_f, p_t):
    default = ('center', 'left')
    assert parse_position((p_f, p_t), 'anything') == (p_f, p_t)
    assert parse_position({'something': (p_f, p_t)}, 'something') == (p_f, p_t)
    assert parse_position({'something': (p_f, p_t)}, 'somethong') == default


@pytest.mark.parametrize('p, relative', [(('left', 'center'), 'right'),
                                         (('center', 'left'), 'left'),
                                         (('center', 'right'), 'right'),
                                         (('right', 'center'), 'left')])
def test_position_to_relative(p, relative):
    assert position_to_relative(p[0], p[1]) == relative

    with pytest.raises(RuntimeError):
        position_to_relative('left', 'right')


def test_auto_pad():
    a = auto_pad(ds_original['z'], 'z', 'left')
    b = auto_pad(ds_original['z'], 'z', 'right')
    aa = auto_pad(ds_original['zz'], 'z', 'left')
    bb = auto_pad(ds_original['zz'], 'z', 'right')
    assert a == -(dx/2)
    assert b == 10 + (dx/2)
    assert aa == -(dx/2)
    assert bb == 10 + (dx/2)


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
