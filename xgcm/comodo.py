from __future__ import print_function
from future.utils import iteritems
import xarray as xr

def assert_valid_comodo(ds):
    """Verify that the dataset meets comodo conventions

    Parameters
    ----------
    ds : xarray.dataset
    """

    # TODO: implement
    assert True


def get_all_axes(ds):
    axes = set()
    for d in ds.dims:
        if 'axis' in ds[d].attrs:
            axes.add(ds[d].attrs['axis'])
    return axes


def get_axis_coords(ds, axis_name):
    """Find the name of the coordinates associated with a comodo axis.

    Parameters
    ----------
    ds : xarray.dataset or xarray.dataarray
    axis_name : str
        The name of the axis to find (e.g. 'X')

    Returns
    -------
    coord_name : list
        The names of the coordinate matching that axis
    """

    coord_names = []
    for d in ds.dims:
        axis = ds[d].attrs.get('axis')
        if axis==axis_name:
            coord_names.append(d)
    return coord_names

def _assert_data_on_grid(da):
    pass

def _get_vars_by_attrs(ds, **attrs):
    """Return all variable / coordiante names matching an attribute dict.

    Parameters
    ----------
    ds : xarray.dataset or xarray.dataarray
    attrs: dict
        Attributes to match

    Returns
    -------
    varnames : list
    """

    matches = []
    for varname in ds:
        var = ds[varname]
        for key, val in iteritems(attrs):
            if key in var.attrs:
                if var.attrs[key] == val:
                    pass
