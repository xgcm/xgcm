from __future__ import print_function
from future.utils import iteritems
from .grid import Axis


def generate_axis(ds,
                  axis,
                  name,
                  axis_dim,
                  pos_from='center',
                  pos_to='left',
                  wrap=None,
                  pad=None,
                  new_name=None,
                  raw_switch=True):
    """
    Creates c-grid dimensions (or coordinates) along an axis of
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with gridinformation used to construct c-grid
    axis : str
        The appropriate xgcm axis. E.g. 'X' for longitudes.
    name : str
        The name of the variable in ds, providing the original grid.
    axis_dim : str
        The dimension of ds[name] corresponding to axis. If name itself is a
        dimension, this should be equal to name.
    pos_from : {'center','left','right'}
        Position of the gridpoints given in 'ds'.
    pos_to : {'center','left','right'}
        Position of the gridpoints to be generated.
    wrap : None or float
        If specified, marks the value of discontinuity across boundary, e.g.
        360 for global longitude values and 180 for global latitudes.
    pad : {None, float, 'auto'}
        If specified, determines the padding to be applied across boundary.
        If float is specified, that value is used as padding. Auto attempts to
        pad linearly extrapolated values. Can be useful for e.g. depth
        coordinates (to reconstruct 0 depth). Can lead to unexpected values
        when coordinate is multidimensional.
    new_name : str
        Name of the inferred grid variable. Defaults to name+'_inferred'
    raw_switch : bool
        Determines if the attributes are created from scratch. Should be
        enabled for dimensions and deactivated for multidimensional
        coordinates. These can only be calculated after the dims are created.
    """
    if new_name is None:
        new_name = name+'_inferred'

    # Determine the relative position to interpolate to based on current and
    # desired position
    relative_pos_to = position_to_relative(pos_from, pos_to)

    if (wrap is not None) and (pad is not None):
        raise RuntimeError('Coordinate cannot be wrapped and padded at the\
                            same time')
    elif (wrap is None) and (pad is not None):
        if pad == 'auto':
            left_pad, right_pad = auto_pad(ds[name], axis_dim)
            if relative_pos_to == 'right':
                fill_value = right_pad
            elif relative_pos_to == 'left':
                fill_value = left_pad
        else:
            fill_value = pad
        periodic = False
        boundary = 'fill'
    elif (wrap is not None) and (pad is None):
        periodic = True
        fill_value = 0.0
        boundary = None
    else:
        raise RuntimeError('Either "wrap" or "pad" have to be specified')

    ds = ds.copy()

    # For a set of coordinates there are two fundamental cases. The coordinates
    # are a) one dimensional (dimensions) or 2) multidimensional. These are
    # separated by the keyword raw_switch.
    # These two cases are treated differently because for each dataset we need
    # to recreate all a) cases before we can proceed to 2), hence this is
    # really the 'raw' data processing step. If we have working one dimensional
    # coordinates (e.g. after we looped over the axes_dims_dict, we can use the
    # regular xgcm.Axis to interpolate multidimensional coordinates.
    # This assures that any changes to the Axis.interp method can directly
    # propagate to this module.

    if raw_switch:
        # Input coordinate has to be declared as center,
        # or xgcm.Axis throws error. Will be rewrapped below.
        ds[name] = fill_attrs(ds[name], 'center', axis)

        ax = Axis(ds, axis, periodic=periodic, wrap=wrap)
        ds.coords[new_name] = \
            ax._neighbor_binary_func_raw(ds[name],
                                         raw_interp,
                                         relative_pos_to,
                                         boundary=boundary,
                                         fill_value=fill_value)
        # ds.coords[new_name] = ax.interp(ds[name], relative_pos_to,
        #                                 boundary=boundary,
        #                                 fill_value=fill_value)
        # ax__neighbor_binary_func_raw(da, ax., to,
        #                                           boundary=boundary,
        #                                           fill_value=fill_value)

        # Place the correct attributes
        ds[name] = fill_attrs(ds[name], pos_from, axis)
        ds[new_name] = fill_attrs(ds[new_name], pos_to, axis)
    else:
        ax = Axis(ds, axis, periodic=periodic, wrap=wrap)
        ds.coords[new_name] = ax.interp(ds[name], pos_to, boundary=boundary,
                                        fill_value=fill_value)
    return ds


def generate_grid_ds(ds,
                         axes_dims_dict,
                         axes_coords_dict=None,
                         position=None,
                         wrap=None,
                         pad=None):
    """
    Add c-grid dimensions and coordinates (optional) to observational Dataset

    Parameters
    ----------
    ds : xarray.Dataset
     Dataset with gridinformation used to construct c-grid
    axes_dims_dict : dict
     Dict with information on the dimension in ds corrsponding to the xgcm
     axis. E.g. {'X':'lon','Y':'lat'}
    axes_coords_dict : dict
     Dict with information on the coordinates in ds corrsponding to the
     xgcm axis. E.g. {'X':'geolon','Y':'geolat'}
    position : {None,tuple, dict}
     Position of the gridpoints given in 'ds' and the desired position to be
     generated. Defaults to ('center','left'). Can be a tuple like
     ('center','left'), or a dict with corresponding axes
     (e.g. {'X':('center','left'),'Z':('left','center')})
    wrap : {None, float, dict}
     Specifies the discontinuity at the boundary to wrap e.g. longitudes
     without artifacts. Can be defined globally (for all fields defined in
     axes_dims_dict and axes_coords_dict) {float, None} or per dataset
     variable (dict e.g. {'longitude':360,'latitude':180})
    pad : {None, float, 'auto'}
     Specifies the discontinuity at the boundary to wrap e.g. longitudes
     without artifacts. Can be defined globally (for all fields defined in
     axes_dims_dict and axes_coords_dict) {float, None} or per dataset
     variable ({dict} e.g. {'z':'auto','latitude':0.0})
    """

    if axes_coords_dict is not None:
        combo_dict = [axes_dims_dict, axes_coords_dict]
    else:
        combo_dict = [axes_dims_dict]

    for di, dd in enumerate(combo_dict):
        if di == 0:
            raw_switch = True
            infer_dim = False
        elif di == 1:
            raw_switch = False
            infer_dim = True

        for ax in dd.keys():
            # Get variable name
            ax_v = dd[ax]
            # Get dimension name
            if infer_dim:
                ax_d = axes_dims_dict[ax]
            else:
                ax_d = ax_v

            # Parse position
            pos_from, pos_to = parse_position(position, ax)
            # Pass wrap characteristics
            is_wrapped = parse_wrap_pad(wrap, ax_v)
            # Pass pad characteristics
            is_padded = parse_wrap_pad(pad, ax_v)
            ds = generate_axis(ds, ax, ax_v, ax_d,
                               pos_from=pos_from, pos_to=pos_to,
                               wrap=is_wrapped, pad=is_padded,
                               raw_switch=raw_switch)
    return ds


def parse_wrap_pad(in_val, varname):
    """Parse wrap or pad parameters"""
    if isinstance(in_val, dict):
        try:
            is_valued = in_val[varname]
        except KeyError:
            # Set defaults
            is_valued = None
    else:
        is_valued = in_val
    return is_valued


def parse_position(position, axname, pos_default=('center', 'left')):
    if isinstance(position, dict):
        try:
            pos_from = position[axname][0]
        except KeyError:
            pos_from = pos_default[0]
        try:
            pos_to = position[axname][1]
        except KeyError:
            pos_to = pos_default[1]
    elif isinstance(position, tuple):
        pos_from = position[0]
        pos_to = position[1]
    else:
        # Set defaults
        pos_from = pos_default[0]
        pos_to = pos_default[1]
    return pos_from, pos_to


def position_to_relative(pos_from, pos_to):
    """Translate from to positions in relative movement"""
    if ((pos_from == 'left' and pos_to == 'center') or
         (pos_from == 'center' and pos_to == 'right')):
            to = 'right'
    elif ((pos_from == 'center' and pos_to == 'left') or
          (pos_from == 'right' and pos_to == 'center')):
            to = 'left'
    else:
        raise RuntimeError("Cannot infer '%s' coordinates \
    from '%s'" % (pos_to, pos_from))
    return to


def raw_interp(data_left, data_right):
    # TODO: This is not great, but I am not sure how to pass the nested
    # function in xgcm.grid.Axis.interp to Axis._neighbor_binary_func_raw
    # instead.
    return 0.5*(data_left + data_right)


def auto_pad(da, dim):
    "infer padding values from data array by linear extrapolation"
    da_min = da.min().data
    da_max = da.max().data
    # The difference between values at the top and
    # bottom is used to pad the array. For multidimensional arrays the min
    # difference is chosen. This could lead to undesired results if the
    # depth spacing is spatially irregular
    min_diff = da.diff(dim).isel(**{dim: 0}).min()
    max_diff = da.diff(dim).isel(**{dim: -1}).min()
    min_ex = da_min.data - min_diff.data
    max_ex = da_max.data + max_diff.data
    #TODO: This assumes that the dim is increasing. Build check or option for
    # decreasing coordinates
    return min_ex, max_ex


def fill_attrs(da, pos, axis):
    """Replace comdo attributes according to pos and axis"""
    attrs = da.attrs
    attrs['axis'] = axis

    if pos == 'center':
        attrs.pop('c_grid_axis_shift', None)
    elif pos == 'left':
        attrs['c_grid_axis_shift'] = -0.5
    elif pos == 'right':
        attrs['c_grid_axis_shift'] = 0.5

    da.attrs = attrs
    return da
