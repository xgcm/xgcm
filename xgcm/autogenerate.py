from __future__ import print_function
from future.utils import iteritems
from xgcm.grid import Axis, raw_interp_function
import xarray as xr


def generate_axis(
    ds,
    axis,
    name,
    axis_dim,
    pos_from="center",
    pos_to="left",
    boundary_discontinuity=None,
    pad="auto",
    new_name=None,
    attrs_from_scratch=True,
):
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
    pos_from : {'center','left','right'}, optional
        Position of the gridpoints given in 'ds'.
    pos_to : {'left','center','right'}, optional
        Position of the gridpoints to be generated.
    boundary_discontinuity : {None, float}, optional
        If specified, marks the value of discontinuity across boundary, e.g.
        360 for global longitude values and 180 for global latitudes.
    pad : {'auto', None, float}, optional
        If specified, determines the padding to be applied across boundary.
        If float is specified, that value is used as padding. Auto attempts to
        pad linearly extrapolated values. Can be useful for e.g. depth
        coordinates (to reconstruct 0 depth). Can lead to unexpected values
        when coordinate is multidimensional.
    new_name : str, optional
        Name of the inferred grid variable. Defaults to name+'_'+pos_to'
    attrs_from_scratch : bool, optional
        Determines if the attributes are created from scratch. Should be
        enabled for dimensions and deactivated for multidimensional
        coordinates. These can only be calculated after the dims are created.
    """
    if not isinstance(ds, xr.Dataset):
        raise ValueError("'ds' needs to be xarray.Dataset")

    if new_name is None:
        new_name = name + "_" + pos_to

    # Determine the relative position to interpolate to based on current and
    # desired position

    relative_pos_to = _position_to_relative(pos_from, pos_to)

    # This is bloated. We can probably retire the 'auto' logic in favor of
    # using 'boundary' and 'fill_value'. But first lets see if this all works.

    if (boundary_discontinuity is not None) and (pad is not None):
        raise ValueError(
            "Coordinate cannot be wrapped and padded at the\
                            same time"
        )
    elif (boundary_discontinuity is None) and (pad is None):
        raise ValueError(
            'Either "boundary_discontinuity" or "pad" have \
                            to be specified'
        )

    if pad is None:
        fill_value = 0.0
        boundary = None
        periodic = True
    elif pad == "auto":
        fill_value = 0.0
        boundary = "extrapolate"
        periodic = False
    else:
        fill_value = pad
        boundary = "fill"
        periodic = False

    kwargs = dict(
        boundary_discontinuity=boundary_discontinuity,
        fill_value=fill_value,
        boundary=boundary,
        position_check=False,
    )

    ds = ds.copy()

    # For a set of coordinates there are two fundamental cases. The coordinates
    # are a) one dimensional (dimensions) or 2) multidimensional. These are
    # separated by the keyword attrs_from_scratch.
    # These two cases are treated differently because for each dataset we need
    # to recreate all a) cases before we can proceed to 2), hence this is
    # really the 'raw' data processing step. If we have working one dimensional
    # coordinates (e.g. after we looped over the axes_dims_dict, we can use the
    # regular xgcm.Axis to interpolate multidimensional coordinates.
    # This assures that any changes to the Axis.interp method can directly
    # propagate to this module.

    if attrs_from_scratch:
        # Input coordinate has to be declared as center,
        # or xgcm.Axis throws error. Will be rewrapped below.
        ds[name] = _fill_attrs(ds[name], "center", axis)

        ax = Axis(ds, axis, periodic=periodic)
        args = ds[name], raw_interp_function, relative_pos_to
        ds.coords[new_name] = ax._neighbor_binary_func_raw(*args, **kwargs)

        # Place the correct attributes
        ds[name] = _fill_attrs(ds[name], pos_from, axis)
        ds[new_name] = _fill_attrs(ds[new_name], pos_to, axis)
    else:
        kwargs.pop("position_check", None)
        ax = Axis(ds, axis, periodic=periodic)
        args = ds[name], pos_to
        ds.coords[new_name] = ax.interp(*args, **kwargs)
    return ds


def generate_grid_ds(
    ds,
    axes_dims_dict,
    axes_coords_dict=None,
    position=None,
    boundary_discontinuity=None,
    pad="auto",
    new_name=None,
):
    """
    Add c-grid dimensions and coordinates (optional) to observational Dataset

    Parameters
    ----------
    ds : xarray.Dataset
     Dataset with gridinformation used to construct c-grid
    axes_dims_dict : dict
     Dict with information on the dimension in ds corrsponding to the xgcm
     axis. E.g. {'X':'lon','Y':'lat'}
    axes_coords_dict : dict, optional
     Dict with information on the coordinates in ds corrsponding to the
     xgcm axis. E.g. {'X':'geolon','Y':'geolat'}
    position : {None,tuple, dict}, optional
     Position of the gridpoints given in 'ds' and the desired position to be
     generated. Defaults to ('center','left'). Can be a tuple like
     ('center','left'), or a dict with corresponding axes
     (e.g. {'X':('center','left'),'Z':('left','center')})
    boundary_discontinuity : {None, float, dict}, optional
     Specifies the discontinuity at the boundary to wrap e.g. longitudes
     without artifacts. Can be defined globally (for all fields defined in
     axes_dims_dict and axes_coords_dict) {float, None} or per dataset
     variable (dict e.g. {'longitude':360,'latitude':180})
    pad : {'auto', None, float}, optional
     Specifies the padding at the boundary to extend values past the boundary.
     Can be defined globally (for all fields defined in
     axes_dims_dict and axes_coords_dict) {float, None} or per dataset
     variable ({dict} e.g. {'z':'auto','latitude':0.0})
    new_name : str, optional
     Name of the inferred grid variable. Defaults to name+'_'+position[1]
    """

    if axes_coords_dict is not None:
        combo_dict = [axes_dims_dict, axes_coords_dict]
    else:
        combo_dict = [axes_dims_dict]

    for di, dd in enumerate(combo_dict):
        if di == 0:
            attrs_from_scratch = True
            infer_dim = False
        elif di == 1:
            attrs_from_scratch = False
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
            pos_from, pos_to = _parse_position(position, ax)
            # Pass wrap characteristics
            is_discontinous = _parse_boundary_params(boundary_discontinuity, ax_v)
            # Pass pad characteristics
            is_padded = _parse_boundary_params(pad, ax_v)
            ds = generate_axis(
                ds,
                ax,
                ax_v,
                ax_d,
                pos_from=pos_from,
                pos_to=pos_to,
                boundary_discontinuity=is_discontinous,
                pad=is_padded,
                new_name=new_name,
                attrs_from_scratch=attrs_from_scratch,
            )
    return ds


def _parse_boundary_params(in_val, varname):
    """Parse boundary_discontinuity or pad parameters"""
    if isinstance(in_val, dict):
        try:
            is_valued = in_val[varname]
        except KeyError:
            # Set defaults
            is_valued = None
    else:
        is_valued = in_val
    return is_valued


def _parse_position(position, axname, pos_default=("center", "left")):
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


def _position_to_relative(pos_from, pos_to):
    """Translate from to positions in relative movement"""
    if (pos_from == "left" and pos_to == "center") or (
        pos_from == "center" and pos_to == "right"
    ):
        to = "right"
    elif (pos_from == "center" and pos_to == "left") or (
        pos_from == "right" and pos_to == "center"
    ):
        to = "left"
    elif pos_from == "center" and pos_to == "outer":
        to = "outer"
    elif pos_from == "center" and pos_to == "inner":
        to = "inner"
    else:
        raise RuntimeError(
            "Cannot infer '%s' coordinates \
    from '%s'"
            % (pos_to, pos_from)
        )
    return to


def _fill_attrs(da, pos, axis):
    """Replace comdo attributes according to pos and axis"""
    attrs = da.attrs
    attrs["axis"] = axis

    if pos == "center":
        attrs.pop("c_grid_axis_shift", None)
    elif pos in ["left", "outer"]:
        attrs["c_grid_axis_shift"] = -0.5
    elif pos in ["right", "inner"]:
        attrs["c_grid_axis_shift"] = 0.5

    da.attrs = attrs
    return da
