from .grid import Axis,Grid
import docrep

docstrings = docrep.DocstringProcessor(doc_key='My doc string')


class AutoGenerate(Axis):

        @docstrings.get_sectionsf('generate_coord')
        @docstrings.dedent
        def _generate_coord(self, da, to, boundary=None, fill_value=0.0,
                            wrap=None):
            """
            Generate (interpolate) unavailable grid variables.

            Parameters
            ----------
            da : xarray.DataArray
                The data on which to operate
            to : {'center', 'left', 'right', 'inner', 'outer'}
                The direction in which to shift the array. If not specified,
                default will be used.
            boundary : {None, 'fill', 'extend'}
                A flag indicating how to handle boundaries:
                * None:  Do not apply any boundary conditions. Raise an error
                if boundary conditions are required for the operation.
                * 'fill':  Set values outside the array boundary to fill_value
                  (i.e. a Neumann boundary condition.)
                * 'extend': Set values outside the array to the nearest array
                  value. (i.e. a limited form of Dirichlet boundary condition.)

            fill_value : float, optional
                 The value to use in the boundary condition with
                 `boundary='fill'`.

            wrap_value : float, optional
                if value is given, cyclic coordinate is wrapped approriately,
                e.g. longitude is wrapped around 180 or 360 discontinuity
            Returns
            -------
            da_i : xarray.DataArray
                The differenced data
            """
            #TODO I think we can modify xgcm.Axis so that this can all be
            # absorbed in there.

            position_from, dim = self._get_axis_coord(da)
            if to is None:
                to = self._default_shifts[position_from]

            # get the two neighboring sets of raw data
            data_left, data_right = \
                self._get_neighbor_data_pairs(da, to,
                                              boundary=boundary,
                                              fill_value=fill_value)

            # Is there a way to use the existing grid.Axis.interp logic?
            # For now just hardcode it
            data_new = 0.5*(data_left + data_right)

            return data_new


def generate(ds, axes, dims, to,
             x_wrap, y_wrap, z_pad,
             to_shift, from_shift, dim_switch=True):
        """
        Generate new dataarray from coordinates or dimensions, which might be
        missing in order to get a 'full' c-grid.

        INPUTS:

        ds - xarray.Dataset

        axes = dictionary containing comodo axis and corresponding DataArray
        in ds (e.g. axes={'X':'llon','Y':'llon'}). These could be
        multidimensional coordinates like geospatial longitudes/latitudes

        dims = dictionary of dimensons in ds corresponding to comodo axis
        e.g. (dims={'X':'lon','Y':'lon'}). These have to be names of
        strictly dimensions of ds.

        to - {'left','right'} Direction of shift, similar as the logic in
        xgcm.Axis, needed to apply shift correctly

        x_wrap - float: value of discontinuity in x direction (defaults to
        360 for global longitude)

        y_wrap - float: as x_wrap but for y direction (defaults to
        180 for global longitude)

        z_pad - float: boundary padding value for the depth interpolation. If
        not specified it will default to the min() or max() of the appropriate
        field, depending on 'to'

        dim_switch - Bool: switch between input for dimesions (need to
        load class from xgcm.autogenerate) and coordinates (uses existing
        xgc.grid logic)
        """

        #TODO: I can definitely infer the dims from the comodo stuff, but for
        #now leave it explicit

        for ki in axes.keys():
            if ki not in dims.keys():
                raise RuntimeError('dimension '+ki+' was not specified in axes \
                                    input')
            old_name = axes[ki]
            dim = dims[ki]
            new_name = old_name+'_inferred'

            if ki == 'X':
                periodic = True
                boundary = None
                fill_value = 0.0
                wrap = x_wrap
            elif ki == 'Y':
                periodic = True
                boundary = None
                fill_value = 0.0
                wrap = y_wrap
            elif ki == 'Z':
                periodic = False
                boundary = 'fill'
                # For depth decide between surface or bottom fill
                if z_pad is None:
                    # Infer limits from depth data
                    z_min = ds[old_name].min().data
                    z_max = ds[old_name].max().data
                    # The value below is an attempt on extrapolating the values
                    # linearly. The difference between values at the top and
                    # bottom is used to pad the depth array at the appropriate
                    # boundary. For multidimensional arrays the min difference
                    # is chosen. This could lead to undesired results if the
                    # depth spacing is spatially irregular (though thats a rare
                    # case IMO)
                    top_diff = ds[old_name].diff(dim).isel(**{dim: 0}).min()
                    bot_diff = ds[old_name].diff(dim).isel(**{dim: -1}).min()
                    top = z_min.data - top_diff.data
                    bot = z_max.data + bot_diff.data
                    if to is 'left':
                        fill_value = top
                    elif to is 'right':
                        fill_value = bot
                else:
                    fill_value = z_pad

            # Input coordinate has to be declared as center,
            # or xgcm.Axis throws error. Will be rewrapped below.
            attrs = ds[old_name].attrs
            attrs['axis'] = ki
            attrs.pop('c_grid_axis_shift', None)
            ds[old_name].attrs = attrs

            if dim_switch:
                ax = AutoGenerate(ds, ki, periodic=periodic, wrap=wrap)
                ds.coords[new_name] = ax._generate_coord(ds[old_name], to,
                                                         boundary=boundary,
                                                         fill_value=fill_value)

            else:
                ax = Axis(ds, ki, periodic=periodic, wrap=wrap)
                ds.coords[new_name] = ax.interp(ds[old_name], boundary=boundary,
                                         fill_value=fill_value)

            attrs_new = ds[new_name].attrs
            attrs_new['axis'] = ki
            attrs_new.pop('c_grid_axis_shift', None)
            # Reset appropriate attributes for old and new coordinates
            if abs(to_shift) > 0.0:
                attrs_new['c_grid_axis_shift'] = to_shift
            ds[new_name].attrs = attrs_new

            if abs(from_shift) > 0.0:
                ds[old_name].attrs['c_grid_axis_shift'] = from_shift

        return ds


def autogenerate_ds(ds,
                    axes={'X': 'lon', 'Y': 'lat'},
                    position='center',
                    coord_axes=None,
                    x_wrap=360,
                    y_wrap=180,
                    z_pad=None,
                    x_coord_wrap=360,
                    y_coord_wrap=180,
                    z_coord_pad=None):
    """
    Regenerate all c-grid information from an existing dataset

    Parameters
    ----------
    ds : xarray.Dataset
        The data on which to operate
    axes : axes on which to regenerate dims (and optional coords, see
        'coord_axes' input). Needs to be in the form of a dict with keys
        ('X','Y'or'Z'), the values need to be the corresponding dimension names
        in ds.
    position : {'center', 'left', 'right'}
        The position of the datapoints relative to the coordinates given
        This will usually be the cell center, but can sometimes be the
        lower left (option:'left') or perhaps upper right (option:'right')
        corner of the gridcell.
    coord_axes : (optional) Like 'axes' but for multidimensional coordinates in
        ds
    x_wrap : float - Value of discontinuity at the 'X' boundary
    y_wrap : float - Value of discontinuity at the 'Y' boundary
    z_pad : float -  Depth padding value. If None, it will be determined from
        data
    x_coord_wrap : float - Like 'x_wrap' but for coordinate variable
    y_coord_wrap : float - Like 'y_wrap' but for coordinate variable
    z_coord_pad : float -  Like 'z_pad' but for coordinate variable

    Returns
    -------
    ds : xarray.DataArray
        Dataset with inferred dims (and coordinates)
    """

    # set position of original coordinates and define how to interpolated
    # For now I am assuming that observational datasets will use the left/right
    # convention, rather then inner/outer...but that can be added.

    # 'to' input maps the correct movement of coordinates if the input
    # coord is set to center (this will be relabled appropriately below)
    if position == 'center':
        to = 'left'
        to_shift = -0.5
        from_shift = 0
    elif position == 'right':
        to = 'left'
        to_shift = 0
        from_shift = 0.5
    elif position == 'left':
        to = 'right'
        to_shift = 0
        from_shift = -0.5

    # Do we want to modify in place? or copy?
    ds = ds.copy()

    # Loop over all specified axes and build new coordinates
    ds = generate(ds, axes, axes, to, x_wrap, y_wrap,
                  z_pad, to_shift, from_shift, dim_switch=True)

    # Generate shifted multidimensional coordinates if specified in
    # 'coord_axes'
    if coord_axes:
        ds = generate(ds, coord_axes, axes, to, x_coord_wrap, y_coord_wrap,
                      z_coord_pad, to_shift, from_shift, dim_switch=False)

    return ds
