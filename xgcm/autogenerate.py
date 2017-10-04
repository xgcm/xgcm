from .grid import Axis
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
                e.g. longitude is wrapped around 190 or 360 discontinuity
            Returns
            -------
            da_i : xarray.DataArray
                The differenced data
            """

            position_from, dim = self._get_axis_coord(da)
            if to is None:
                to = self._default_shifts[position_from]

            # get the two neighboring sets of raw data
            data_left, data_right = \
                self._get_neighbor_data_pairs(da, to,
                                              boundary=boundary,
                                              fill_value=fill_value,
                                              wrap=wrap)

            # Is there a way to use the existing grid.Axis.interp logic?
            # For now just hardcode it
            data_new = 0.5*(data_left + data_right)

            return data_new


def autogenerate_ds(ds,
                    axes={'X': 'lon', 'Y': 'lat'},
                    position='center',
                    x_wrap=360,
                    y_wrap=180,
                    z_pad=None):
    """
    Add missing axis coordinates to dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The data on which to operate
    position : {'center', 'left', 'right'}
        The position of the datapoints relative to the coordinates given
        This will usually be the cell center, but can sometimes be the
        lower left (option:'left') or perhaps upper right (option:'right')
        corner of the gridcell

    boundary : {None, 'fill', 'extend'}
        A flag indicating how to handle boundaries:

        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.)

    fill_value : float, optional
         The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    da_i : xarray.DataArray
        The differenced data
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
    for ki in axes.keys():
        kk = axes[ki]
        new_name = kk+'_inferred'

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
                top = ds[kk].min().data-ds[kk].diff(kk).isel(**{kk: 0}).data
                bot = ds[kk].max().data+ds[kk].diff(kk).isel(**{kk: -1}).data
                if to is 'left':
                    fill_value = top
                elif to is 'right':
                    fill_value = bot
            else:
                fill_value = z_pad

        # Input coordinate has to be declared as center,
        # or xgcm.Axis throws error. Will be rewrapped below.
        attrs = ds.coords[kk].attrs
        attrs['axis'] = ki
        attrs.pop('c_grid_axis_shift', None)
        ds.coords[kk].attrs = attrs

        ax = AutoGenerate(ds, ki, periodic=periodic)
        ds.coords[new_name] = ax._generate_coord(ds[kk], to, boundary=boundary,
                                                 fill_value=fill_value,
                                                 wrap=wrap)
        # Reset appropriate attributes for old and new coordinates
        ds.coords[new_name].attrs['axis'] = ki
        if to_shift is not 0.0:
            ds.coords[new_name].attrs['c_grid_axis_shift'] = to_shift
        if from_shift is not 0.0:
            ds.coords[kk].attrs['c_grid_axis_shift'] = to_shift

    return ds
