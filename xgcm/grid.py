from __future__ import print_function
from __future__ import absolute_import
from future.utils import iteritems
from collections import OrderedDict
import docrep
import xarray as xr
import numpy as np

from . import comodo
from .duck_array_ops import _pad_array, _apply_boundary_condition, concatenate

docstrings = docrep.DocstringProcessor(doc_key='My doc string')



class Axis:
    """
    An object that represents a group of coodinates that all lie along the same
    physical dimension but at different positions with respect to a grid cell.
    There are four possible positions::

         Center
         |------o-------|------o-------|------o-------|------o-------|
               [0]            [1]            [2]            [3]

         Left
         |------o-------|------o-------|------o-------|------o-------|
        [0]            [1]            [2]            [3]

         Right
         |------o-------|------o-------|------o-------|------o-------|
                       [0]            [1]            [2]            [3]

         Inner
         |------o-------|------o-------|------o-------|------o-------|
                       [0]            [1]            [2]

         Outer
         |------o-------|------o-------|------o-------|------o-------|
        [0]            [1]            [2]            [3]            [4]

    The `center` position is the only one without the `c_grid_axis_shift`
    attribute, which must be present for the other four. However, the actual
    value of `c_grid_axis_shift` is ignored for `inner` and `outer`, which are
    differentiated by their length.
    """

    def __init__(self, ds, axis_name, periodic=True, default_shifts={}):
        """
        Create a new Axis object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant grid information. Coordinate attributes
            should conform to Comodo conventions [1]_.
        axis_name : str
            The name of the axis (should match axis attribute)
        periodic : bool, optional
            Whether the domain is periodic along this axis
        default_shifts : dict, optional
            Default mapping from and to grid positions
            (e.g. `{'center': 'left'}`). Will be inferred if not specified.


        REFERENCES
        ----------
        .. [1] Comodo Conventions http://pycomodo.forge.imag.fr/norm.html
        """

        self._ds = ds
        self._name = axis_name
        self._periodic = periodic

        self.coords = comodo.get_axis_positions_and_coords(ds, axis_name)
        # self.coords is a dictionary with the following structure
        #   key: position_name {'center' ,'left' ,'right', 'outer', 'inner'}
        #   value: xr.DataArray of the coordinate
        #     (dimension name is accessible via the .name attribute)

        # set default position shifts
        fallback_shifts = {'center': ('left', 'right', 'outer', 'inner'),
                           'left': ('center',), 'right': ('center',),
                           'outer': ('center',), 'inner': ('center',)}
        self._default_shifts = {}
        for pos in self.coords:
            # use user-specified value if present
            if pos in default_shifts:
                self._default_shifts[pos] = default_shifts[pos]
            else:
                for possible_shift in fallback_shifts[pos]:
                    if possible_shift in self.coords:
                        self._default_shifts[pos] = possible_shift
                        break

        ########################################################################
        # DEVELOPER DOCUMENTATION
        #
        # The attributes below are my best attempt to represent grid topology
        # in a general way. The data structures are complicated, but I can't
        # think of any way to simplify them.
        #
        # self._facedim (str) is the name of a dimension (e.g. 'face') or None.
        # If it is None, that means that the grid topology is _simple_, i.e.
        # that this is not a cubed-sphere grid or similar. For example:
        #
        #     ds.dims == ('time', 'lat', 'lon')
        #
        # If _facedim is set to a dimension name, that means that shifting
        # grid positions requires exchanging data among multiple "faces"
        # (a.k.a. "tiles", "facets", etc.). For this to work, there must be a
        # dimension corresponding to the different faces. This is `_facedim`.
        # For example:
        #
        #     ds.dims == ('time', 'face', 'lat', 'lon')
        #
        # In this case, `self._facedim == 'face'`
        #
        # We initialize all of this to None and let the `Grid` class handle
        # setting these attributes for complex geometries.
        self._facedim = None
        #
        # `self._connections` is a dictionary. It contains information about the
        # connectivity among this axis and other axes.
        # It should have the structure
        #
        #     {facedim_index: ((left_facedim_index, left_axis, left_reverse),
        #                      (right_facedim_index, right_axis, right_reverse)}
        #
        # `facedim_index` : a value used to index the `self._facedim` dimension
        #   (If `self._facedim` is `None`, then there should be only one key in
        #   `facedim_index` and that key should be `None`.)
        # `left_facedim_index` : the facedim index of the neighbor to the left.
        #   (If `self._facedim` is `None`, this must also be `None`.)
        # `left_axis` : an `Axis` object for the values to the left of this axis
        # `left_reverse` : bool, whether the connection should be reversed. By
        #   default, the left side of this axis will be connected to the right
        #   side of the neighboring axis. `left_reverse` overrides this and
        #   instead connects to the left side of the neighboring axis
        self._connections = {None: (None, None)}

        # now we implement periodic coordinates by setting appropriate
        # connections
        if periodic:
            self._connections = {None: ((None, self, False),
                                        (None, self, False))}


    def __repr__(self):
        is_periodic = 'periodic' if self._periodic else 'not periodic'
        summary = ["<xgcm.Axis '%s' %s>" % (self._name, is_periodic)]
        summary.append('Axis Coodinates:')
        summary += self._coord_desc()
        return '\n'.join(summary)

    def _coord_desc(self):
        summary = []
        for name, coord in iteritems(self.coords):
            coord_info = ('  * %-8s %s (%g)' % (name, coord.name, len(coord)))
            if name in self._default_shifts:
                coord_info += ' --> %s' % self._default_shifts[name]
            summary.append(coord_info)
        return summary


    @docstrings.get_sectionsf('neighbor_binary_func')
    @docstrings.dedent
    def _neighbor_binary_func(self, da, f, to, boundary=None, fill_value=0.0,
                              boundary_discontinuity=None):
        """
        Apply a function to neighboring points.

        Parameters
        ----------
        da : xarray.DataArray
            The data on which to operate
        f : function
            With signature f(da_left, da_right, shift)
        to : {'center', 'left', 'right', 'inner', 'outer'}
            The direction in which to shift the array. If not specified,
            default will be used.
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
        position_from, dim = self._get_axis_coord(da)
        if to is None:
            to = self._default_shifts[position_from]

        data_new = self._neighbor_binary_func_raw(da, f, to,
                                                  boundary=boundary,
                                                  fill_value=fill_value,
                                                  boundary_discontinuity=
                                                  boundary_discontinuity)
        # wrap in a new xarray wrapper
        da_new = self._wrap_and_replace_coords(da, data_new, to)

        return da_new

    docstrings.delete_params('neighbor_binary_func.parameters', 'f')

    def _neighbor_binary_func_raw(self, da, f, to, boundary=None,
                                  fill_value=0.0,
                                  boundary_discontinuity=None):

        # get the two neighboring sets of raw data
        data_left, data_right = \
            self._get_neighbor_data_pairs(da,
                                          to,
                                          boundary=boundary,
                                          fill_value=fill_value,
                                          boundary_discontinuity=\
                                          boundary_discontinuity)

        # apply the function
        data_new = f(data_left, data_right)

        return data_new

    def _get_edge_data(self, da, is_left_edge=True, boundary=None,
                       fill_value=0.0, ignore_connections=False):
        """Get the appropriate edge data given axis connectivity and / or
        boundary conditions.
        """

        position, this_dim = self._get_axis_coord(da)
        this_axis_num = da.get_axis_num(this_dim)

        def face_edge_data(fnum, face_axis, count=1):
            # get the edge data for a single face

            # There will not necessarily be connection data for every face
            # for every axis. If there is no connection data, fnum will not
            # be a key for self._connections.

            if fnum in self._connections:
                # it should always be a len 2 tuple
                face_connection = self._connections[fnum][
                                            0 if is_left_edge else 1]
            else:
                face_connection = None

            if (face_connection is None) or ignore_connections:
                # no connection: use specified boundary condition instead
                if self._facedim:
                    da_face = da.isel(**{self._facedim: slice(fnum, fnum+1)})
                else:
                    da_face = da
                return _apply_boundary_condition(da_face, this_dim,
                                                 is_left_edge,
                                                 boundary=boundary,
                                                 fill_value=0.0)

            neighbor_fnum, neighbor_axis, reverse = face_connection

            # check for consistency
            if face_axis is None:
                assert neighbor_fnum is None

            # Build up a slice that selects the correct edge region for a
            # given face. We work directly with variables rather than
            # DataArrays in the hopes of greater efficiency, avoiding
            # indexing / alignment

            # Start with getting all the data
            edge_slice = [slice(None),] * da.ndim
            if face_axis is not None:
                # get the neighbor face
                edge_slice[face_axis] = slice(neighbor_fnum, neighbor_fnum+1)

            # get the edge we need
            neighbor_edge_dim = neighbor_axis.coords[position].name
            neighbor_edge_axis_num = da.get_axis_num(neighbor_edge_dim)
            if (is_left_edge and not reverse):
                neighbor_edge_slice = slice(-count, None)
            else:
                neighbor_edge_slice = slice(None, count)
            edge_slice[neighbor_edge_axis_num] = neighbor_edge_slice

            # the orthogonal dimension need to be reoriented if we are
            # connected to the other axis. Is this because of some deep
            # topological principle?
            if neighbor_axis is not self:
                ortho_axis = da.get_axis_num(self.coords[position].name)
                ortho_slice = slice(None, None, -1)
                edge_slice[ortho_axis] = ortho_slice

            edge = da.variable[tuple(edge_slice)].data

            # the axis of the edge on THIS face is not necessarily the same
            # as the axis on the OTHER face
            if neighbor_axis is not self:
                edge = edge.swapaxes(neighbor_edge_axis_num, this_axis_num)

            return edge

        if self._facedim:
            face_axis_num = da.get_axis_num(self._facedim)
            arrays = [face_edge_data(fnum, face_axis_num)
                      for fnum in da[self._facedim].values]
            edge_data = concatenate(arrays, face_axis_num)
        else:
            edge_data = face_edge_data(None, None)

        return edge_data


    def _extend_left(self, da, boundary=None, fill_value=0.0,
                     ignore_connections=False):
        axis_num = self._get_axis_dim_num(da)
        edge_data = self._get_edge_data(da, is_left_edge=True,
                                        boundary=boundary,
                                        fill_value=fill_value,
                                        ignore_connections=ignore_connections)
        return concatenate([edge_data, da.data], axis=axis_num)


    def _extend_right(self, da, boundary=None, fill_value=0.0,
                      ignore_connections=False):
        axis_num = self._get_axis_dim_num(da)
        edge_data = self._get_edge_data(da, is_left_edge=False,
                                        boundary=boundary,
                                        fill_value=fill_value,
                                        ignore_connections=ignore_connections)
        return concatenate([da.data, edge_data], axis=axis_num)


    def _get_neighbor_data_pairs(self, da, position_to, boundary=None,
                                 fill_value=0.0, ignore_connections=False,
                                 boundary_discontinuity=None):

        position_from, dim = self._get_axis_coord(da)
        axis_num = da.get_axis_num(dim)

        boundary_kwargs = dict(boundary=boundary, fill_value=fill_value,
                               ignore_connections=ignore_connections)

        valid_positions = ['outer', 'inner', 'left', 'right', 'center']

        if position_to not in valid_positions:
            raise ValueError("`%s` is not a valid axis position" % position_to)

        if position_to not in self.coords:
            raise ValueError("This axis doesn't contain a `%s` position"
                             % position_to)

        transition = (position_from, position_to)

        if ((transition == ('outer', 'center')) or
            (transition == ('center', 'inner'))):
            # don't need any edge values
            left = da.isel(**{dim: slice(None, -1)}).data
            right = da.isel(**{dim: slice(1, None)}).data
        elif ((transition == ('center', 'outer')) or
              (transition == ('inner', 'center'))):
            # need both edge values
            left = self._extend_left(da, **boundary_kwargs)
            right = self._extend_right(da, **boundary_kwargs)
        elif (transition == ('center', 'left') or
              transition == ('right', 'center')):
            # need to slice *after* getting edge because otherwise we could
            # mess up complicated connections (e.g. cubed-sphere)
            left = self._extend_left(da, **boundary_kwargs)
            # unfortunately left is not an xarray so we have to slice
            # it the long numpy way
            slc = axis_num * (slice(None),) + (slice(0, -1),)
            left = left[slc]
            right = da.data
        elif (transition == ('center', 'right') or
              transition == ('left', 'center')):
            # need to slice *after* getting edge because otherwise we could
            # mess up complicated connections (e.g. cubed-sphere)
            right = self._extend_right(da, **boundary_kwargs)
            # unfortunately left is not an xarray so we have to slice
            # it the long numpy way
            slc = axis_num * (slice(None),) + (slice(1, None),)
            right = right[slc]
            left = da.data
        else:
            raise NotImplementedError(' to '.join(transition) +
                                      ' transition not yet supported.')

        return left, right


    def _get_neighbor_data_pairs_old(self, da, position_to, boundary=None,
                                 fill_value=0.0, boundary_discontinuity=None):
        """Returns data_left, data_right.
        boundary_discontinuity option enables periodic coordinate interpolation
        (see xgcm.autogenerate)"""

        position_from, dim = self._get_axis_coord(da)

        valid_positions = ['outer', 'inner', 'left', 'right', 'center']
        if position_to not in valid_positions:
            raise ValueError("`%s` is not a valid axis position" % position_to)

        if self._periodic and boundary:
            raise ValueError("`boundary=%s` is not allowed with periodic "
                             "axis %s." % (boundary, self._name))

        if position_from == position_to:
            raise ValueError("Can't get neighbor pairs for the same position.")

        transition = (position_from, position_to)

        if ((transition == ('outer', 'center')) or
            (transition == ('center', 'inner'))):
            # doesn't matter if domain is periodic or not
            left = da.isel(**{dim: slice(None, -1)}).data
            right = da.isel(**{dim: slice(1, None)}).data
        elif ((transition == ('center', 'outer')) or
              (transition == ('inner', 'center'))):
            # pad both sides of the array
            left = _pad_array(da, dim, left=True,
                              boundary=boundary, fill_value=fill_value)
            right = _pad_array(da, dim, boundary=boundary,
                               fill_value=fill_value)
        # TODO: figure out if it matters whether we slice the original array
        # before or after padding
        elif (not self._periodic and ((transition == ('center', 'left')) or
                                       (transition == ('right', 'center')))):
            # pad only left
            left = _pad_array(da.isel(**{dim: slice(0, -1)}), dim, left=True,
                              boundary=boundary, fill_value=fill_value)
            right = da.data
        elif (not self._periodic and ((transition == ('center', 'right')) or
                                      (transition == ('left', 'center')))):
            # pad only left
            right = _pad_array(da.isel(**{dim: slice(1, None)}), dim,
                               boundary=boundary, fill_value=fill_value)
            left = da.data
        elif (self._periodic and ((transition == ('center', 'left')) or
                                  (transition == ('right', 'center')))):

            left = da.roll(**{dim: 1})
            if boundary_discontinuity is not None:
                left = add_to_slice(left, dim, 0, -boundary_discontinuity)
            left = left.data
            right = da.data
        elif (self._periodic and ((transition == ('center', 'right')) or
                                  (transition == ('left', 'center')))):
            left = da.data
            right = da.roll(**{dim: -1})
            if boundary_discontinuity is not None:
                right = add_to_slice(right, dim, -1, boundary_discontinuity)
            right = right.data
        else:
            is_periodic = 'periodic' if self._periodic else 'non-periodic'
            raise NotImplementedError(' to '.join(transition) +
                                      ' (%s) transition not yet supported.'
                                      % is_periodic)

        return left, right


    @docstrings.dedent
    def interp(self, da, to=None, boundary=None, fill_value=0.0,
               boundary_discontinuity=None):
        """
        Interpolate neighboring points to the intermediate grid point along
        this axis.

        Parameters
        ----------
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_i : xarray.DataArray
            The interpolated data

        """

        return self._neighbor_binary_func(da, raw_interp_function, to,
                                          boundary, fill_value,
                                          boundary_discontinuity)

    @docstrings.dedent
    def diff(self, da, to=None, boundary=None, fill_value=0.0,
             boundary_discontinuity=None):
        """
        Difference neighboring points to the intermediate grid point.

        Parameters
        ----------
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        return self._neighbor_binary_func(da, raw_diff_function, to,
                                          boundary, fill_value,
                                          boundary_discontinuity)

    @docstrings.dedent
    def cumsum(self, da, to=None, boundary=None, fill_value=0.0):
        """
        Cumulatively sum a DataArray, transforming to the intermediate axis
        position.

        Parameters
        ----------
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_cum : xarray.DataArray
            The cumsummed data
        """

        pos, dim = self._get_axis_coord(da)

        if to is None:
            to = self._default_shifts[pos]

        # first use xarray's cumsum method
        da_cum = da.cumsum(dim=dim)

        boundary_kwargs = dict(boundary=boundary, fill_value=fill_value)

        # now pad / trim the data as necessary
        # here we enumerate all the valid possible shifts
        if ((pos == 'center' and to == 'right') or
            (pos == 'left' and to == 'center')):
            # do nothing, this is the default for how cumsum works
            data = da_cum.data
        elif ((pos == 'center' and to == 'left') or
              (pos == 'right' and to == 'center')):
            data = _pad_array(da_cum.isel(**{dim: slice(0, -1)}), dim,
                              left=True, **boundary_kwargs)
        elif ((pos == 'center' and to == 'inner') or
              (pos == 'outer' and to == 'center')):
            data = da_cum.isel(**{dim: slice(0, -1)}).data
        elif ((pos == 'center' and to == 'outer') or
              (pos == 'inner' and to == 'center')):
            data = _pad_array(da_cum, dim, left=True, **boundary_kwargs)
        else:
            raise ValueError("From `%s` to `%s` is not a valid position "
                             "shift for cumsum operation." % (pos, to))

        da_cum_newcoord = self._wrap_and_replace_coords(da, data, to)
        return da_cum_newcoord

    def _wrap_and_replace_coords(self, da, data_new, position_to):
        """
        Take the base coords from da, the data from data_new, and return
        a new DataArray with a coordinate on position_to.
        """
        position_from, old_dim = self._get_axis_coord(da)
        try:
            new_coord = self.coords[position_to]
        except KeyError:
            raise KeyError("Position '%s' was not found in axis.coords."
                           % position_to)
        new_dim = new_coord.name

        orig_dims = da.dims

        coords = OrderedDict()
        dims = []
        for d in orig_dims:
            if d == old_dim:
                dims.append(new_dim)
                coords[new_dim] = new_coord
            else:
                dims.append(d)
                coords[d] = da.coords[d]

        return xr.DataArray(data_new, dims=dims, coords=coords)


    def _get_axis_coord(self, da):
        """Return the position and name of the axis coordiante in a DataArray.
        """
        for position, coord in iteritems(self.coords):
            # TODO: should we have more careful checking of alignment here?
            if coord.name in da.dims:
                return position, coord.name

        raise KeyError("None of the DataArray's dims %s were found in axis "
                       "coords." % repr(da.dims))


    def _get_axis_dim_num(self, da):
        """Return the dimension number of the axis coordinate in a DataArray.
        """
        _, coord_name = self._get_axis_coord(da)
        return da.get_axis_num(coord_name)



class Grid:
    """
    An object with multiple :class:`xgcm.Axis` objects representing different
    independent axes.
    """

    def __init__(self, ds, check_dims=True, periodic=True, default_shifts={},
                 face_connections=None):
        """
        Create a new Grid object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant grid information. Coordinate attributes
            should conform to Comodo conventions [1]_.
        check_dims : bool, optional
            Whether to check the compatibility of input data dimensions before
            performing grid operations.
        periodic : {True, False, list}
            Whether the grid is periodic (i.e. "wrap-around"). If a list is
            specified (e.g. `['X', 'Y']`), the axis names in the list will be
            be periodic and any other axes founds will be assumed non-periodic.
        default_shifts : dict
            A dictionary of dictionaries specifying default grid position
            shifts (e.g. `{'X': {'center': 'left', 'left': 'center'}}`)

        REFERENCES
        ----------
        .. [1] Comodo Conventions http://pycomodo.forge.imag.fr/norm.html
        """
        self._ds = ds
        self._check_dims = check_dims

        all_axes = comodo.get_all_axes(ds)

        self.axes = OrderedDict()
        for axis_name in all_axes:
            try:
                is_periodic = axis_name in periodic
            except TypeError:
                is_periodic = periodic
            if axis_name in default_shifts:
                axis_default_shifts = default_shifts[axis_name]
            else:
                axis_default_shifts = {}
            self.axes[axis_name] = Axis(ds, axis_name, is_periodic,
                                        default_shifts=axis_default_shifts)

        if face_connections is not None:
            self._assign_face_connections(face_connections)


    def _assign_face_connections(self, fc):
        """Check a dictionary of face connections to make sure all the links are
        consistent.
        """

        if len(fc) > 1:
            raise ValueError('Only one face dimension is supported for now. '
                             'Instead found %r' % repr(fc.keys()))

        # we will populate this with the axes we find in face_connections
        axis_connections = {}

        facedim = list(fc.keys())[0]
        assert facedim in self._ds

        face_links = fc[facedim]
        for fidx, face_axis_links in face_links.items():
            for axis, axis_links in face_axis_links.items():
                # initialize the axis dict if necssary
                if axis not in axis_connections:
                    axis_connections[axis] = {}
                link_left, link_right = axis_links

                def check_neighbor(link, position):
                    if link is None:
                        return
                    idx, ax, rev = link
                    # need to swap position if the link is reversed
                    correct_position = int(not position) if rev else position
                    try:
                        neighbor_link = face_links[idx][ax][correct_position]
                    except (KeyError, IndexError):
                        raise KeyError("Couldn't find a face link for face %r"
                                       "in axis %r at position %r"
                                       % (idx, ax, correct_position))
                    idx_n, ax_n, rev_n = neighbor_link
                    if ax not in self.axes:
                        raise KeyError('axis %r is not a valid axis' % ax)
                    if ax_n not in self.axes:
                        raise KeyError('axis %r is not a valid axis' % ax_n)
                    if idx not in self._ds[facedim].values:
                        raise IndexError('%r is not a valid index for face'
                                         'dimension %r' % (idx, facedim))
                    if idx_n not in self._ds[facedim].values:
                        raise IndexError('%r is not a valid index for face'
                                         'dimension %r' % (idx, facedim))
                    # check for consistent links from / to neighbor
                    if (idx_n != fidx) or (ax_n != axis) or (rev_n != rev):
                        raise ValueError("Face link mismatch: neighbor doesn't"
                                         " correctly link back to this face. "
                                         "face: %r, axis: %r, position: %r, "
                                         "rev: %r, link: %r, neighbor_link: %r"
                                         % (fidx, axis, position, rev, link,
                                            neighbor_link))
                    # convert the axis name to an acutal axis object
                    actual_axis = self.axes[ax]
                    return idx, actual_axis, rev

                left = check_neighbor(link_left, 1)
                right = check_neighbor(link_right, 0)
                axis_connections[axis][fidx] = (left, right)

        for axis, axis_links in axis_connections.items():
            self.axes[axis]._facedim = facedim
            self.axes[axis]._connections = axis_links


    def __repr__(self):
        summary = ['<xgcm.Grid>']
        for name, axis in iteritems(self.axes):
            is_periodic = 'periodic' if axis._periodic else 'not periodic'
            summary.append('%s Axis (%s):' % (name, is_periodic))
            summary += axis._coord_desc()
        return '\n'.join(summary)

    @docstrings.dedent
    def interp(self, da, axis, **kwargs):
        """
        Interpolate neighboring points to the intermediate grid point along
        this axis.

        Parameters
        ----------
        axis : str
            Name of the axis on which ot act
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_i : xarray.DataArray
            The interpolated data
        """

        ax = self.axes[axis]
        return ax.interp(da, **kwargs)

    @docstrings.dedent
    def diff(self, da, axis, **kwargs):
        """
        Difference neighboring points to the intermediate grid point.

        Parameters
        ----------
        axis : str
            Name of the axis on which ot act
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data
        """

        ax = self.axes[axis]
        return ax.diff(da, **kwargs)


    @docstrings.dedent
    def cumsum(self, da, axis, **kwargs):
        """
        Cumulatively sum a DataArray, transforming to the intermediate axis
        position.

        Parameters
        ----------
        axis : str
            Name of the axis on which ot act
        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        da_i : xarray.DataArray
            The cumsummed data
        """

        ax = self.axes[axis]
        return ax.cumsum(da, **kwargs)


def add_to_slice(da, dim, sl, value):
    # split array into before, middle and after (if slice is the
    # beginning or end before or after will be empty)
    before = da[{dim: slice(0, sl)}]
    middle = da[{dim: sl}]
    after = da[{dim: slice(sl+1, None)}]
    if sl < -1:
        raise RuntimeError('slice can not be smaller value than -1')
    elif sl == -1:
        da_new = xr.concat([before, middle+value], dim=dim)
    else:
        da_new = xr.concat([before, middle+value, after], dim=dim)
    # then add 'value' to middle and concatenate again
    return da_new


def raw_interp_function(data_left, data_right):
    # linear, centered interpolation
    # TODO: generalize to higher order interpolation
    return 0.5*(data_left + data_right)


def raw_diff_function(data_left, data_right):
    return data_right - data_left






_other_docstring_options="""
    * 'dirichlet'
       The value of the array at the boundary point is specified by
       `fill_value`.
    * 'neumann'
       The value of the array diff at the boundary point is
       specified[1]_ by `fill_value`.

        .. [1] https://en.wikipedia.org/wiki/Dirichlet_boundary_condition
        .. [2] https://en.wikipedia.org/wiki/Neumann_boundary_condition
"""
