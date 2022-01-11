import functools
import inspect
import itertools
import operator
import warnings
from collections import OrderedDict

import docrep  # type: ignore
import numpy as np
import xarray as xr

from . import comodo, gridops
from .grid_ufunc import GridUFunc, _signatures_equivalent, apply_as_grid_ufunc
from .metrics import iterate_axis_combinations

try:
    import numba  # type: ignore

    from .transform import conservative_interpolation, linear_interpolation
except ImportError:
    numba = None

from typing import Any, Dict, Iterable, Union

docstrings = docrep.DocstringProcessor(doc_key="My doc string")


def _maybe_promote_str_to_list(a):
    # TODO: improve this
    if isinstance(a, str):
        return [a]
    else:
        return a


_VALID_BOUNDARY = [None, "fill", "extend", "extrapolate"]


_XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG = {
    "periodic": "wrap",
    "fill": "constant",
    "extend": "edge",
}


class Grid:
    """
    An object with multiple :class:`xgcm.Axis` objects representing different
    independent axes.
    """

    def __init__(
        self,
        ds,
        check_dims=True,
        periodic=True,
        default_shifts={},
        face_connections=None,
        coords=None,
        metrics=None,
        boundary=None,
        fill_value=None,
    ):
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
            specified (e.g. ``['X', 'Y']``), the axis names in the list will be
            be periodic and any other axes founds will be assumed non-periodic.
        default_shifts : dict
            A dictionary of dictionaries specifying default grid position
            shifts (e.g. ``{'X': {'center': 'left', 'left': 'center'}}``)
        face_connections : dict
            Grid topology
        coords : dict, optional
            Specifies positions of dimension names along axes X, Y, Z, e.g
            ``{'X': {'center': 'XC', 'left: 'XG'}}``.
            Each key should be an axis name (e.g., `X`, `Y`, or `Z`) and map
            to a dictionary which maps positions (`center`, `left`, `right`,
            `outer`, `inner`) to dimension names in the dataset
            (in the example above, `XC` is at the `center` position and `XG`
            at the `left` position along the `X` axis).
            If the values are not present in ``ds`` or are not dimensions,
            an error will be raised.
        metrics : dict, optional
            Specification of grid metrics mapping axis names (X, Y, Z) to corresponding
            metric variable names in the dataset
            (e.g. {('X',):['dx_t'], ('X', 'Y'):['area_tracer', 'area_u']}
            for the cell distance in the x-direction ``dx_t`` and the
            horizontal cell areas ``area_tracer`` and ``area_u``, located at
            different grid positions).
        boundary : {None, 'fill', 'extend', 'extrapolate', dict}, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            Optionally a dict mapping axis name to seperate values for each axis
            can be passed.
        fill_value : {float, dict}, optional
            The value to use in boundary conditions with `boundary='fill'`.
            Optionally a dict mapping axis name to seperate values for each axis
            can be passed.
        keep_coords : boolean, optional
            Preserves compatible coordinates. False by default.

        REFERENCES
        ----------
        .. [1] Comodo Conventions https://web.archive.org/web/20160417032300/http://pycomodo.forge.imag.fr/norm.html
        """
        self._ds = ds
        self._check_dims = check_dims

        if coords:
            all_axes = coords.keys()
        else:
            all_axes = comodo.get_all_axes(ds)
            coords = {}

        # check coords input validity
        for axis, positions in coords.items():
            for pos, dim in positions.items():
                if not (dim in ds.variables or dim in ds.dims):
                    raise ValueError(
                        f"Could not find dimension `{dim}` (for the `{pos}` position on axis `{axis}`) in input dataset."
                    )
                if dim not in ds.dims:
                    raise ValueError(
                        f"Input `{dim}` (for the `{pos}` position on axis `{axis}`) is not a dimension in the input datasets `ds`."
                    )

        # Convert all inputs to axes-kwarg mappings
        # Parse axis properties
        boundary = self._as_axis_kwarg_mapping(boundary, axes=all_axes)
        fill_value = self._as_axis_kwarg_mapping(fill_value, axes=all_axes)

        # Parse list input. This case does only apply to periodic.
        # Since we plan on deprecating it soon handle it here, so we can easily
        # remove it later
        if isinstance(periodic, list):
            periodic = {axname: True for axname in periodic}
        periodic = self._as_axis_kwarg_mapping(periodic, axes=all_axes)

        # Set properties on grid object. I think we are better of
        # getting completely rid of the axis object and storing/getting info from the grid object properties
        # (all of which need to be supplied/converted to axis-value mapping).
        self.boundary: Dict[str, Union[str, float, int]] = boundary
        self.fill_value: Dict[str, Union[str, float, int]] = fill_value
        self.periodic: Dict[str, Union[str, float, int]] = periodic
        # TODO we probably want to properly define these as class properties with setter/getter?
        # TODO: This also needs to check valid inputs for each one.

        """
        # Populate axes. Much of this is just for backward compatibility.
        self.axes = OrderedDict()
        for axis_name in all_axes:
            is_periodic = periodic.get(axis_name, False)

            if axis_name in default_shifts:
                axis_default_shifts = default_shifts[axis_name]
            else:
                axis_default_shifts = {}

            if isinstance(boundary, dict):
                axis_boundary = boundary.get(axis_name, None)
            elif isinstance(boundary, str) or boundary is None:
                axis_boundary = boundary
            else:
                raise ValueError(
                    f"boundary={boundary} is invalid. Please specify a dictionary "
                    "mapping axis name to a boundary option; a string or None."
                )
            # TODO Reimplement value checking on grid properties. See comment above
            if isinstance(fill_value, dict):
                axis_fillvalue = fill_value.get(axis_name, None)
            elif isinstance(fill_value, (int, float)) or fill_value is None:
                axis_fillvalue = fill_value
            else:
                raise ValueError(
                    f"fill_value={fill_value} is invalid. Please specify a dictionary "
                    "mapping axis name to a boundary option; a number or None."
                )

            self.axes[axis_name] = Axis(
                ds,
                axis_name,
                is_periodic,
                default_shifts=axis_default_shifts,
                coords=coords.get(axis_name),
                boundary=axis_boundary,
                fill_value=axis_fillvalue,
            )
        """

        if face_connections is not None:
            self._assign_face_connections(face_connections)

        self._metrics = {}

        if metrics is not None:
            for key, value in metrics.items():
                self.set_metrics(key, value)

        # Finish setup

    def _as_axis_kwarg_mapping(
        self,
        kwargs: Union[Any, Dict[str, Any]],
        axes: Iterable[str] = None,
    ) -> Dict[str, Any]:
        """Convert kwarg input into dict for each available axis
        E.g. for a grid with 2 axes for the keyword argument `periodic`
        periodic = True --> periodic = {'X': True, 'Y':True}
        or if not all axes are provided, the other axes will be parsed as defaults (None)
        periodic = {'X':True} --> periodic={'X': True, 'Y':None}
        """
        if axes is None:
            axes = self.axes

        parsed_kwargs: Dict[str, Any] = dict()
        if isinstance(kwargs, dict):
            parsed_kwargs = kwargs
        else:
            for axis in axes:
                parsed_kwargs[axis] = kwargs
        return parsed_kwargs

    def _assign_face_connections(self, fc):
        """Check a dictionary of face connections to make sure all the links are
        consistent.
        """

        if len(fc) > 1:
            raise ValueError(
                "Only one face dimension is supported for now. "
                "Instead found %r" % repr(fc.keys())
            )

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
                        raise KeyError(
                            "Couldn't find a face link for face %r"
                            "in axis %r at position %r" % (idx, ax, correct_position)
                        )
                    idx_n, ax_n, rev_n = neighbor_link
                    if ax not in self.axes:
                        raise KeyError("axis %r is not a valid axis" % ax)
                    if ax_n not in self.axes:
                        raise KeyError("axis %r is not a valid axis" % ax_n)
                    if idx not in self._ds[facedim].values:
                        raise IndexError(
                            "%r is not a valid index for face"
                            "dimension %r" % (idx, facedim)
                        )
                    if idx_n not in self._ds[facedim].values:
                        raise IndexError(
                            "%r is not a valid index for face"
                            "dimension %r" % (idx, facedim)
                        )
                    # check for consistent links from / to neighbor
                    if (idx_n != fidx) or (ax_n != axis) or (rev_n != rev):
                        raise ValueError(
                            "Face link mismatch: neighbor doesn't"
                            " correctly link back to this face. "
                            "face: %r, axis: %r, position: %r, "
                            "rev: %r, link: %r, neighbor_link: %r"
                            % (fidx, axis, position, rev, link, neighbor_link)
                        )
                    # convert the axis name to an acutal axis object
                    actual_axis = self.axes[ax]
                    return idx, actual_axis, rev

                left = check_neighbor(link_left, 1)
                right = check_neighbor(link_right, 0)
                axis_connections[axis][fidx] = (left, right)

        for axis, axis_links in axis_connections.items():
            self.axes[axis]._facedim = facedim
            self.axes[axis]._connections = axis_links

    def set_metrics(self, key, value, overwrite=False):
        metric_axes = frozenset(_maybe_promote_str_to_list(key))
        axes_not_found = [ma for ma in metric_axes if ma not in self.axes]
        if len(axes_not_found) > 0:
            raise KeyError(
                f"Metric axes {axes_not_found!r} not compatible with grid axes {tuple(self.axes)!r}"
            )

        metric_value = _maybe_promote_str_to_list(value)
        for metric_varname in metric_value:
            if metric_varname not in self._ds.variables:
                raise KeyError(
                    f"Metric variable {metric_varname} not found in dataset."
                )

        existing_metric_axes = set(self._metrics.keys())
        if metric_axes in existing_metric_axes:
            value_exist = self._metrics.get(metric_axes)
            # resetting coords avoids potential broadcasting / alignment issues
            value_new = self._ds[metric_varname].reset_coords(drop=True)
            did_overwrite = False
            # go through each existing value until data array with matching dimensions is selected
            for idx, ve in enumerate(value_exist):
                # double check if dimensions match
                if set(value_new.dims) == set(ve.dims):
                    if overwrite:
                        # replace existing data array with new data array input
                        self._metrics[metric_axes][idx] = value_new
                        did_overwrite = True
                    else:
                        raise ValueError(
                            f"Metric variable {ve.name} with dimensions {ve.dims} already assigned in metrics."
                            f" Overwrite {ve.name} with {metric_varname} by setting overwrite=True."
                        )
            # if no existing value matches new value dimension-wise, just append new value
            if not did_overwrite:
                self._metrics[metric_axes].append(value_new)
        else:
            # no existing metrics for metric_axes yet; initialize empty list
            self._metrics[metric_axes] = []
            for metric_varname in metric_value:
                metric_var = self._ds[metric_varname].reset_coords(drop=True)
                self._metrics[metric_axes].append(metric_var)

    def _get_dims_from_axis(self, da, axis):
        dim = []
        axis = _maybe_promote_str_to_list(axis)
        for ax in axis:
            if ax in self.axes:
                all_dim = self.axes[ax].coords.values()
                matching_dim = [di for di in all_dim if di in da.dims]
                if len(matching_dim) == 1:
                    dim.append(matching_dim[0])
                else:
                    raise ValueError(
                        f"Did not find single matching dimension {da.dims} from {da.name} corresponding to axis {ax}, got {matching_dim}."
                    )
            else:
                raise KeyError(f"Did not find axis {ax} from data array {da.name}")
        return dim

    def get_metric(self, array, axes):
        """
        Find the metric variable associated with a set of axes for a particular
        array.

        Parameters
        ----------
        array : xarray.DataArray
            The array for which we are looking for a metric. Only its dimensions are considered.
        axes : iterable
            A list of axes for which to find the metric.

        Returns
        -------
        metric : xarray.DataArray
            A metric which can broadcast against ``array``
        """

        metric_vars = None
        array_dims = set(array.dims)

        # Will raise a Value Error if array doesn't have a dimension corresponding to metric axes specified
        # See _get_dims_from_axis
        self._get_dims_from_axis(array, frozenset(axes))

        possible_metric_vars = set(tuple(k) for k in self._metrics.keys())
        possible_combos = set(itertools.permutations(tuple(axes)))
        overlap_metrics = possible_metric_vars.intersection(possible_combos)

        if len(overlap_metrics) > 0:
            # Condition 1: metric with matching axes and dimensions exist
            overlap_metrics = frozenset(*overlap_metrics)
            possible_metrics = self._metrics[overlap_metrics]
            for mv in possible_metrics:
                metric_dims = set(mv.dims)
                if metric_dims.issubset(array_dims):
                    metric_vars = mv
                    break
            if metric_vars is None:
                # Condition 2: interpolate metric with matching axis to desired dimensions
                warnings.warn(
                    f"Metric at {array.dims} being interpolated from metrics at dimensions {mv.dims}. Boundary value set to 'extend'."
                )
                metric_vars = self.interp_like(mv, array, "extend", None)
        else:
            for axis_combinations in iterate_axis_combinations(axes):
                try:
                    # will raise KeyError if the axis combination is not in metrics
                    possible_metric_vars = [
                        self._metrics[ac] for ac in axis_combinations
                    ]
                    for possible_combinations in itertools.product(
                        *possible_metric_vars
                    ):
                        metric_dims = set(
                            [d for mv in possible_combinations for d in mv.dims]
                        )
                        if metric_dims.issubset(array_dims):
                            # Condition 3: use provided metrics with matching dimensions to calculate for required metric
                            metric_vars = possible_combinations
                            break
                        else:
                            # Condition 4: metrics in the wrong position (must interpolate before multiplying)
                            possible_dims = [pc.dims for pc in possible_combinations]
                            warnings.warn(
                                f"Metric at {array.dims} being interpolated from metrics at dimensions {possible_dims}. Boundary value set to 'extend'."
                            )
                            metric_vars = tuple(
                                self.interp_like(pc, array, "extend", None)
                                for pc in possible_combinations
                            )
                    if metric_vars is not None:
                        # return the product of the metrics
                        metric_vars = functools.reduce(operator.mul, metric_vars, 1)
                        break
                except KeyError:
                    pass
        if metric_vars is None:
            raise KeyError(
                f"Unable to find any combinations of metrics for array dims {array_dims!r} and axes {axes!r}"
            )
        return metric_vars

    @docstrings.dedent
    def interp_like(self, array, like, boundary=None, fill_value=None):
        """Compares positions between two data arrays and interpolates array to the position of like if necessary

        Parameters
        ----------
        array : DataArray
            DataArray to interpolate to the position of like
        like : DataArray
            DataArray with desired grid positions for source array
        boundary : str or dict, optional,
            boundary can either be one of {None, 'fill', 'extend', 'extrapolate'}
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition where
              the difference at the boundary will be zero.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            This sets the default value. It can be overriden by specifying the
            boundary kwarg when calling specific methods.
        fill_value : float, optional
            The value to use in the boundary condition when `boundary='fill'`.

        Returns
        -------
        array : DataArray
            Source data array with updated positions along axes matching with target array
        """

        interp_axes = []
        for axname, axis in self.axes.items():
            try:
                position_array, _ = axis._get_position_name(array)
                position_like, _ = axis._get_position_name(like)
            # This will raise a KeyError if you have multiple axes contained in self,
            # since the for-loop will go through all axes, but the method is applied for only 1 axis at a time
            # This is for cases where an axis is present in self that is not available for either array or like.
            # For the axis you are interested in interpolating, there should be data for it in grid, array, and like.
            except KeyError:
                continue
            if position_like != position_array:
                interp_axes.append(axname)

        array = self.interp(
            array,
            interp_axes,
            fill_value=fill_value,
            boundary=boundary,
        )
        return array

    def __repr__(self):
        summary = ["<xgcm.Grid>"]
        for name, axis in self.axes.items():
            is_periodic = "periodic" if axis._periodic else "not periodic"
            summary.append(
                "%s Axis (%s, boundary=%r):" % (name, is_periodic, axis.boundary)
            )
            summary += axis._coord_desc()
        return "\n".join(summary)

    @docstrings.get_sectionsf("grid_func", sections=["Parameters", "Examples"])
    @docstrings.dedent
    def _grid_func(self, funcname, da, axis, **kwargs):
        """this function calls appropriate functions from `Axis` objects.
        It handles multiple axis input and weighting with metrics

        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            If not specified, default will be used.
            Optionally a dict with seperate values for each axis can be passed (see example)
            Optionally a dict with seperate values for each axis can be passed (see example)
        boundary : None or str or dict, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)

            Optionally a dict with separate values for each axis can be passed (see example)
        fill_value : {float, dict}, optional
            The value to use in the boundary condition with `boundary='fill'`.
            Optionally a dict with seperate values for each axis can be passed (see example)
        vector_partner : dict, optional
            A single key (string), value (DataArray).
            Optionally a dict with seperate values for each axis can be passed (see example)
        metric_weighted : str or tuple of str or dict, optional
            If an axis or list of axes is specified,
            the appropriate grid metrics will be used to determined the weight for interpolation.
            E.g. if passing `metric_weighted=['X', 'Y']`, values will be weighted by horizontal area.
            If `False` (default), the points will be weighted equally.
            Optionally a dict with seperate values for each axis can be passed (see example)

        """

        if (not isinstance(axis, list)) and (not isinstance(axis, tuple)):
            axis = [axis]
        # parse multi axis kwargs like e.g. `boundary`
        multi_kwargs = {k: self._as_axis_kwarg_mapping(v) for k, v in kwargs.items()}

        out = da
        for axx in axis:
            kwargs = {k: v[axx] for k, v in multi_kwargs.items()}
            ax = self.axes[axx]
            kwargs.setdefault("boundary", ax.boundary)
            func = getattr(ax, funcname)
            metric_weighted = kwargs.pop("metric_weighted", False)

            if isinstance(metric_weighted, str):
                metric_weighted = (metric_weighted,)

            if metric_weighted:
                metric = self.get_metric(out, metric_weighted)
                out = out * metric

            out = func(out, **kwargs)

            if metric_weighted:
                metric_new = self.get_metric(out, metric_weighted)
                out = out / metric_new

        return out

    def pad(self, *arrays, boundary_width=None, boundary=None, fill_value=None):
        """
        Pads the boundary of given arrays along given Axes, according to information in Axes.boundary.

        Parameters
        ----------
        arrays : Sequence[xarray.DataArray]
            Arrays to pad according to boundary and boundary_width.
        boundary_width : Dict[str: Tuple[int, int]
            The widths of the boundaries at the edge of each array.
            Supplied in a mapping of the form {axis_name: (lower_width, upper_width)}.
        boundary : {None, 'fill', 'extend', 'extrapolate', dict}, optional
            A flag indicating how to handle boundaries:
            * None: Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            Optionally a dict mapping axis name to separate values for each axis
            can be passed.
        fill_value : {float, dict}, optional
            The value to use in boundary conditions with `boundary='fill'`.
            Optionally a dict mapping axis name to separate values for each axis
            can be passed. Default is 0.
        """
        # TODO accept a general padding function like numpy.pad does as an argument to boundary

        if not boundary_width:
            raise ValueError("Must provide the widths of the boundaries")

        if boundary and isinstance(boundary, str):
            boundary = {ax_name: boundary for ax_name in self.axes.keys()}
        if fill_value is None:
            fill_value = 0.0
        if isinstance(fill_value, float):
            fill_value = {ax_name: fill_value for ax_name in self.axes.keys()}
        # xgcm uses a default fill value of 0, while xarray uses nan.
        fill_value = {k: 0.0 if v is None else v for k, v in fill_value.items()}

        padded = []
        for da in arrays:
            new_da = da
            for ax, widths in boundary_width.items():
                axis = self.axes[ax]
                _, dim = axis._get_position_name(da)

                # Use default boundary for axis unless overridden
                if boundary:
                    ax_boundary = boundary[ax]
                else:
                    ax_boundary = axis.boundary

                if ax_boundary == "extrapolate":
                    # TODO implement extrapolation
                    raise NotImplementedError
                elif ax_boundary is None:
                    # TODO this is necessary, but also seems inconsistent with the docstring, which says that None = "no boundary condition"
                    ax_boundary = "periodic"

                # TODO avoid repeatedly calling xarray pad
                try:
                    mode = _XGCM_BOUNDARY_KWARG_TO_XARRAY_PAD_KWARG[ax_boundary]
                except KeyError:
                    raise ValueError(
                        f"{ax_boundary} is not a supported type of boundary"
                    )

                if mode == "constant":
                    new_da = new_da.pad(
                        {dim: widths}, mode, constant_values=fill_value[ax]
                    )
                else:
                    new_da = new_da.pad({dim: widths}, mode)

            padded.append(new_da)

        return padded

    def _1d_grid_ufunc_dispatch(
        self,
        funcname,
        da,
        axis,
        to=None,
        keep_coords=False,
        metric_weighted: Union[
            str, Iterable[str], Dict[str, Union[str, Iterable[str]]]
        ] = None,
        **kwargs,
    ):
        """
        Calls appropriate 1D grid ufuncs on data, along the specified axes, sequentially.

        Parameters
        ----------
        axis : str or list or tuple
            Name of the axis on which to act. Multiple axes can be passed as list or
            tuple (e.g. ``['X', 'Y']``). Functions will be executed over each axis in the
            given order.
        to : str or dict, optional
            The direction in which to shift the array (can be ['center','left','right','inner','outer']).
            Can be passed as a single str to use for all axis, or as a dict with separate values for each axis.
            If not specified, the `default_shifts` stored in each Axis object will be used for that axis.
        """

        if isinstance(axis, str):
            axis = [axis]

        # convert input arguments into axes-kwarg mappings
        to = self._as_axis_kwarg_mapping(to)

        if isinstance(metric_weighted, str):
            metric_weighted = (metric_weighted,)
        metric_weighted = self._as_axis_kwarg_mapping(metric_weighted)

        # If I understand correctly, this could contain some other kwargs, which are not appropriate to convert
        # Note that the option to pass boundary (actually padding) options on the method will be deprecated,
        # so for now I will just store everything I need here.
        VALID_BOUNDARY_KWARGS = ["boundary", "periodic", "fill_value"]
        kwargs = {
            k: self._as_axis_kwarg_mapping(v) if k in VALID_BOUNDARY_KWARGS else v
            for k, v in kwargs.items()
        }

        # Now check if some of the values not provided in the kwargs are set as grid properties
        # In the future grid object properties should be the only way to access that information
        # and we can remove this.
        kwargs_with_defaults = {}

        for k in set(
            list(kwargs.keys()) + VALID_BOUNDARY_KWARGS
        ):  # iterate over all axis properties and any additional keys of `kwarg`
            if k in VALID_BOUNDARY_KWARGS:
                defaults = getattr(self, k, {})
                kwargs_input = kwargs.get(k, {})
                # overwrite if given as input
                defaults.update(kwargs_input)
                kwargs_with_defaults[k] = defaults
            else:
                kwargs_with_defaults[k] = kwargs[k]
        kwargs = kwargs_with_defaults

        # Lastly go through the inputs axis by axis, and replace
        # boundary with 'periodic' if periodic is True
        # (again this is temporary until `periodic` is deprecated)
        if "periodic" in kwargs.keys():
            boundary = kwargs.get("boundary", {})
            new_boundary = {
                k: "periodic"
                for k in kwargs["periodic"].keys()
                if kwargs["periodic"].get(k, False)
            }
            boundary.update(new_boundary)
            kwargs["boundary"] = boundary
        # remove the periodic input
        kwargs = {k: v for k, v in kwargs.items() if k != "periodic"}

        signatures = self._create_1d_grid_ufunc_signatures(da, axis=axis, to=to)

        array = da
        # Apply 1D function over multiple axes
        # TODO This will call xarray.apply_ufunc once for each axis, but if signatures + kwargs are the same then we
        # TODO only actually need to call apply_ufunc once for those axes
        for signature_1d, ax_name in zip(signatures, axis):

            grid_ufunc, remaining_kwargs = _select_grid_ufunc(
                funcname, signature_1d, module=gridops, **kwargs
            )
            ax_metric_weighted = metric_weighted[ax_name]

            if ax_metric_weighted:
                metric = self.get_metric(array, ax_metric_weighted)
                array = array * metric

            array = grid_ufunc(
                self, array, axis=[ax_name], keep_coords=keep_coords, **remaining_kwargs
            )

            if ax_metric_weighted:
                metric = self.get_metric(array, ax_metric_weighted)
                array = array / metric

        return self._transpose_to_keep_same_dim_order(da, array, axis)

    def _create_1d_grid_ufunc_signatures(self, da, axis, to):
        """
        Create a list of signatures to pass to apply_grid_ufunc.

        Created from data, list of input axes, and list of target axis positions.
        One separate signature is created for each axis the 1D ufunc is going to be applied over.
        """
        signatures = []
        for ax_name in axis:
            ax = self.axes[ax_name]

            from_pos, _ = ax._get_position_name(da)  # removed `dim` since it wasnt used

            to_pos = to[ax_name]
            if to_pos is None:
                to_pos = ax._default_shifts[from_pos]

            signature_1d = f"({ax_name}:{from_pos})->({ax_name}:{to_pos})"
            signatures.append(signature_1d)

        return signatures

    def _transpose_to_keep_same_dim_order(self, da, result, axis):
        """Reorder DataArray dimensions to match the original input."""

        initial_dims = da.dims

        shifted_dims = {}
        for ax_name in axis:
            ax = self.axes[ax_name]

            _, old_dim = ax._get_position_name(da)
            _, new_dim = ax._get_position_name(result)
            shifted_dims[old_dim] = new_dim

        output_dims_but_in_original_order = [
            shifted_dims[dim] if dim in shifted_dims else dim for dim in initial_dims
        ]

        return result.transpose(*output_dims_but_in_original_order)

    def apply_as_grid_ufunc(
        self,
        func,
        *args,
        signature="",
        boundary_width=None,
        boundary=None,
        fill_value=None,
        dask="forbidden",
        **kwargs,
    ):
        """
        Apply a function to the given arguments in a grid-aware manner.

        The relationship between xgcm axes on the input and output are specified by
        `signature`. Wraps xarray.apply_ufunc, but determines the core dimensions
        from the grid and signature passed.

        Parameters
        ----------
        func : callable
            Function to call like `func(*args, **kwargs)` on numpy-like unlabeled
            arrays (`.data`).

            Passed directly on to `xarray.apply_ufunc`.
        signature : string
            Grid universal function signature. Specifies the xgcm.Axis names and
            positions for each input and output variable, e.g.,

            ``"(X:center)->(X:left)"`` for ``diff_center_to_left(a)`.
        boundary_width : Dict[str: Tuple[int, int]
            The widths of the boundaries at the edge of each array.
            Supplied in a mapping of the form {axis_name: (lower_width, upper_width)}.
        boundary : {None, 'fill', 'extend', 'extrapolate', dict}, optional
            A flag indicating how to handle boundaries:
            * None: Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            Optionally a dict mapping axis name to separate values for each axis
            can be passed.
        fill_value : {float, dict}, optional
            The value to use in boundary conditions with `boundary='fill'`.
            Optionally a dict mapping axis name to separate values for each axis
            can be passed. Default is 0.
        dask : {"forbidden", "allowed", "parallelized"}, default: "forbidden"
            How to handle applying to objects containing lazy data in the form of
            dask arrays. Passed directly on to `xarray.apply_ufunc`.

        Returns
        -------
        results
            The result of the call to `xarray.apply_ufunc`, but including the coordinates
            given by the signature, which are read from the grid. Output is either a single
            object or a tuple of such objects.

        See Also
        --------
        apply_as_grid_ufunc
        as_grid_ufunc
        """
        return apply_as_grid_ufunc(
            func,
            *args,
            grid=self,
            signature=signature,
            boundary_width=boundary_width,
            boundary=boundary,
            fill_value=fill_value,
            dask=dask,
            **kwargs,
        )

    @docstrings.dedent
    def interp(self, da, axis, **kwargs):
        """
        Interpolate neighboring points to the intermediate grid point along
        this axis.


        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

        Returns
        -------
        da_i : xarray.DataArray
            The interpolated data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if a global 2D dataset should be interpolated on both X and Y axis, but it is
        only periodic in the X axis, we can do this:

        >>> grid.interp(da, ["X", "Y"], periodic={"X": True, "Y": False})
        """
        return self._1d_grid_ufunc_dispatch("interp", da, axis, **kwargs)

    @docstrings.dedent
    def diff(self, da, axis, **kwargs):
        """
        Difference neighboring points to the intermediate grid point.

        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

        Returns
        -------
        da_i : xarray.DataArray
            The differenced data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if a global 2D dataset should be differenced on both X and Y axis, but the fill
        value at the boundary should be differenc for each axis, we can do this:

        >>> grid.diff(da, ["X", "Y"], fill_value={"X": 0, "Y": 100})
        """
        return self._1d_grid_ufunc_dispatch("diff", da, axis, **kwargs)

    @docstrings.dedent
    def min(self, da, axis, **kwargs):
        """
        Minimum of neighboring points on the intermediate grid point.

        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

        Returns
        -------
        da_i : xarray.DataArray
            The mimimum data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if we want to find the minimum of sourrounding grid cells for a global 2D dataset
        in both X and Y axis, but the fill value at the boundary should be different
        for each axis, we can do this:

        >>> grid.min(da, ["X", "Y"], fill_value={"X": 0, "Y": 100})
        """
        return self._1d_grid_ufunc_dispatch("min", da, axis, **kwargs)

    @docstrings.dedent
    def max(self, da, axis, **kwargs):
        """
        Maximum of neighboring points on the intermediate grid point.

        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

        Returns
        -------
        da_i : xarray.DataArray
            The maximum data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if we want to find the maximum of sourrounding grid cells for a global 2D dataset
        in both X and Y axis, but the fill value at the boundary should be different
        for each axis, we can do this:

        >>> grid.max(da, ["X", "Y"], fill_value={"X": 0, "Y": 100})
        """
        return self._1d_grid_ufunc_dispatch("max", da, axis, **kwargs)

    @docstrings.dedent
    def cumsum(self, da, axis, **kwargs):
        """
        Cumulatively sum a DataArray, transforming to the intermediate axis
        position.

        Parameters
        ----------
        %(grid_func.parameters)s

        Examples
        --------
        %(grid_func.examples)s

        Returns
        -------
        da_i : xarray.DataArray
            The cumsummed data

        Examples
        --------
        Each keyword argument can be provided as a `per-axis` dictionary. For instance,
        if we want to compute the cumulative sum of global 2D dataset
        in both X and Y axis, but the fill value at the boundary should be different
        for each axis, we can do this:

        >>> grid.max(da, ["X", "Y"], fill_value={"X": 0, "Y": 100})
        """
        return self._grid_func("cumsum", da, axis, **kwargs)

    @docstrings.dedent
    def _apply_vector_function(self, function, vector, **kwargs):
        # the keys, should be axis names
        assert len(vector) == 2

        # this is currently only tested for c-grid vectors defined on edges
        # moving to cell centers. We need to detect if we got something else
        to = kwargs.get("to", "center")
        if to != "center":
            raise NotImplementedError(
                "Only vector interpolation to cell "
                "center is implemented, but got "
                "to=%r" % to
            )
        for axis_name, component in vector.items():
            axis = self.axes[axis_name]
            position, coord = axis._get_position_name(component)
            if position == "center":
                raise NotImplementedError(
                    "Only vector interpolation to cell "
                    "center is implemented, but vector "
                    "%s component is defined at center "
                    "(dims: %r)" % (axis_name, component.dims)
                )

        x_axis_name, y_axis_name = list(vector)
        x_axis, y_axis = self.axes[x_axis_name], self.axes[y_axis_name]

        # apply for each component
        x_component = function(
            x_axis,
            vector[x_axis_name],
            vector_partner={y_axis_name: vector[y_axis_name]},
            **kwargs,
        )

        y_component = function(
            y_axis,
            vector[y_axis_name],
            vector_partner={x_axis_name: vector[x_axis_name]},
            **kwargs,
        )

        return {x_axis_name: x_component, y_axis_name: y_component}

    @docstrings.dedent
    def interp_2d_vector(self, vector, **kwargs):
        """
        Interpolate a 2D vector to the intermediate grid point. This method is
        only necessary for complex grid topologies.

        Parameters
        ----------
        vector : dict
            A dictionary with two entries. Keys are axis names, values are
            vector components along each axis.

        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        vector_interp : dict
            A dictionary with two entries. Keys are axis names, values
            are interpolated vector components along each axis
        """

        return self._apply_vector_function(Axis.interp, vector, **kwargs)

    @docstrings.dedent
    def derivative(self, da, axis, **kwargs):
        """
        Take the centered-difference derivative along specified axis.

        Parameters
        ----------
        axis : str
            Name of the axis on which to act
        %(grid_func.parameters)s

        Returns
        -------
        da_i : xarray.DataArray
            The differentiated data
        """

        ax = self.axes[axis]
        diff = ax.diff(da, **kwargs)
        dx = self.get_metric(diff, (axis,))
        return diff / dx

    @docstrings.dedent
    def integrate(self, da, axis, **kwargs):
        """
        Perform finite volume integration along specified axis or axes,
        accounting for grid metrics. (e.g. cell length, area, volume)

        Parameters
        ----------
        axis : str, list of str
            Name of the axis on which to act
        **kwargs: dict
            Additional arguments passed to `xarray.DataArray.sum`

        Returns
        -------
        da_i : xarray.DataArray
            The integrated data
        """

        weight = self.get_metric(da, axis)
        weighted = da * weight
        # TODO: We should integrate xarray.weighted once available.

        # get dimension(s) corresponding to `da` and `axis` input
        dim = self._get_dims_from_axis(da, axis)

        return weighted.sum(dim, **kwargs)

    @docstrings.dedent
    def cumint(self, da, axis, **kwargs):
        """
        Perform cumulative integral along specified axis or axes,
        accounting for grid metrics. (e.g. cell length, area, volume)

        Parameters
        ----------
        axis : str, list of str
            Name of the axis on which to act
        %(grid_func.parameters)s

        Returns
        -------
        da_i : xarray.DataArray
            The cumulatively integrated data
        """

        weight = self.get_metric(da, axis)
        weighted = da * weight
        # TODO: We should integrate xarray.weighted once available.

        return self.cumsum(weighted, axis, **kwargs)

    @docstrings.dedent
    def average(self, da, axis, **kwargs):
        """
        Perform weighted mean reduction along specified axis or axes,
        accounting for grid metrics. (e.g. cell length, area, volume)

        Parameters
        ----------
        axis : str, list of str
            Name of the axis on which to act
        **kwargs: dict
            Additional arguments passed to `xarray.DataArray.weighted.mean`

        Returns
        -------
        da_i : xarray.DataArray
            The averaged data
        """

        weight = self.get_metric(da, axis)
        weighted = da.weighted(weight)

        # get dimension(s) corresponding to `da` and `axis` input
        dim = self._get_dims_from_axis(da, axis)
        return weighted.mean(dim, **kwargs)

    def transform(self, da, axis, target, **kwargs):
        """Convert an array of data to new 1D-coordinates along `axis`.
        The method takes a multidimensional array of data `da` and
        transforms it onto another data_array `target_data` in the
        direction of the axis (for each 1-dimensional 'column').

        `target_data` can be e.g. the existing coordinate along an
        axis, like depth. xgcm automatically detects the appropriate
        coordinate and then transforms the data from the input
        positions to the desired positions defined in `target`. This
        is the default behavior. The method can also be used for more
        complex cases like transforming a dataarray into new
        coordinates that are defined by e.g. a tracer field like
        temperature, density, etc.

        Currently two methods are supported to carry out the
        transformation:

        - 'linear': Values are linear interpolated between 1D columns
          along `axis` of `da` and `target_data`. This method requires
          `target_data` to increase/decrease monotonically. `target`
          values are interpreted as new cell centers in this case. By
          default this method will return nan for values in `target` that
          are outside of the range of `target_data`, setting
          `mask_edges=False` results in the default np.interp behavior of
          repeated values.

        - 'conservative': Values are transformed while conserving the
          integral of `da` along each 1D column. This method can be used
          with non-monotonic values of `target_data`. Currently this will
          only work with extensive quantities (like heat, mass, transport)
          but not with intensive quantities (like temperature, density,
          velocity). N given `target` values are interpreted as cell-bounds
          and the returned array will have N-1 elements along the newly
          created coordinate, with coordinate values that are interpolated
          between `target` values.

        Parameters
        ----------
        da : xr.DataArray
            Input data
        axis : str
            Name of the axis on which to act
        target : {np.array, xr.DataArray}
            Target points for transformation. Dependin on the method is
            interpreted as cell center (method='linear') or cell bounds
            (method='conservative).
            Values correpond to `target_data` or the existing coordinate
            along the axis (if `target_data=None`). The name of the
            resulting new coordinate is determined by the input type.
            When passed as numpy array the resulting dimension is named
            according to `target_data`, if provided as xr.Dataarray
            naming is inferred from the `target` input.
        target_data : xr.DataArray, optional
            Data to transform onto (e.g. a tracer like density or temperature).
            Defaults to None, which infers the appropriate coordinate along
            `axis` (e.g. the depth).
        method : str, optional
            Method used to transform, by default "linear"
        mask_edges : bool, optional
            If activated, `target` values outside the range of `target_data`
            are masked with nan, by default True. Only applies to 'linear' method.
        bypass_checks : bool, optional
            Only applies for `method='linear'`.
            Option to bypass logic to flip data if monotonically decreasing along the axis.
            This will improve performance if True, but the user needs to ensure that values
            are increasing alon the axis.
        suffix : str, optional
            Customizable suffix to the name of the output array. This will
            be added to the original name of `da`. Defaults to `_transformed`.

        Returns
        -------
        xr.DataArray
            The transformed data


        """

        ax = self.axes[axis]
        return ax.transform(da, target, **kwargs)

    @docstrings.dedent
    def diff_2d_vector(self, vector, **kwargs):
        """
        Difference a 2D vector to the intermediate grid point. This method is
        only necessary for complex grid topologies.

        Parameters
        ----------
        vector : dict
            A dictionary with two entries. Keys are axis names, values are
            vector components along each axis.

        %(neighbor_binary_func.parameters.no_f)s

        Returns
        -------
        vector_diff : dict
            A dictionary with two entries. Keys are axis names, values
            are differenced vector components along each axis
        """

        return self._apply_vector_function(Axis.diff, vector, **kwargs)


def _select_grid_ufunc(funcname, signature, module, **kwargs):
    # TODO to select via other kwargs (e.g. boundary) the signature of this function needs to be generalised

    def is_grid_ufunc(obj):
        return isinstance(obj, GridUFunc)

    # This avoids defining a list of functions in gridops.py
    all_predefined_ufuncs = inspect.getmembers(module, is_grid_ufunc)

    name_matching_ufuncs = [
        f for name, f in all_predefined_ufuncs if name.startswith(funcname)
    ]
    if len(name_matching_ufuncs) == 0:
        raise NotImplementedError(
            f"Could not find any pre-defined {funcname} grid ufuncs"
        )

    signature_matching_ufuncs = [
        f
        for f in name_matching_ufuncs
        if _signatures_equivalent(f.signature, signature)
    ]
    if len(signature_matching_ufuncs) == 0:
        raise NotImplementedError(
            f"Could not find any pre-defined {funcname} grid ufuncs with signature {signature}"
        )

    matching_ufuncs = signature_matching_ufuncs

    # TODO select via any other kwargs (such as boundary? metrics?) once implemented
    all_kwargs = kwargs.copy()
    # boundary = kwargs.pop("boundary", None)
    # if boundary:
    #    matching_ufuncs = [uf for uf in matching_ufuncs if uf.boundary == boundary]

    if len(matching_ufuncs) > 1:
        # TODO include kwargs used to match in this error message
        raise ValueError(
            f"Function {funcname} with signature='{signature}' and kwargs={all_kwargs} is an ambiguous selection"
        )
    elif len(matching_ufuncs) == 0:
        raise NotImplementedError(
            f"Could not find any pre-defined {funcname} grid ufuncs with signature='{signature}' and kwargs"
            f"={all_kwargs}"
        )
    else:
        # Exactly 1 matching function
        return matching_ufuncs[0], kwargs


def raw_interp_function(data_left, data_right):
    # linear, centered interpolation
    # TODO: generalize to higher order interpolation
    return 0.5 * (data_left + data_right)


def raw_diff_function(data_left, data_right):
    return data_right - data_left


def raw_min_function(data_left, data_right):
    return np.minimum(data_right, data_left)


def raw_max_function(data_left, data_right):
    return np.maximum(data_right, data_left)


_other_docstring_options = """
    * 'dirichlet'
       The value of the array at the boundary point is specified by
       `fill_value`.
    * 'neumann'
       The value of the array diff at the boundary point is
       specified[1]_ by `fill_value`.

        .. [1] https://en.wikipedia.org/wiki/Dirichlet_boundary_condition
        .. [2] https://en.wikipedia.org/wiki/Neumann_boundary_condition
"""
