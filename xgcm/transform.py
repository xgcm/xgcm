"""
Classes and functions for 1D coordinate transformation.
"""

import functools
import warnings

import numpy as np
import xarray as xr
from numba import boolean, float32, float64, guvectorize  # type: ignore

"""Low level functions (numba/numpy)"""


@guvectorize(
    [
        (float64[:], float64[:], float64[:], boolean, boolean, float64[:]),
        (float32[:], float32[:], float32[:], boolean, boolean, float32[:]),
    ],
    "(n),(n),(m),(),()->(m)",
    nopython=True,
)
def _interp_1d_linear(
    phi, theta, target_theta_levels, mask_edges, bypass_checks, output
):
    # rough check if the data is decreasing with depth. If that is the case, flip.
    if not bypass_checks:
        theta_sign_test = theta[~np.isnan(theta)]
        if theta_sign_test[-1] < theta_sign_test[0]:
            theta = theta[::-1]
            phi = phi[::-1]

    output[:] = np.interp(target_theta_levels, theta, phi)

    if mask_edges:
        theta_max = np.nanmax(theta)
        theta_min = np.nanmin(theta)
        for i in range(len(target_theta_levels)):
            theta_lev = target_theta_levels[i]
            if (theta_lev < theta_min) or (theta_lev > theta_max):
                output[i] = np.nan


def interp_1d_linear(
    phi,
    theta,
    target_theta_levels,
    mask_edges=False,
    bypass_checks=False,
    logarithmic=False,
):
    """
    Vectorized interpolation of scalar phi to isosurfaces of scalar theta
    along the final axis.

    Parameters
    ----------
    phi : array_like
        Array of shape (..., n), scalar field to be interpolated
    theta : array_like
        Array of shape (..., n), scalar field which defines the isosurfaces
    target_theta_levels : array_like
        Array of shape (m) specificying target isosurface levels
    mask_edges : bool, optional
        Determines how to handle theta values that exceed the bounds of
        target_theta_levels. If False, fill with nearest valid values. If
        True, fill with NaNs.
    bypass_checks : bool, optional
        Option to bypass logic to flip data if monotonically decreasing along the axis.
        This will improve performance if True, but the user needs to ensure that values
        are increasing alon the axis.
    logarithmic : bool, optional
        Apply a logarithmic transform to theta and target_theta_levels so that the
        interpolation is done in logarithmic, rather than linear, space. In turn, theta
        must be positive. Defaults to False.

    Returns
    -------
    phi_interp : array
        Array of shape (..., m) of phi interpolated to theta isosurfaces.
    """
    if logarithmic:
        theta = np.log(theta)
        target_theta_levels = np.log(target_theta_levels)
    return _interp_1d_linear(phi, theta, target_theta_levels, mask_edges, bypass_checks)


@guvectorize(
    [
        (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]),
        (float32[:], float32[:], float32[:], float32[:], float32[:], float32[:]),
    ],
    "(n),(n),(n),(m),(m)->(m)",
    nopython=True,
)
def _interp_1d_conservative(phi, theta_1, theta_2, theta_hat_1, theta_hat_2, output):
    output[:] = 0

    n = len(theta_1)
    m = len(theta_hat_1)

    for i in range(n):
        # handle missing values
        if np.isnan(theta_1[i]) and np.isnan(theta_2[i]):
            continue
        # in the next two cases, we are effectively applying a boundary condition
        # by assuming that theta is homogenous over the cell
        elif np.isnan(theta_1[i]):
            theta_min = theta_max = theta_2[i]
        elif np.isnan(theta_2[i]):
            theta_min = theta_max = theta_1[i]
        # handle non-monotonic stratification
        elif theta_1[i] < theta_2[i]:
            theta_min = theta_1[i]
            theta_max = theta_2[i]
        else:
            theta_min = theta_2[i]
            theta_max = theta_1[i]

        for j in range(m):
            if (theta_hat_1[j] > theta_max) or (theta_hat_2[j] < theta_min):
                # there is no overlap between the cell and the bin
                pass
            elif theta_max == theta_min:
                output[j] += phi[i]
            else:
                # from here on there is some overlap
                theta_hat_min = max(theta_min, theta_hat_1[j])
                theta_hat_max = min(theta_max, theta_hat_2[j])
                alpha = (theta_hat_max - theta_hat_min) / (theta_max - theta_min)
                # now assign based on this weight
                output[j] += alpha * phi[i]


def interp_1d_conservative(phi, theta, target_theta_bins):
    """
    Accumulate extensive cell-centered quantity phi into new vertical coordinate
    defined by scalar theta.

    Parameters
    ----------
    phi : array_like
        Array of shape (..., n) defining an extensive quanitity in a cell
        bounded by two vertices.
    theta : array_like
        Array of shape (..., n+1) giving values of scalar theta  on the
        cell vertices. Phi is assumed to vary linearly between vertices.
    target_theta_bins : array_like
        Array of shape (m) defining the bounds of bins in which to accumulate
        phi.

    Returns
    -------
    phi_accum : array_like
        Array of shape (..., m-1) giving the values of phi partitioned into
        specified theta bins.
    """

    assert phi.shape[-1] == (theta.shape[-1] - 1)
    assert target_theta_bins.ndim == 1

    # flip target_theta_bins if needed (only needed for the conservative method,
    # np.interp handles this by itself)
    target_diff = np.diff(target_theta_bins)
    if all(target_diff < 0):
        flip_switch = True
        target_theta_bins = target_theta_bins[::-1]
    elif all(target_diff > 0):
        flip_switch = False
    else:
        raise ValueError("Target values are not monotonic")

    theta_1 = theta[..., :-1]
    theta_2 = theta[..., 1:]
    theta_hat_1 = target_theta_bins[:-1]
    theta_hat_2 = target_theta_bins[1:]

    out = _interp_1d_conservative(phi, theta_1, theta_2, theta_hat_1, theta_hat_2)
    if flip_switch:
        out = out[::-1]
    return out


"""Mid level functions (xarray)"""


def input_handling(func):
    """Decorator that handles input naming for interpolations."""

    @functools.wraps(func)
    def wrapper_input_handling(*args, **kwargs):
        # unpack args
        phi, theta, target_theta_levels, phi_dim, theta_dim, target_dim = args

        # pop kwargs used for naming
        suffix = kwargs.pop("suffix", "")

        # rename all input dims to unique names to avoid conflicts in xr.apply_ufunc
        temp_dim = "temp_dim_target"
        target_theta_levels = target_theta_levels.rename({target_dim: temp_dim})

        # The phi_dim doesnt matter for the final product, so just rename to
        # # something unique to avoid conflicts in apply_ufunc
        temp_dim2 = "temp_unique"
        phi = phi.rename({phi_dim: temp_dim2})

        # Execute function with temporary names
        args = (phi, theta, target_theta_levels, temp_dim2, theta_dim, temp_dim)
        value = func(*args, **kwargs)

        # rename back to original name
        value = value.rename({temp_dim: target_dim})

        # name the output according to input name and user customizable suffix
        if phi.name:
            value.name = phi.name + suffix

        return value

    return wrapper_input_handling


@input_handling
def linear_interpolation(
    phi, theta, target_theta_levels, phi_dim, theta_dim, target_dim, **kwargs
):
    out = xr.apply_ufunc(
        interp_1d_linear,
        phi,
        theta,
        target_theta_levels,
        kwargs=kwargs,
        input_core_dims=[[phi_dim], [theta_dim], [target_dim]],
        output_core_dims=[[target_dim]],
        exclude_dims=set((phi_dim, theta_dim)),
        dask="parallelized",
        output_dtypes=[phi.dtype],
    )
    return out


@input_handling
def conservative_interpolation(
    phi, theta, target_theta_levels, phi_dim, theta_dim, target_dim, **kwargs
):
    out = xr.apply_ufunc(
        interp_1d_conservative,
        phi,
        theta,
        target_theta_levels,
        kwargs=kwargs,
        input_core_dims=[[phi_dim], [theta_dim], [target_dim]],
        output_core_dims=[["remapped"]],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"remapped": len(target_theta_levels) - 1}},
        # Since we are introducing a new dimension instead of changing it we need to declare the output size.
        output_dtypes=[phi.dtype],
    ).rename({"remapped": target_dim})

    # assign the target cell center
    target_centers = (target_theta_levels.data[1:] + target_theta_levels.data[:-1]) / 2
    out = out.assign_coords({target_dim: target_centers})

    # TODO: Somehow preserve the original bounds

    return out


def transform(
    grid,
    axis_name,
    da,
    target,
    target_data=None,
    target_dim=None,
    method="linear",
    mask_edges=True,
    bypass_checks=False,
    suffix="_transformed",
):
    """Convert an array of data to new 1D-coordinates.
    The method takes a multidimensional array of data `da` and
    transforms it onto another data_array `target_data` in the
    direction of the axis (for each 1-dimensional 'column').
    `target_data` can be e.g. the existing coordinate along an
    axis, like depth. xgcm automatically detects the appropriate
    coordinate and then transforms the data from the input
    positions to the desired positions defined in `target`. This
    is the default behavior. If the `target` has more than one
    dimension, the `target_dim` has to be explicitly set. Otherwise,
    it is inferred from `target`. The method can also be used for more
    complex cases like transforming a dataarray into new
    coordinates that are defined by e.g. a tracer field like
    temperature, density, etc.
    Currently three methods are supported to carry out the
    transformation:
    - 'linear': Values are linear interpolated between 1D columns
      along `axis` of `da` and `target_data`. This method requires
      `target_data` to increase/decrease monotonically. `target`
      values are interpreted as new cell centers in this case. By
      default this method will return nan for values in `target` that
      are outside of the range of `target_data`, setting
      `mask_edges=False` results in the default np.interp behavior of
      repeated values.
    - 'log': Same as 'linear', but with values interpolated
      logarithmically between 1D columns. Operates by applying `np.log`
      to the target and target data values prior to linear interpolation.
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
    grid : xgcm.Grid
        xgcm Grid object
    axis_name : str
        Name of the axis along which to operate
    da : xr.DataArray
        Input data
    target : {np.array, xr.DataArray}
        Target points for transformation. Depending on the method is
        interpreted as cell center (method='linear' and method='log') or
        cell bounds (method='conservative).
        Values correspond to `target_data` or the existing coordinate
        along the axis (if `target_data=None`). 
    target_data : xr.DataArray, optional
        Data to transform onto (e.g. a tracer like density or temperature).
        Defaults to None, which infers the appropriate coordinate along
        `axis` (e.g. the depth).
    target_dim : str, optional
        Dimension name associated with the `target` points. If `target` has more than one dimension, this 
        parameter must be explicitly set to specify which dimension corresponds to the transformation. If `target` 
        is one-dimensional, `target_dim` is inferred automatically.
    method : str, optional
        Method used to transform, by default "linear"
    mask_edges : bool, optional
        If activated, `target` values outside the range of `target_data`
        are masked with nan, by default True. Only applies to 'linear' and 'log'
        methods.
    bypass_checks : bool, optional
        Only applies for `method='linear'` and `method='log'`.
        Option to bypass logic to flip data if monotonically decreasing along the axis.
        This will improve performance if True, but the user needs to ensure that values
        are increasing along the axis.
    suffix : str, optional
        Customizable suffix to the name of the output array. This will
        be added to the original name of `da`. Defaults to `_transformed`.

    Returns
    -------
    xr.DataArray
        The transformed data

    Notes
    -----
    - If `target` is multi-dimensional, you must specify `target_dim` explicitly.
    - If `target_dim` is not specified and `target_data` has no name, a default name "TRANSFORMED_DIMENSION" will be used for the transformed dimension.
    - For `conservative` transformations, `target_data` must be located on cell bounds.

    """

    axis = grid.axes[axis_name]

    # raise error if axis is periodic
    if axis.boundary == "periodic":
        raise ValueError(
            "`transform` can only be used on axes that are non-periodic. Pass `periodic=False` to `xgcm.Grid`."
        )

    # raise error if the target values are not provided as xr.dataarray
    for var_name, variable, allowed_types in [
        ("da", da, [xr.DataArray]),
        ("target", target, [xr.DataArray, np.ndarray]),
        ("target_data", target_data, [xr.DataArray]),
    ]:
        if not (isinstance(variable, tuple(allowed_types)) or variable is None):
            raise ValueError(
                f"`{var_name}` needs to be a {' or '.join([str(a) for a in allowed_types])}. Found {type(variable)}"
            )

    def _target_data_name_handling(target_data):
        """Handle target_data input without a name"""
        if target_data.name is None:
            warnings.warn(
                "Input`target_data` has no name, but we need a name for the transformed dimension. The name `TRANSFORMED_DIMENSION` will be used. To avoid this warning, call `.rename` on `target_data` before calling `transform`."
            )
            target_data.name = "TRANSFORMED_DIMENSION"

    def _check_other_dims(target_da):
        # check if other dimensions (excluding ones associated with the transform axis) are the
        # same between `da` and `target_data`. If not provide instructions how to work around.

        da_other_dims = set(da.dims) - set(axis.coords.values())
        target_da_other_dims = set(target_da.dims) - set(axis.coords.values())
        if not target_da_other_dims.issubset(da_other_dims):
            raise ValueError(
                f"Found additional dimensions [{target_da_other_dims - da_other_dims}]"
                "in `target_data` not found in `da`. This could mean that the target "
                "array is not on the same position along other axes."
                " If the additional dimensions are associated witha staggered axis, "
                "use grid.interp() to move values to other grid position. "
                "If additional dimensions are not related to the grid (e.g. climate "
                "model ensemble members or similar), use xr.broadcast() before using transform."
            )

    def _parse_target(target, target_dim, target_data_dim, target_data):
        """Parse target values into correct xarray naming and set default naming based on input data"""
        # if target_data is not provided, assume the target to be one of the staggered dataset dimensions.
        if target_data is None:
            target_data = grid._ds[target_data_dim]

        if target_dim is None:
            # Infer target_dim from target
            if isinstance(target, xr.DataArray):
                if len(target.dims) == 1:
                    if target_dim is None:
                        target_dim = list(target.dims)[0]
                else:
                    if target_dim is not None and target_dim not in target.dims:
                        raise ValueError(
                                f"The specified `target_dim` {target_dim} is not within the dimensions of the target: [{target.dims}]."
                        )
            else:
                # if the target is not provided as xr.Dataarray we take the name of the target_data as new dimension name
                _target_data_name_handling(target_data)
                target_dim = target_data.name
        if not isinstance(target, xr.DataArray):
            target = xr.DataArray(
                target, dims=[target_dim], coords={target_dim: target}
            )

        _check_other_dims(target_data)
        return target, target_dim, target_data

    _, dim = axis._get_position_name(da)
    if method == "linear" or method == "log":
        target, target_dim, target_data = _parse_target(
            target, target_dim, dim, target_data
        )
        out = linear_interpolation(
            da,
            target_data,
            target,
            dim,
            dim,  # in this case the dimension of phi and theta are the same
            target_dim,
            mask_edges=mask_edges,
            bypass_checks=bypass_checks,
            logarithmic=(method == "log"),
        )
    elif method == "conservative":
        # the conservative method requires `target_data` to be on the `outer` coordinate.
        # If that is not the case (a very common use case like transformation on any tracer),
        # we need to infer the boundary values (using the interp logic)
        # for this method we need the `outer` position. Error out if its not there.
        try:
            target_data_dim = axis.coords["outer"]
        except KeyError:
            raise RuntimeError(
                "In order to use the method `conservative` the grid object needs to have `outer` coordinates."
            )

        target, target_dim, target_data = _parse_target(
            target, target_dim, target_data_dim, target_data
        )

        # check on which coordinate `target_data` is, and interpolate if needed
        if target_data_dim not in target_data.dims:
            warnings.warn(
                "The `target data` input is not located on the cell bounds. This method will continue with linear interpolation with repeated boundary values. For most accurate results provide values on cell bounds.",
                UserWarning,
            )
            target_data = grid.interp(target_data, axis_name, boundary="extend")
            # This seems to end up with chunks along the axis dimension.
            # Rechunk to keep xr.apply_func from complaining.
            # TODO: This should be made obsolete, when the internals are refactored using numba
            target_data = target_data.chunk(
                {axis._get_position_name(target_data)[1]: -1}
            )

        out = conservative_interpolation(
            da,
            target_data,
            target,
            dim,
            target_data_dim,  # in this case the dimension of phi and theta are the same
            target_dim,
        )

    return out
