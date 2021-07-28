"""
Classes and functions for 1D coordinate transformation.
"""
import functools

import numpy as np
import xarray as xr
from numba import boolean, float32, float64, guvectorize

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
    phi, theta, target_theta_levels, mask_edges=False, bypass_checks=False
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

    Returns
    -------
    phi_interp : array
        Array of shape (..., m) of phi interpolated to theta isosurfaces.
    """
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
        # Since we are introducing a new dimension instead of changing it we need to declare the output size.
        output_sizes={"remapped": len(target_theta_levels) - 1},
        output_dtypes=[phi.dtype],
    ).rename({"remapped": target_dim})

    # assign the target cell center
    target_centers = (target_theta_levels.data[1:] + target_theta_levels.data[:-1]) / 2
    out = out.assign_coords({target_dim: target_centers})

    # TODO: Somehow preserve the original bounds

    return out
