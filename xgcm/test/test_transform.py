"""Tests for low transform logic."""

import pytest

import numpy as np
import xarray as xr
from ..transform import (
    interp_1d_linear,
    interp_1d_conservative,
    linear_interpolation,
    conservative_interpolation,
)

from xgcm.grid import Grid, Axis

"""Tests for low level routines."""


def test_interp_1d_linear():
    nz, nx = 100, 1000
    z_vertex = np.linspace(0, 1, nz + 1)
    z = 0.5 * (z_vertex[:-1] + z_vertex[1:])
    x = 2 * np.pi * np.linspace(0, 1, nx)
    # uniformly stratified scalar
    theta = z + 0.1 * np.cos(3 * x)[:, None]
    # the scalar to interpolate
    phi = np.sin(theta) + 0.1 * np.cos(5 * x)[:, None]
    target_theta_levels = np.arange(0.2, 0.9, 0.025)
    phi_at_theta_expected = np.sin(target_theta_levels) + 0.1 * np.cos(5 * x)[:, None]

    # the action
    phi_at_theta = interp_1d_linear(phi, theta, target_theta_levels, mask_edges=False)
    np.testing.assert_allclose(phi_at_theta, phi_at_theta_expected, rtol=1e-4)


def test_interp_1d_conservative():
    nz = 30
    k = np.arange(nz)
    dz = 10 + np.linspace(0, 90, nz - 1)
    z = np.concatenate([[0], np.cumsum(dz)])
    H = z.max()
    theta = z / H + 0.2 * np.cos(np.pi * z / H)
    # phi = np.sin(5 * np.pi * z/H)

    nbins = 100
    theta_bins = np.linspace(theta.min() - 0.1, theta.max() + 0.1, nbins)

    # lazy way to check that it vectorizes: just copy the 1d array
    nx = 5
    dz_2d = np.tile(dz, (nx, 1))
    theta_2d = np.tile(theta, (nx, 1))

    dz_theta = interp_1d_conservative(dz_2d, theta_2d, theta_bins)

    np.testing.assert_allclose(dz_theta.sum(axis=-1), dz.sum(axis=-1))


"""1D Test datasets for various transformations."""

# TODO: I am not sure how we would handle periodic axes atm. Should we raise an error?
def raw_datasets():
    z = np.array([5, 25, 60])
    z_outer = np.array([0, 10, 50, 75])

    #####
    # First set of examples: Linear interpolation on different depth levels
    target_z = np.array([0, 7, 30, 60, 70])
    data = np.random.rand(len(z))

    data_expected = np.interp(target_z, z, data)

    input_linear_depth_depth = xr.Dataset(
        {"data": xr.DataArray(data, dims=["z"], coords={"z": z})}
    )
    expected_linear_depth_depth = xr.DataArray(
        data_expected, dims=["z"], coords={"z": target_z}
    )

    # same with masked outcrops
    data_expected_masked_outcrops = data_expected.copy()
    data_expected_masked_outcrops[0] = np.nan
    data_expected_masked_outcrops[-1] = np.nan
    expected_linear_depth_depth_masked_outcrops = xr.DataArray(
        data_expected_masked_outcrops, dims=["z"], coords={"z": target_z}
    )
    #####
    # Second set of examples: Conservative remapping to other depth levels.
    data = np.array([1, 4, 0])
    target_z_bounds = np.array([0, 1, 10, 50, 80])
    input_conservative_depth_depth = xr.Dataset(
        {"data": xr.DataArray(data, dims=["z"], coords={"z": z})}
    )
    input_conservative_depth_depth = input_conservative_depth_depth.assign_coords(
        {"zc": xr.DataArray(z_outer, dims=["zc"], coords={"zc": z_outer})}
    )
    # TODO: reimplement the weighted matrix from xrutils for testing
    # TODO: TEST for conservation...
    # for now just take the recorded result (this is really cheating.)
    expected_conservative_depth_depth = xr.DataArray(
        np.array([0.1, 0.9, 4.0, 0.0]),
        dims=["zc"],
        coords={
            "zc": (target_z_bounds[1:] + target_z_bounds[0:-1]) / 2
        },  # simply infer the cell center
    )

    datasets = {
        "linear_depth_depth": (
            input_linear_depth_depth,
            {"coords": {"Z": {"center": "z"}}},
            target_z,
            {"mask_edges": False, "method": "linear"},
            expected_linear_depth_depth,
        ),
        "linear_depth_depth_renamed": (
            input_linear_depth_depth.rename({"z": "test"}),
            {"coords": {"Z": {"center": "test"}}},
            target_z,
            {"mask_edges": True, "method": "linear"},
            expected_linear_depth_depth_masked_outcrops.rename({"z": "test"}),
        ),
        "linear_depth_depth_masked_outcrops": (
            input_linear_depth_depth,
            {"coords": {"Z": {"center": "z"}}},
            target_z,
            {"mask_edges": True, "method": "linear"},
            expected_linear_depth_depth_masked_outcrops,
        ),
        "conservative_depth_depth": (
            input_conservative_depth_depth,
            {"coords": {"Z": {"inner": "z", "outer": "zc"}}},
            target_z_bounds,
            {"method": "conservative"},
            expected_conservative_depth_depth,
        ),
    }
    return datasets


@pytest.fixture(
    scope="module",
    params=[
        "linear_depth_depth",
        "linear_depth_depth_masked_outcrops",
        "linear_depth_depth_renamed",
        "conservative_depth_depth",
    ],
)
def dataset(request):
    input, grid_kwargs, target, transform_kwargs, expected = raw_datasets()[
        request.param
    ]
    return input, grid_kwargs, target, transform_kwargs, expected


"""Test suite."""


def test_linear_interpolation_target_value_error():
    """Test that linear_interpolation throws an error when `target` is a np array"""
    input, grid_kwargs, target, transform_kwargs, expected = raw_datasets()[
        "linear_depth_depth"
    ]
    # method keyword is only for high level tests
    transform_kwargs.pop("method")
    with pytest.raises(ValueError):
        interpolated = linear_interpolation(input.data, input["z"], target, "z", "z")
    # TODO: test with the other method


# Test the linear interpolations only
@pytest.mark.parametrize(
    "name",
    [
        "linear_depth_depth",
        "linear_depth_depth_masked_outcrops",
        "linear_depth_depth_renamed",
    ],
)
def test_low_level_linear(name):
    input, grid_kwargs, target, transform_kwargs, expected = raw_datasets()[name]
    dim = grid_kwargs["coords"]["Z"]["center"]
    da_target = xr.DataArray(target, dims=[dim], coords={dim: target})

    # method keyword is only for high level tests
    transform_kwargs.pop("method")

    interpolated = linear_interpolation(
        input.data, input[dim], da_target, dim, dim, dim, **transform_kwargs
    )
    xr.testing.assert_allclose(interpolated, expected)


# Test the conservative interpolations only
@pytest.mark.parametrize(
    "name",
    [
        "conservative_depth_depth",
    ],
)
def test_low_level_conservative(name):
    input, grid_kwargs, target, transform_kwargs, expected = raw_datasets()[name]
    dim = grid_kwargs["coords"]["Z"]["inner"]
    target_dim = grid_kwargs["coords"]["Z"]["outer"]
    da_target = xr.DataArray(target, dims=[target_dim], coords={target_dim: target})

    # method keyword is only for high level tests
    transform_kwargs.pop("method")

    interpolated = conservative_interpolation(
        input.data,
        input[target_dim],
        da_target,
        dim,
        target_dim,
        target_dim,
        **transform_kwargs
    )
    xr.testing.assert_allclose(interpolated, expected)


def test_grid_transform(dataset):
    input, grid_kwargs, target, transform_kwargs, expected = dataset

    axis = list(grid_kwargs["coords"].keys())[0]
    dim = grid_kwargs["coords"][axis]["center"]
    da_target = xr.DataArray(target, dims=[dim], coords={dim: target})

    grid = Grid(input, **grid_kwargs)
    transformed = grid.transform(input.data, axis, da_target, **transform_kwargs)
    xr.testing.assert_allclose(transformed, expected)
