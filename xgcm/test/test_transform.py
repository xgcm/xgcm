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


"""1D Test datasets for various transformations.
This part of the module is organized as follows:

1. A nested dictionary with all parameters needed to construct a set of test datasets
2. Fixtures that use a constructor function and all or a subset of the dictionary entries from 1. to returns:
     (input_dataset, grid_kwargs, target, transform_kwargs, expected_dataarray)
3. Test functions

"""
cases = {
    "linear_depth_depth": {
        "input_coord": ("z", [5, 25, 60]),
        "input_data": ("data", [0.23246861, 0.45175654, 0.58320681]),  # random numbers
        "target_coord": ("z", [0, 7, 30, 60, 70]),
        "target_data": ("z", [0, 7, 30, 60, 70]),
        "expected_coord": ("z", [0, 7, 30, 60, 70]),
        "expected_data": (
            "data",
            np.interp(
                [0, 7, 30, 60, 70], [5, 25, 60], [0.23246861, 0.45175654, 0.58320681]
            ),
        ),  # same input as in `input_data`
        "grid_kwargs": {"coords": {"Z": {"center": "z"}}},
        "transform_kwargs": {"mask_edges": False, "method": "linear"},
    },
    "linear_depth_depth_masked_outcrops": {
        "input_coord": ("z", [5, 25, 60]),
        "input_data": ("data", [0.23246861, 0.45175654, 0.58320681]),  # random numbers
        "target_coord": ("z", [0, 7, 30, 60, 70]),
        "target_data": ("z", [0, 7, 30, 60, 70]),
        "expected_coord": ("z", [0, 7, 30, 60, 70]),
        "expected_data": (
            "data",
            np.interp(
                [0, 7, 30, 60, 70], [5, 25, 60], [0.23246861, 0.45175654, 0.58320681]
            ),
        ),  # same input as in `input_data`
        "expected_data_mask_index": [0, -1],
        "expected_data_mask_value": np.nan,
        "grid_kwargs": {"coords": {"Z": {"center": "z"}}},
        "transform_kwargs": {"mask_edges": True, "method": "linear"},
    },
    "linear_depth_depth_renamed": {
        "input_coord": ("test", [5, 25, 60]),
        "input_data": ("data", [0.23246861, 0.45175654, 0.58320681]),  # random numbers
        "target_coord": ("something", [0, 7, 30, 60, 70]),
        "target_data": ("something", [0, 7, 30, 60, 70]),
        "expected_coord": ("something", [0, 7, 30, 60, 70]),
        "expected_data": (
            "data",
            np.interp(
                [0, 7, 30, 60, 70], [5, 25, 60], [0.23246861, 0.45175654, 0.58320681]
            ),
        ),  # same input as in `input_data`
        "grid_kwargs": {"coords": {"Z": {"center": "test"}}},
        "transform_kwargs": {"mask_edges": False, "method": "linear"},
    },
    "conservative_depth_depth": {
        "input_coord": ("z", [5, 25, 60]),
        "input_bounds_coord": ("zc", [0, 10, 50, 75]),
        "input_data": ("data", [1, 4, 0]),
        "target_coord": ("z", [0, 1, 10, 50, 80]),
        "target_data": ("z", [0, 1, 10, 50, 80]),
        "expected_coord": ("z", [0.5, 5.5, 30, 65]),
        "expected_data": (
            "data",
            [0.1, 0.9, 4.0, 0.0],
        ),  # same input as in `input_data`
        "grid_kwargs": {"coords": {"Z": {"center": "z", "outer": "zc"}}},
        "transform_kwargs": {"method": "conservative"},
    },
}


def constructor(case_param_dict):
    """create test components from `cases` dictionary parameters"""

    def _construct_ds(dict, prefix):
        # ds = xr.Dataset({'test':xr.DataArray(case_param_dict[prefix+"_data"][1])})
        data = case_param_dict[prefix + "_data"][1]

        ds = xr.Dataset(
            {
                case_param_dict[prefix + "_data"][0]: xr.DataArray(
                    data,
                    dims=[case_param_dict[prefix + "_coord"][0]],
                    coords={
                        case_param_dict[prefix + "_coord"][0]: case_param_dict[
                            prefix + "_coord"
                        ][1]
                    },
                )
            }
        )
        # add additional coords (bounds)
        if prefix + "_bounds_coord" in case_param_dict.keys():
            bounds = case_param_dict[prefix + "_bounds_coord"]
            ds = ds.assign_coords({bounds[0]: bounds[1]})

        # mask values from the output
        if prefix + "_data_mask_index" in case_param_dict.keys():
            for ii in case_param_dict[prefix + "_data_mask_index"]:
                ds.data.data[ii] = case_param_dict[prefix + "_data_mask_value"]
        return ds

    input = _construct_ds(case_param_dict, "input")
    expected = _construct_ds(case_param_dict, "expected")
    target = xr.DataArray(
        case_param_dict["target_data"][1],
        dims=[case_param_dict["target_coord"][0]],
        coords={case_param_dict["target_coord"][0]: case_param_dict["target_coord"][1]},
        name=case_param_dict["target_data"][0],
    )

    return (
        input,
        case_param_dict["grid_kwargs"],
        target,
        case_param_dict["transform_kwargs"],
        expected,
    )


# TODO: I am not sure how we would handle periodic axes atm. Should we raise an error?


@pytest.fixture(
    scope="module",
    params=list(cases.keys()),
)
def all_cases(request):
    return constructor(cases[request.param])


@pytest.fixture(
    scope="module",
    params=[c for c in list(cases.keys()) if "linear" in c],
)
def linear_cases(request):
    return constructor(cases[request.param])


@pytest.fixture(
    scope="module",
    params=[c for c in list(cases.keys()) if "conservative" in c],
)
def conservative_cases(request):
    return constructor(cases[request.param])


"""Test suite."""


def test_linear_interpolation_target_value_error():
    """Test that linear_interpolation throws an error when `target` is a np array"""
    input, grid_kwargs, target, transform_kwargs, expected = constructor(
        cases["linear_depth_depth"]
    )

    # method keyword is only for high level tests
    transform_kwargs.pop("method")

    with pytest.raises(ValueError):
        interpolated = linear_interpolation(input.data, input.z, target.data, "z", "z")
    # TODO: test with the other method


def _parse_dim(da):
    # utility that check that the input array has only one dim and return that
    assert len(da.dims) == 1
    return list(da.dims)[0]


def test_low_level_linear(linear_cases):
    """Test the linear interpolations on the xarray wrapper level"""
    input, grid_kwargs, target, transform_kwargs, expected = linear_cases

    # method keyword is only for high level tests
    transform_kwargs.pop("method")

    input_dim = _parse_dim(input.data)
    target_dim = _parse_dim(target)

    interpolated = linear_interpolation(
        input.data,
        input[input_dim],
        target,
        input_dim,
        input_dim,
        target_dim,
        **transform_kwargs
    )
    xr.testing.assert_allclose(interpolated, expected.data)


def test_low_level_conservative(conservative_cases):
    """Test the conservative interpolations on the xarray wrapper level"""
    input, grid_kwargs, target, transform_kwargs, expected = conservative_cases

    # method keyword is only for high level tests
    transform_kwargs.pop("method")

    input_dim = grid_kwargs["coords"]["Z"]["center"]
    bounds_dim = grid_kwargs["coords"]["Z"]["outer"]
    target_dim = _parse_dim(target)

    interpolated = conservative_interpolation(
        input.data,
        input[bounds_dim],
        target,
        input_dim,
        bounds_dim,
        target_dim,
        **transform_kwargs
    )
    xr.testing.assert_allclose(interpolated, expected.data)


# def test_grid_transform(all_cases):
#     input, grid_kwargs, target, transform_kwargs, expected = dataset

#     axis = list(grid_kwargs["coords"].keys())[0]
#     dim = grid_kwargs["coords"][axis]["center"]
#     da_target = xr.DataArray(target, dims=[dim], coords={dim: target})

#     grid = Grid(input, **grid_kwargs)
#     transformed = grid.transform(input.data, axis, da_target, **transform_kwargs)
#     xr.testing.assert_allclose(transformed, expected)
