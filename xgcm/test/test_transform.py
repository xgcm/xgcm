"""Tests for transform logic.
The tests are split into three levels:
low level: functions operating on numpy(numba) level (located in transform.py)
mid level: function wrappers that operate on xarray objects (located in transform.py)
high level: API implementations in the grid object (located in grid.py)
"""

# TODO:
# - What happens when target_data and data are on difference coords? Should we interpolate onto the same.
# - performance test? It would be nice to have these right after so we can track the performance as this feature evolves

import pytest
import dask
import numpy as np
import xarray as xr
from ..transform import (
    interp_1d_linear,
    interp_1d_conservative,
    linear_interpolation,
    conservative_interpolation,
)

from xgcm.grid import Grid, Axis


"""1D Test datasets for various transformations.
This part of the module is organized as follows:

1. A nested dictionary with all parameters needed to construct a set of test datasets
2. Fixtures that use a construct_test_source_data function and all or a subset of the dictionary entries from 1. to returns:
     (source_dataset, grid_kwargs, target, transform_kwargs, expected_dataarray)
3. Test functions

"""

cases = {
    "linear_depth_depth": {
        "source_coord": ("z", [5, 25, 60]),
        "source_data": (
            "data",
            [0.23246861, 0.45175654, 0.58320681],
        ),  # random numbers
        "target_coord": ("z", [0, 7, 30, 60, 70]),
        "target_data": ("z", [0, 7, 30, 60, 70]),
        "expected_coord": ("z", [0, 7, 30, 60, 70]),
        "expected_data": (
            "data",
            np.interp(
                [0, 7, 30, 60, 70],
                [5, 25, 60],
                [0.23246861, 0.45175654, 0.58320681],
            ),
        ),  # same source as in `source_data`
        "grid_kwargs": {"coords": {"Z": {"center": "z"}}},
        "transform_kwargs": {"mask_edges": False, "method": "linear"},
    },
    "linear_depth_depth_masked_outcrops": {
        "source_coord": ("z", [5, 25, 60]),
        "source_data": (
            "data",
            [0.23246861, 0.45175654, 0.58320681],
        ),  # random numbers
        "target_coord": ("z", [0, 7, 30, 60, 70]),
        "target_data": ("z", [0, 7, 30, 60, 70]),
        "expected_coord": ("z", [0, 7, 30, 60, 70]),
        "expected_data": (
            "data",
            np.interp(
                [0, 7, 30, 60, 70],
                [5, 25, 60],
                [0.23246861, 0.45175654, 0.58320681],
            ),
        ),  # same source as in `source_data`
        "expected_data_mask_index": [0, -1],
        "expected_data_mask_value": np.nan,
        "grid_kwargs": {"coords": {"Z": {"center": "z"}}},
        "transform_kwargs": {"mask_edges": True, "method": "linear"},
    },
    "linear_depth_depth_renamed": {
        "source_coord": ("test", [5, 25, 60]),
        "source_data": (
            "data",
            [0.23246861, 0.45175654, 0.58320681],
        ),  # random numbers
        "target_coord": ("something", [0, 7, 30, 60, 70]),
        "target_data": ("something", [0, 7, 30, 60, 70]),
        "expected_coord": ("something", [0, 7, 30, 60, 70]),
        "expected_data": (
            "data",
            np.interp(
                [0, 7, 30, 60, 70],
                [5, 25, 60],
                [0.23246861, 0.45175654, 0.58320681],
            ),
        ),  # same source as in `source_data`
        "grid_kwargs": {"coords": {"Z": {"center": "test"}}},
        "transform_kwargs": {"mask_edges": False, "method": "linear"},
    },
    # example of interpolating onto a tracer that increases with depth
    # but with inverted target
    "linear_depth_dens": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35]),
        "target_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "target_data": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_data": (
            "data",
            [1.0, 4.0, 6.0, 5.6, 4.0, 2.0, -0.272727, -0.818182],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "transform_kwargs": {"method": "linear", "target_data": "dens"},
    },
    "linear_depth_negative_dens": {
        "source_coord": ("depth", [-5, -25, -60, -80, -100, -120]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [-5, -25, -60, -80, -100, -120]),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35]),
        "target_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "target_data": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_data": (
            "data",
            [1.0, 4.0, 6.0, 5.6, 4.0, 2.0, -0.272727, -0.818182],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "transform_kwargs": {"method": "linear", "target_data": "dens"},
    },
    # example of interpolating onto a tracer that increases with depth
    # with masked values
    "linear_depth_dens_masked": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35]),
        "target_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "target_data": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_data": (
            "data",
            [np.nan, 4.0, 6.0, 5.6, 4.0, 2.0, -0.272727, -0.818182],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "transform_kwargs": {
            "method": "linear",
            "target_data": "dens",
            "mask_edges": True,
        },
    },
    # example of interpolating onto a tracer that increases with depth
    # but with inverted target
    "linear_depth_dens_reverse": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35]),
        "target_coord": ("something", [27, 25, 20, 15, 11, 10, 5, 0]),
        "target_data": ("something", [27, 25, 20, 15, 11, 10, 5, 0]),
        "expected_coord": ("something", [27, 25, 20, 15, 11, 10, 5, 0]),
        "expected_data": (
            "data",
            [-0.818182, -0.272727, 2.0, 4.0, 5.6, 6.0, 4.0, 1.0],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "transform_kwargs": {"method": "linear", "target_data": "dens"},
    },
    "conservative_depth_depth": {
        "source_coord": ("z", [5, 25, 60]),
        "source_bounds_coord": ("zc", [0, 10, 50, 75]),
        "source_data": ("data", [1, 4, 0]),
        "target_coord": ("z", [0, 1, 10, 50, 80]),
        "target_data": ("z", [0, 1, 10, 50, 80]),
        "expected_coord": ("z", [0.5, 5.5, 30, 65]),
        "expected_data": (
            "data",
            [0.1, 0.9, 4.0, 0.0],
        ),  # same source as in `source_data`
        "grid_kwargs": {"coords": {"Z": {"center": "z", "outer": "zc"}}},
        "transform_kwargs": {"method": "conservative"},
    },
    "conservative_depth_depth_rename": {
        "source_coord": ("depth", [5, 25, 60]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 50, 75]),
        "source_data": ("data", [1, 4, 0]),
        "target_coord": ("something", [0, 1, 10, 50, 80]),
        "target_data": ("something", [0, 1, 10, 50, 80]),
        "expected_coord": ("something", [0.5, 5.5, 30, 65]),
        "expected_data": (
            "data",
            [0.1, 0.9, 4.0, 0.0],
        ),  # same source as in `source_data`
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "transform_kwargs": {"method": "conservative"},
    },
    # This works but is an uncommon case, where the 'tracer' which is the target
    # is located on the cell bounds
    "conservative_depth_dens_on_bounds": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 30, 70, 90, 110, 170]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": (
            "depth_bnds",
            [0, 10, 30, 70, 90, 110, 170],
        ),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35, 37]),
        "target_coord": ("dens", [0, 5, 36]),
        "target_data": ("dens", [0, 5, 36]),
        "expected_coord": ("dens", [2.5, 20.5]),
        "expected_data": (
            "data",
            [1, 10.5],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
    # same as above but with a decreasing tracer (e.g. temp)
    "conservative_depth_temp_on_bounds": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 30, 70, 90, 110, 170]),
        "source_data": ("data", [-3, 0, 2, 6, 4, 1]),
        "source_additional_data_coord": (
            "depth_bnds",
            [0, 10, 30, 70, 90, 110, 170],
        ),
        "source_additional_data": ("temp", [37, 35, 24, 20, 10, 5, 1]),
        "target_coord": ("temp", [0, 5, 36]),
        "target_data": ("temp", [0, 5, 36]),
        "expected_coord": ("temp", [2.5, 20.5]),
        "expected_data": (
            "data",
            [1, 10.5],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "temp",
        },
    },
    # example of interpolating onto a tracer that descreases with depth
    # This fails due to the temp not increasing. We should implement a heuristic
    # to switch the direction...
    "linear_depth_temp": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_additional_data": ("temp", [35, 24, 20, 10, 5, 1]),
        "target_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "target_data": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_coord": ("something", [0, 5, 10, 11, 15, 20, 25, 27]),
        "expected_data": (
            "data",
            [0, 5, 10, 11, 15, 20, 25, 27],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "error": True,  # this currently fails but shouldnt
        "transform_kwargs": {"method": "linear", "target_data": "temp"},
    },
    # This should error out I think. I think we need to interpolate the additional
    # dens data on the cell faces. Somehow this does compute though and returns 3 values?
    # seems like a bug.
    "conservative_depth_dens": {
        "source_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_bounds_coord": ("depth_bnds", [0, 10, 30, 70, 90, 110, 170]),
        "source_data": ("data", [1, 4, 6, 2, 0, -3]),
        "source_additional_data_coord": ("depth", [5, 25, 60, 80, 100, 120]),
        "source_additional_data": ("dens", [1, 5, 10, 20, 24, 35]),
        "target_coord": ("dens", [0, 5, 36]),
        "target_data": ("dens", [0, 5, 36]),
        "expected_coord": ("dens", [2.5, 7.5]),
        "expected_data": (
            "data",
            [5.0, 5.0],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "error": True,  # this currently fails but shouldnt
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
    # Check for bugfix. Tranform onto column with two or more equal values.
    "conservative_depth_dens_equalvalues": {
        "source_coord": ("depth", [0, 20]),
        "source_bounds_coord": ("depth_bnds", [10, 30, 50]),
        "source_data": ("data", [100, 2210]),
        "source_additional_data_coord": ("depth_bnds", [10, 30, 50]),
        "source_additional_data": ("dens", [2, 2, 1.9]),
        "target_coord": ("dens", [1.95, 3]),
        "target_data": ("dens", [1.95, 3]),
        "expected_coord": ("dens", [2.475]),
        "expected_data": (
            "data",
            [100 + 2210 / 2],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
    # adopted from G MacGilchrist's example here: https://github.com/gmacgilchrist/so_decadal_variability/blob/master/dev_nonmonotonic-test.ipynb
    "conservative_depth_dens_nonmono_edge": {
        "source_coord": ("depth", [10, 30, 50, 70, 90]),
        "source_bounds_coord": ("depth_bnds", [0, 20, 40, 60, 80, 100]),
        "source_data": ("data", [2200, 2600, 3000, 3400, 3800]),
        "source_additional_data_coord": ("depth_bnds", [0, 20, 40, 60, 80, 100]),
        "source_additional_data": ("dens", [1, 3, 5, 7, 5, 3]),
        "target_coord": ("dens", [0, 2, 4, 6, 8]),
        "target_data": ("dens", [0, 2, 4, 6, 8]),
        "expected_coord": ("dens", [1, 3, 5, 7]),
        "expected_data": (
            "data",
            [1100.0, 4300.0, 6400.0, 3200.0],
        ),
        "grid_kwargs": {"coords": {"Z": {"center": "depth", "outer": "depth_bnds"}}},
        "transform_kwargs": {
            "method": "conservative",
            "target_data": "dens",
        },
    },
}


def construct_test_source_data(case_param_dict):
    """create test components from `cases` dictionary parameters"""
    # make sure the original dict is not modified
    case_param_dict = {k: v for k, v in case_param_dict.items()}

    def _construct_ds(param_dict, prefix):
        data = param_dict[prefix + "_data"][1]

        ds = xr.Dataset(
            {
                param_dict[prefix + "_data"][0]: xr.DataArray(
                    data,
                    dims=[param_dict[prefix + "_coord"][0]],
                    coords={
                        param_dict[prefix + "_coord"][0]: param_dict[prefix + "_coord"][
                            1
                        ]
                    },
                )
            }
        )
        # Add additional data
        if (
            prefix + "_additional_data" in param_dict.keys()
            and prefix + "_additional_data_coord" in param_dict.keys()
        ):
            add_data = param_dict[prefix + "_additional_data"]
            add_data_coord = param_dict[prefix + "_additional_data_coord"]
            ds[add_data[0]] = xr.DataArray(
                add_data[1],
                dims=[add_data_coord[0]],
                coords={add_data_coord[0]: add_data_coord[1]},
            )

        # add additional coords (bounds)
        if prefix + "_bounds_coord" in param_dict.keys():
            bounds = param_dict[prefix + "_bounds_coord"]
            ds = ds.assign_coords({bounds[0]: bounds[1]})

        # mask values from the output
        if prefix + "_data_mask_index" in param_dict.keys():
            for ii in param_dict[prefix + "_data_mask_index"]:
                ds.data.data[ii] = param_dict[prefix + "_data_mask_value"]
        return ds

    source = _construct_ds(case_param_dict, "source")
    expected = _construct_ds(case_param_dict, "expected")
    target = xr.DataArray(
        case_param_dict["target_data"][1],
        dims=[case_param_dict["target_coord"][0]],
        coords={case_param_dict["target_coord"][0]: case_param_dict["target_coord"][1]},
        name=case_param_dict["target_data"][0],
    )

    # parse the 'target_data' from the actual source
    transform_kwargs = {k: v for k, v in case_param_dict["transform_kwargs"].items()}
    if "target_data" in transform_kwargs.keys():
        if transform_kwargs["target_data"] is not None:
            transform_kwargs["target_data"] = source[transform_kwargs["target_data"]]

    error_flag = case_param_dict.pop("error", None)

    return (
        source,
        {k: v for k, v in case_param_dict["grid_kwargs"].items()},
        target,
        transform_kwargs,
        expected,
        error_flag,
    )


# TODO: I am not sure how we would handle periodic axes atm. Should we raise an error?


@pytest.fixture(
    scope="module",
    params=list(cases.keys()),
)
def all_cases(request):
    return construct_test_source_data(cases[request.param])


@pytest.fixture(
    scope="module",
    params=[c for c in list(cases.keys()) if "linear" in c],
)
def linear_cases(request):
    return construct_test_source_data(cases[request.param])


@pytest.fixture(
    scope="module",
    params=[c for c in list(cases.keys()) if "conservative" in c],
)
def conservative_cases(request):
    return construct_test_source_data(cases[request.param])


@pytest.fixture(
    scope="module",
    params=[
        "conservative_depth_dens_nonmono_edge",
        "linear_depth_dens",
        "conservative_depth_dens",
    ],  # just pick a few to test the broadcasting and dask
)
def multidim_cases(request):
    return construct_test_source_data(cases[request.param])


"""Test suite."""


def _parse_dim(da):
    # utility that check that the source array has only one dim and return that
    assert len(da.dims) == 1
    return list(da.dims)[0]


"""Low level tests"""


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


"""Mid level tests"""


def test_linear_interpolation_target_value_error():
    """Test that linear_interpolation/conservative_interpolation throws an error when `target` is a np array"""
    (
        source,
        grid_kwargs,
        target,
        transform_kwargs,
        expected,
        error_flag,
    ) = construct_test_source_data(cases["linear_depth_depth"])

    with pytest.raises(ValueError):
        interpolated = linear_interpolation(
            source.data, source.z, target.data, "z", "z"
        )

    (
        source,
        grid_kwargs,
        target,
        transform_kwargs,
        expected,
        error_flag,
    ) = construct_test_source_data(cases["conservative_depth_depth"])
    with pytest.raises(ValueError):
        interpolated = conservative_interpolation(
            source.data, source.z, target.data, "z", "z"
        )


def test_mid_level_linear(linear_cases):
    """Test the linear interpolations on the xarray wrapper level"""
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = linear_cases

    # method keyword is only for high level tests
    transform_kwargs.pop("method")

    source_dim = _parse_dim(source.data)
    target_dim = _parse_dim(target)

    # parse the target_data manually
    target_data = transform_kwargs.pop("target_data", None)
    if target_data is None:
        target_data = source[source_dim]

    if error_flag:
        with pytest.xfail():
            interpolated = linear_interpolation(
                source.data,
                target_data,
                target,
                source_dim,
                source_dim,
                target_dim,
                **transform_kwargs
            )
    else:
        interpolated = linear_interpolation(
            source.data,
            target_data,
            target,
            source_dim,
            source_dim,
            target_dim,
            **transform_kwargs
        )
        xr.testing.assert_allclose(interpolated, expected.data)


def test_mid_level_conservative(conservative_cases):
    """Test the conservative interpolations on the xarray wrapper level"""
    (
        source,
        grid_kwargs,
        target,
        transform_kwargs,
        expected,
        error_flag,
    ) = conservative_cases

    # method keyword is only for high level tests
    transform_kwargs.pop("method")

    source_dim = grid_kwargs["coords"]["Z"]["center"]
    bounds_dim = grid_kwargs["coords"]["Z"]["outer"]
    target_dim = _parse_dim(target)

    # parse the target_data manually
    target_data = transform_kwargs.pop("target_data", None)
    if target_data is None:
        target_data = source[bounds_dim]
    if error_flag:
        with pytest.xfail():
            interpolated = conservative_interpolation(
                source.data,
                target_data,
                target,
                source_dim,
                bounds_dim,
                target_dim,
                **transform_kwargs
            )
    else:
        interpolated = conservative_interpolation(
            source.data,
            target_data,
            target,
            source_dim,
            bounds_dim,
            target_dim,
            **transform_kwargs
        )
        xr.testing.assert_allclose(interpolated, expected.data)


"""High level tests"""


def test_grid_transform(all_cases):
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = all_cases
    print(error_flag)

    axis = list(grid_kwargs["coords"].keys())[0]

    grid = Grid(source, **grid_kwargs)
    if error_flag:
        with pytest.xfail():
            transformed = grid.transform(source.data, axis, target, **transform_kwargs)
    else:
        transformed = grid.transform(source.data, axis, target, **transform_kwargs)
        xr.testing.assert_allclose(transformed, expected.data)


def test_grid_transform_auto_naming(all_cases):
    """Check that the naming for the new dimension is adapted for the output if the target is not passed as xr.Dataarray"""
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = all_cases

    # ? Not sure if we need to go through all the cases here...

    axis = list(grid_kwargs["coords"].keys())[0]

    grid = Grid(source, **grid_kwargs)

    # modify the expected naming and convert target to numpy array
    target_data = transform_kwargs.setdefault("target_data", None)

    if transform_kwargs["target_data"] is None:
        # When no target_data is provided default to axis coordinates depending on method
        if transform_kwargs["method"] == "linear":
            expected_data_coord = grid.axes[axis].coords["center"]
        elif transform_kwargs["method"] == "conservative":
            expected_data_coord = grid.axes[axis].coords["outer"]
    else:
        # When target_data is provided check against the name of the dataarray
        expected_data_coord = target_data.name

    target = target.data

    if error_flag:
        with pytest.xfail():
            transformed = grid.transform(source.data, axis, target, **transform_kwargs)
    else:
        transformed = grid.transform(source.data, axis, target, **transform_kwargs)
        assert expected_data_coord in transformed.coords


""" Multidimensional tests with dask scheduler """


# dask client fixtures (adapted from https://github.com/xgcm/fastjmd95/blob/2cbccf2144e73c13a91741029eb7fd273d380d28/fastjmd95/test/test_jmd95.py#L61)
@pytest.fixture
def no_client():
    return None


@pytest.fixture
def threaded_client():
    with dask.config.set(scheduler="threads"):
        yield


@pytest.fixture
def processes_client():
    with dask.config.set(scheduler="processes"):
        yield


@pytest.fixture(scope="module")
def distributed_client():
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(threads_per_worker=1, n_workers=2, processes=True)
    client = Client(cluster)
    yield
    client.close()
    del client
    cluster.close()
    del cluster


all_clients = ["no_client", "threaded_client", "processes_client", "distributed_client"]


@pytest.mark.parametrize("client", all_clients)
def test_grid_transform_multidim(request, client, multidim_cases):
    # broadcast the 1d column agains some other dims and make sure that the 1d results are still valid
    source, grid_kwargs, target, transform_kwargs, expected, error_flag = multidim_cases

    na = 3
    source = source * xr.DataArray(np.ones([na]), dims=["a"])

    if client != "no_client":
        source = source.chunk({"a": 1})

    # calculate the multidimensional result
    axis = list(grid_kwargs["coords"].keys())[0]
    grid = Grid(source, **grid_kwargs)

    # compare for each depth column
    if error_flag:
        with pytest.xfail():
            transformed = grid.transform(source.data, axis, target, **transform_kwargs)
    else:
        client = request.getfixturevalue(client)
        transformed = grid.transform(
            source.data, axis, target, **transform_kwargs
        ).compute(scheduler=client)
        for aa in range(na):
            transformed_column = transformed.isel({"a": aa})
            xr.testing.assert_allclose(transformed_column, expected.data)