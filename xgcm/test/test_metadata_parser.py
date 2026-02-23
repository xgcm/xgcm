import numpy as np
import pytest
import xarray as xr

from xgcm.metadata_parser import cf_parser


def test_cf_onedimensional():
    # make a super simple cf dataset
    ds_wo_bounds = xr.DataArray(
        np.ones(3),
        dims=["a"],
        coords={
            "a": xr.DataArray(np.arange(3), dims=["a"], attrs={"axis": "X"}),
        },
    ).to_dataset(name="data")

    ds_w_bounds = ds_wo_bounds.assign_coords(
        {
            "a_bounds": xr.DataArray(
                np.vstack([np.arange(3) - 0.5, np.arange(3) + 0.5]),
                dims=["bounds", "a"],
            )
        }
    )
    ds_w_bounds["a"].attrs["bounds"] = "a_bounds"

    parsed_w_bounds = cf_parser(ds_w_bounds)
    with pytest.warns(UserWarning):
        parsed_wo_bounds = cf_parser(ds_wo_bounds)

    # for this simple example I expect the two generated 'outer' dimensions to be identical
    xr.testing.assert_identical(
        parsed_w_bounds["ds"]["a_outer"], parsed_wo_bounds["ds"]["a_outer"]
    )

    # Additionally all arguments besides the dataset should be identical
    for k in set(list(parsed_wo_bounds.keys()) + list(parsed_w_bounds.keys())) - set(
        "ds"
    ):
        print(k)
        assert parsed_w_bounds[k] == parsed_wo_bounds[k]

    # TODO: Check that the grid created is equal and also equal to an expected manual grid
    # TODO: This requires implementation of Grid == Grid
