# TODO: Swap this out for some of our fixtures?
import numpy as np
import pytest
import xarray as xr

from xgcm import Grid
from xgcm.padding import pad
from xgcm.test.datasets import datasets_grid_metric


@pytest.mark.parametrize(
    "boundary_width",
    [
        {"X": (1, 1)},
        {"Y": (0, 1)},
        {"X": (0, 1), "Y": (2, 0)},
    ],
)
class TestPadding:
    @pytest.mark.parametrize("fill_value", [np.nan, 0, 1.5])
    def test_padding_fill(self, boundary_width, fill_value):

        ds, coords, _ = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords)
        data = ds.tracer

        # iterate over all axes
        expected = data.copy(deep=True)
        for ax, widths in boundary_width.items():
            dim = grid._get_dims_from_axis(data, ax)[
                0
            ]  # ? not entirely sure why this is a list
            expected = expected.pad(
                {dim: widths}, "constant", constant_values=fill_value
            )
        result = pad(
            data,
            grid,
            boundary="fill",
            boundary_width=boundary_width,
            fill_value=fill_value,
        )
        xr.testing.assert_allclose(expected, result)

    def test_padding_extend(self, boundary_width):
        ds, coords, _ = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords)
        data = ds.tracer

        # iterate over all axes
        expected = data.copy(deep=True)
        for ax, widths in boundary_width.items():
            dim = grid._get_dims_from_axis(data, ax)[
                0
            ]  # ? not entirely sure why this is a list
            expected = expected.pad({dim: widths}, "edge")
        result = pad(
            data,
            grid,
            boundary="extend",
            boundary_width=boundary_width,
        )
        xr.testing.assert_allclose(expected, result)

    def test_padding_periodic(self, boundary_width):
        ds, coords, _ = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords)
        data = ds.tracer

        # iterate over all axes
        expected = data.copy(deep=True)
        for ax, widths in boundary_width.items():
            dim = grid._get_dims_from_axis(data, ax)[
                0
            ]  # ? not entirely sure why this is a list
            expected = expected.pad({dim: widths}, "wrap")
        result = pad(
            data,
            grid,
            boundary="periodic",
            boundary_width=boundary_width,
        )
        xr.testing.assert_allclose(expected, result)

    def test_padding_mixed(self, boundary_width):
        ds, coords, _ = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords)
        data = ds.tracer

        axis_padding_mapping = {"X": "periodic", "Y": "extend"}
        method_mapping = {
            "periodic": "wrap",
            "extend": "edge",
        }

        # iterate over all axes
        expected = data.copy(deep=True)
        for ax, widths in boundary_width.items():
            dim = grid._get_dims_from_axis(data, ax)[
                0
            ]  # ? not entirely sure why this is a list
            expected = expected.pad(
                {dim: widths}, method_mapping[axis_padding_mapping[ax]]
            )
        result = pad(
            data,
            grid,
            boundary=axis_padding_mapping,
            boundary_width=boundary_width,
        )
        xr.testing.assert_allclose(expected, result)

    # TODO: Add extrapolate, once we support it? Unless we do not support it anymore.


class TestPaddingDefaults:
    """Testing the behavior with default values for methods with additional input arguments"""

    def test_padding_fill(self):

        default_fill_value = 0
        # TODO: Change this to a fixed value (nan?) in future releases.

        ds, coords, _ = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords)
        data = ds.tracer

        expected = data.pad(
            {"xt": (0, 1)}, "constant", constant_values=default_fill_value
        )
        result = pad(data, grid, boundary="fill", boundary_width={"X": (0, 1)})
        xr.testing.assert_allclose(expected, result)


# TODO: We should make both boundary and boundary width positional arguments. Then this class can be deleted.
class TestPaddingErrors:
    def test_padding_no_width(self):
        ds, coords, _ = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords)
        data = ds.tracer

        with pytest.raises(
            ValueError, match="Must provide the widths of the boundaries"
        ):
            pad(data, grid, boundary="fill")

    @pytest.mark.xfail(reason="This currently defaults to boundary='periodic'")
    def test_padding_no_boundary(self):
        ds, coords, _ = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords)
        data = ds.tracer

        with pytest.raises(
            ValueError, match="Must provide the widths of the boundaries"
        ):
            pad(data, grid, boundary_width={"X": (0, 1)})


# TODO: Make sure that we cannot specify mixed methods for padding if the input is something like `cube-sphere` or `tripolar`
