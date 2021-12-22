# TODO: Swap this out for some of our fixtures?
import numpy as np
import pytest
import xarray as xr

from xgcm import Grid
from xgcm.padding import _maybe_swap_dimension_names, pad
from xgcm.test.datasets import datasets_grid_metric
from xgcm.test.test_exchange import ds as ds_faces  # noqa: F401


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
        print(result)
        print(expected)
        xr.testing.assert_allclose(expected, result)

    def test_padding_None(self):
        "we currently expect the default padding to be periodic"
        ds, coords, _ = datasets_grid_metric("C")
        grid = Grid(ds, coords=coords)
        data = ds.tracer

        expected = data.pad({"xt": (0, 1)}, "wrap")
        result = pad(data, grid, boundary=None, boundary_width={"X": (0, 1)})
        print(result)
        print(expected)
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


@pytest.mark.parametrize("fill_value", [np.nan, 0])
class TestPaddingFaceConnection:
    @pytest.mark.parametrize(
        "boundary_width",
        [
            {"X": (1, 1)},  # Test one case with 'regular padding on the 'other' side.
            {"X": (1, 2)},
            {"X": (0, 1), "Y": (1, 0)},
        ],
    )
    def test_face_connections_right_left_same_axis(
        self, boundary_width, ds_faces, fill_value
    ):
        face_connections = {
            "face": {
                0: {"X": (None, (1, "X", False))},
                1: {"X": ((0, "X", False), None)},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        data = ds_faces.data_c
        # restrict data here, so its easier to see the output
        data = data.isel(y=slice(0, 2), x=slice(0, 2))
        data = data.reset_coords(drop=True).reset_index(data.dims, drop=True)

        # Manually put together the data
        face_0 = data.isel(face=0)
        face_1 = data.isel(face=1)

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        # pad on each side except on the connected side
        face_0_padded = face_0.pad(
            x=(boundary_width["X"][0], 0),
            y=(boundary_width["Y"][0], boundary_width["Y"][1]),
            mode="constant",
            constant_values=fill_value,
        )

        face_1_padded = face_1.pad(
            x=(0, boundary_width["X"][1]),
            y=(boundary_width["Y"][0], boundary_width["Y"][1]),
            mode="constant",
            constant_values=fill_value,
        )

        # then simply add the corresponding slice to each face according to the connection
        face_0_expected = xr.concat(
            [face_0_padded, face_1_padded.isel(x=slice(0, boundary_width["X"][1]))],
            dim="x",
        )
        face_1_expected = xr.concat(
            [
                face_0_padded.isel(
                    x=slice(
                        -boundary_width["X"][0],
                        None if boundary_width["X"][0] > 0 else 0,
                        # this is a bit annoying. if the boundary width on this side is
                        # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
                    )
                ),
                face_1_padded,
            ],
            dim="x",
        )

        expected = xr.concat([face_0_expected, face_1_expected], dim="face")
        result = pad(
            data,
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
        )
        xr.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "boundary_width",
        [
            {
                "X": (1, 1),
                "Y": (1, 1),
            },  # Test one case with 'regular padding on the 'other' side.
            {
                "X": (2, 2),
                "Y": (2, 2),
            },  # can we only allow same padding amount on each  side?
            {
                "X": (0, 1),
                "Y": (1, 0),
            },
        ],
    )
    def test_face_connections_right_left_swap_axis(
        self, boundary_width, ds_faces, fill_value
    ):
        face_connections = {
            "face": {
                0: {"X": (None, (1, "Y", False))},
                1: {"Y": ((0, "X", False), None)},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        data = ds_faces.data_c

        # restrict data here, so its easier to see the output
        data = data.isel(y=slice(0, 3), x=slice(0, 3))
        data = data.reset_coords(drop=True).reset_index(data.dims, drop=True)

        # Manually put together the data
        face_0 = data.isel(face=0)
        face_1 = data.isel(face=1)

        # pad on each side except on the connected side
        face_0_padded = face_0.pad(
            x=(boundary_width["X"][0], 0),
            y=(boundary_width["Y"][0], boundary_width["Y"][1]),
            mode="constant",
            constant_values=fill_value,
        )

        face_1_padded = face_1.pad(
            x=(boundary_width["X"][0], boundary_width["X"][1]),
            y=(0, boundary_width["Y"][1]),
            mode="constant",
            constant_values=fill_value,
        )

        # then simply add the corresponding slice to each face according to the connection
        # in this case we also need to rename them

        face_0_addition = face_1_padded.isel(y=slice(0, boundary_width["X"][1]))
        # Flip both of these along the orthogonal axis
        face_0_addition = face_0_addition.isel(x=slice(None, None, -1))
        # In this case we need to rename the 'addition' dimensions
        face_0_addition = _maybe_swap_dimension_names(face_0_addition, "y", "x")

        # Same steps for the other face
        face_1_addition = face_0_padded.isel(
            x=slice(
                -boundary_width["Y"][0],
                None if boundary_width["Y"][0] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        )
        face_1_addition = face_1_addition.isel(y=slice(None, None, -1))
        face_1_addition = _maybe_swap_dimension_names(
            face_1_addition,
            "x",
            "y",
        )

        face_0_expected = xr.concat(
            [face_0_padded, face_0_addition],
            dim="x",
        )
        face_1_expected = xr.concat(
            [
                face_1_addition,
                face_1_padded,
            ],
            dim="y",
        )

        expected = xr.concat([face_0_expected, face_1_expected], dim="face")

        result = pad(
            data,
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
        )
        xr.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "boundary_width",
        [
            {
                "X": (1, 1),
                "Y": (1, 1),
            },  # Test one case with 'regular padding on the 'other' side.
            {
                "X": (2, 2),
                "Y": (2, 2),
            },
            pytest.param(
                {
                    "X": (0, 1),
                    "Y": (1, 0),
                },
                # marks=pytest.mark.xfail(
                #     reason="can we only allow same padding amount on each  side?"
                #     # I think I need to solve this to get the padding to work for the other examples
                # ),
            ),
        ],
    )
    def test_face_connections_right_right_swap_axis(
        self, boundary_width, ds_faces, fill_value
    ):
        face_connections = {
            "face": {
                0: {"X": (None, (1, "Y", True))},
                1: {"Y": (None, (0, "X", True))},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        data = ds_faces.data_c

        # restrict data here, so its easier to see the output
        data = data.isel(y=slice(0, 3), x=slice(0, 3))
        data = data.reset_coords(drop=True).reset_index(data.dims, drop=True)

        # pad along all axes given in boundary_width, and then replace the connected axis
        # Note: If we chose a different method than 'fill' the final result would depend on the order of padding
        # and this might not work properly? For fill it shouldnt matter.
        padded = data.pad(
            x=(boundary_width["X"][0], boundary_width["X"][1]),
            y=(boundary_width["Y"][0], boundary_width["Y"][1]),
            mode="constant",
            constant_values=fill_value,
        )

        padded_x = data.pad(
            x=(boundary_width["X"][0], boundary_width["X"][1]),
            mode="constant",
            constant_values=fill_value,
        )

        padded_y = data.pad(
            y=(boundary_width["Y"][0], boundary_width["Y"][1]),
            mode="constant",
            constant_values=fill_value,
        )

        # manually add the x boundaries
        face_0 = padded.isel(face=0)
        face_1 = padded.isel(face=1)

        # Extract and process the additional data
        face_0_addition = padded_x[dict(face=1, y=slice(-boundary_width["Y"][1], None))]
        face_1_addition = padded_y[dict(face=0, x=slice(-boundary_width["X"][1], None))]

        # Flip both of these along the parallel axis (because the connection is reverse)
        face_0_addition = face_0_addition.isel(y=slice(None, None, -1))
        face_1_addition = face_1_addition.isel(x=slice(None, None, -1))

        # rename the dimension and concatenate each face
        face_0_addition = _maybe_swap_dimension_names(
            face_0_addition, "y", "x"
        ).transpose(*face_0.dims)
        face_1_addition = _maybe_swap_dimension_names(
            face_1_addition, "x", "y"
        ).transpose(*face_1.dims)

        face_0[dict(x=slice(-boundary_width["X"][1], None))] = face_0_addition
        face_1[dict(y=slice(-boundary_width["Y"][1], None))] = face_1_addition

        expected = xr.concat([face_0, face_1], dim="face")

        result = pad(
            data,
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
        )
        print(result)
        print(expected)
        xr.testing.assert_allclose(result, expected)
