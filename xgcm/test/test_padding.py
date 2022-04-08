# TODO: Swap this out for some of our fixtures?
import numpy as np
import pytest
import xarray as xr

from xgcm import Grid
from xgcm.padding import _maybe_swap_dimension_names, _strip_all_coords, pad
from xgcm.test.datasets import datasets_grid_metric
from xgcm.test.test_faceconnections import ds as ds_faces  # noqa: F401


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

        # we intentionally strip all coords from padded arrays
        expected = _strip_all_coords(expected)

        result = pad(
            data,
            grid,
            boundary="fill",
            boundary_width=boundary_width,
            fill_value=fill_value,
            other_component=None,
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

        # we intentionally strip all coords from padded arrays
        expected = _strip_all_coords(expected)

        result = pad(
            data,
            grid,
            boundary="extend",
            boundary_width=boundary_width,
            fill_value=None,
            other_component=None,
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

        # we intentionally strip all coords from padded arrays
        expected = _strip_all_coords(expected)

        result = pad(
            data,
            grid,
            boundary="periodic",
            boundary_width=boundary_width,
            fill_value=None,
            other_component=None,
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

        # we intentionally strip all coords from padded arrays
        expected = _strip_all_coords(expected)

        result = pad(
            data,
            grid,
            boundary=axis_padding_mapping,
            boundary_width=boundary_width,
            fill_value=None,
            other_component=None,
        )
        xr.testing.assert_allclose(expected, result)


# TODO: Make sure that we cannot specify mixed methods for padding if the input is something like `cube-sphere` or `tripolar`


# Some helper functions for the connections padding
def _prepad_right_left_same_axis(da, boundary_width, fill_value, x="x", y="y"):
    # Manually put together the data
    face_0 = da.isel(face=0)
    face_1 = da.isel(face=1)

    # pad on each side except on the connected side
    face_0_padded = face_0.pad(
        **{
            x: (boundary_width["X"][0], 0),
            y: (boundary_width["Y"][0], boundary_width["Y"][1]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )

    face_1_padded = face_1.pad(
        **{
            x: (0, boundary_width["X"][1]),
            y: (boundary_width["Y"][0], boundary_width["Y"][1]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )
    return face_0_padded, face_1_padded


def _prepad_right_right_same_axis(da, boundary_width, fill_value, x="x", y="y"):
    # Manually put together the data
    face_0 = da.isel(face=0)
    face_1 = da.isel(face=1)

    # pad on each side except on the connected side
    face_0_padded = face_0.pad(
        **{
            x: (boundary_width["X"][0], 0),
            y: (boundary_width["Y"][0], boundary_width["Y"][1]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )

    face_1_padded = face_1.pad(
        **{
            x: (boundary_width["X"][0], 0),
            y: (boundary_width["Y"][0], boundary_width["Y"][1]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )
    return face_0_padded, face_1_padded


def _prepad_right_left_swap_axis(da, boundary_width, fill_value, x="x", y="y"):
    # Manually put together the data
    face_0 = da.isel(face=0)
    face_1 = da.isel(face=1)

    # pad on each side except on the connected side
    face_0_padded = face_0.pad(
        **{
            x: (boundary_width["X"][0], 0),
            y: (boundary_width["Y"][0], boundary_width["Y"][1]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )

    face_1_padded = face_1.pad(
        **{
            x: (boundary_width["X"][0], boundary_width["X"][1]),
            y: (0, boundary_width["Y"][1]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )

    # Now pad each face according to the swapped axes, so that the connected slices match
    # This is only relevant when the boundary width is not equal for all sides.
    face_0_padded_swapped = face_0.pad(
        **{
            x: (boundary_width["Y"][0], 0),
            y: (boundary_width["X"][1], boundary_width["X"][0]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )

    face_1_padded_swapped = face_1.pad(
        **{
            x: (boundary_width["Y"][1], boundary_width["Y"][0]),
            y: (0, boundary_width["X"][1]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )
    return face_0_padded, face_1_padded, face_0_padded_swapped, face_1_padded_swapped


def _prepad_right_right_swap_axis(da, boundary_width, fill_value, x="x", y="y"):
    # Manually put together the data
    face_0 = da.isel(face=0)
    face_1 = da.isel(face=1)

    # pad on each side except on the connected side
    face_0_padded = face_0.pad(
        **{
            x: (boundary_width["X"][0], 0),
            y: (boundary_width["Y"][0], boundary_width["Y"][1]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )

    face_1_padded = face_1.pad(
        **{
            x: (boundary_width["X"][0], boundary_width["X"][1]),
            y: (boundary_width["Y"][0], 0),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )

    # Now pad each face according to the swapped axes, so that the connected slices match
    # This is only relevant when the boundary width is not equal for all sides.
    face_0_padded_swapped = face_0.pad(
        **{
            x: (boundary_width["Y"][0], 0),
            y: (boundary_width["X"][0], boundary_width["X"][1]),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )

    face_1_padded_swapped = face_1.pad(
        **{
            x: (boundary_width["Y"][0], boundary_width["Y"][1]),
            y: (boundary_width["X"][0], 0),
            "mode": "constant",
            "constant_values": fill_value,
        }
    )
    return face_0_padded, face_1_padded, face_0_padded_swapped, face_1_padded_swapped


@pytest.mark.parametrize("fill_value", [np.nan, 0])
@pytest.mark.parametrize(
    "boundary_width",
    [
        {"X": (1, 1)},
        {"X": (1, 2)},
        {"X": (0, 1)},
        {
            "X": (1, 1),
            "Y": (1, 1),
        },
        {
            "X": (2, 2),
            "Y": (2, 2),
        },
        {
            "X": (0, 1),
            "Y": (1, 0),
        },
        {
            "X": (0, 2),
            "Y": (1, 0),
        },
    ],
)
class TestPaddingFaceConnection:
    # TODO: Test that an error is raised if the boundary width exceeds the array shape.
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

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        # # restrict data here, so its easier to see the output
        # data = data.isel(y=slice(0, 2), x=slice(0, 2))
        data = data.reset_coords(drop=True).reset_index(data.dims, drop=True)

        # prepad the arrays
        face_0_padded, face_1_padded = _prepad_right_left_same_axis(
            data, boundary_width, fill_value
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
            other_component=None,
        )
        xr.testing.assert_allclose(result, expected)

    def test_face_connections_right_right_same_axis(
        self, boundary_width, ds_faces, fill_value
    ):
        face_connections = {
            "face": {
                0: {"X": (None, (1, "X", True))},
                1: {"X": (None, (0, "X", True))},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        data = ds_faces.data_c

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        # # restrict data here, so its easier to see the output
        # data = data.isel(y=slice(0, 2), x=slice(0, 2))
        data = data.reset_coords(drop=True).reset_index(data.dims, drop=True)

        # prepad the arrays
        face_0_padded, face_1_padded = _prepad_right_right_same_axis(
            data, boundary_width, fill_value
        )

        # Process the padded data
        face_0_addition = face_1_padded.isel(
            x=slice(
                -boundary_width["X"][1],
                None if boundary_width["X"][1] > 0 else 0,
            )
        )

        face_1_addition = face_0_padded.isel(
            x=slice(
                -boundary_width["X"][1],
                None if boundary_width["X"][1] > 0 else 0,
            )
        )

        # do a parallel flip since this connection is reversed
        face_0_addition = face_0_addition.isel(x=slice(None, None, -1))
        face_1_addition = face_1_addition.isel(x=slice(None, None, -1))

        # then simply add the corresponding slice to each face according to the connection
        face_0_expected = xr.concat(
            [face_0_padded, face_0_addition],
            dim="x",
        )
        face_1_expected = xr.concat(
            [face_1_padded, face_1_addition],
            dim="x",
        )

        expected = xr.concat([face_0_expected, face_1_expected], dim="face")
        result = pad(
            data,
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component=None,
        )
        xr.testing.assert_allclose(result, expected)

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

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        # restrict data here, so its easier to see the output
        data = data.isel(y=slice(0, 2), x=slice(0, 2))
        data = data.reset_coords(drop=True).reset_index(data.dims, drop=True)

        (
            face_0_padded,
            face_1_padded,
            face_0_padded_swapped,
            face_1_padded_swapped,
        ) = _prepad_right_left_swap_axis(data, boundary_width, fill_value)

        # then simply add the corresponding slice to each face according to the connection
        # in this case we also need to rename them

        face_0_addition = face_1_padded_swapped.isel(y=slice(0, boundary_width["X"][1]))
        # Flip both of these along the orthogonal axis
        face_0_addition = face_0_addition.isel(x=slice(None, None, -1))
        # In this case we need to rename the 'addition' dimensions
        face_0_addition = _maybe_swap_dimension_names(face_0_addition, "y", "x")

        # Same steps for the other face
        face_1_addition = face_0_padded_swapped.isel(
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
            [face_1_addition, face_1_padded],
            dim="y",
        )

        expected = xr.concat([face_0_expected, face_1_expected], dim="face")

        result = pad(
            data,
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component=None,
        )
        xr.testing.assert_allclose(result, expected)

    def test_face_connections_right_right_swap_axis(
        self, boundary_width, ds_faces, fill_value
    ):

        # set a default for boundary widths
        boundary_width = {k: boundary_width.get(k, (0, 0)) for k in ["X", "Y"]}

        face_connections = {
            "face": {
                0: {"X": (None, (1, "Y", True))},
                1: {"Y": (None, (0, "X", True))},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        data = ds_faces.data_c

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        # restrict data here, so its easier to see the output
        data = data.isel(y=slice(0, 3), x=slice(0, 3))
        data = data.reset_coords(drop=True).reset_index(data.dims, drop=True)

        (
            face_0_padded,
            face_1_padded,
            face_0_padded_swapped,
            face_1_padded_swapped,
        ) = _prepad_right_right_swap_axis(data, boundary_width, fill_value)

        # then simply add the corresponding slice to each face according to the connection
        # in this case we also need to rename them

        face_0_addition = face_1_padded_swapped.isel(
            y=slice(
                -boundary_width["X"][1],
                None if boundary_width["X"][1] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        )
        # Same steps for the other face
        face_1_addition = face_0_padded_swapped.isel(
            x=slice(
                -boundary_width["Y"][1],
                None if boundary_width["Y"][1] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        )

        # Flip both of these along the parallel axis
        face_0_addition = face_0_addition.isel(y=slice(None, None, -1))
        face_1_addition = face_1_addition.isel(x=slice(None, None, -1))
        # In this case we need to rename the 'addition' dimensions
        face_0_addition = _maybe_swap_dimension_names(face_0_addition, "y", "x")
        face_1_addition = _maybe_swap_dimension_names(face_1_addition, "x", "y")

        face_0_expected = xr.concat(
            [face_0_padded, face_0_addition],
            dim="x",
        )
        face_1_expected = xr.concat(
            [face_1_padded, face_1_addition],
            dim="y",
        )

        expected = xr.concat([face_0_expected, face_1_expected], dim="face")

        result = pad(
            data,
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component=None,
        )
        xr.testing.assert_allclose(result, expected)

    def test_vector_face_connections_right_left_same_axis(
        self, boundary_width, ds_faces, fill_value
    ):
        face_connections = {
            "face": {
                0: {"X": (None, (1, "X", False))},
                1: {"X": ((0, "X", False), None)},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        u = ds_faces.u
        v = ds_faces.v

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        # # restrict data here, so its easier to see the output
        # data = data.isel(y=slice(0, 2), x=slice(0, 2))
        u = u.reset_coords(drop=True).reset_index(u.dims, drop=True)
        v = v.reset_coords(drop=True).reset_index(v.dims, drop=True)

        # prepad the arrays
        u_face_0_padded, u_face_1_padded = _prepad_right_left_same_axis(
            u, boundary_width, fill_value, x="xl", y="y"
        )
        v_face_0_padded, v_face_1_padded = _prepad_right_left_same_axis(
            v,
            boundary_width,
            fill_value,
            x="x",
            y="yl",
        )

        # Slice the appropriate portion of the source face to concat
        u_face_0_addition = u_face_1_padded.isel(xl=slice(0, boundary_width["X"][1]))
        u_face_1_addition = u_face_0_padded.isel(
            xl=slice(
                -boundary_width["X"][0],
                None if boundary_width["X"][0] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        )
        v_face_0_addition = v_face_1_padded.isel(x=slice(0, boundary_width["X"][1]))
        v_face_1_addition = v_face_0_padded.isel(
            x=slice(
                -boundary_width["X"][0],
                None if boundary_width["X"][0] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        )

        # then simply add the corresponding slice to each face according to the connection
        u_face_0_expected = xr.concat(
            [u_face_0_padded, u_face_0_addition],
            dim="xl",
        )
        u_face_1_expected = xr.concat(
            [u_face_1_addition, u_face_1_padded],
            dim="xl",
        )

        v_face_0_expected = xr.concat(
            [v_face_0_padded, v_face_0_addition],
            dim="x",
        )
        v_face_1_expected = xr.concat(
            [v_face_1_addition, v_face_1_padded],
            dim="x",
        )

        u_expected = xr.concat([u_face_0_expected, u_face_1_expected], dim="face")
        v_expected = xr.concat([v_face_0_expected, v_face_1_expected], dim="face")

        # test u
        u_result = pad(
            {"X": u},
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component={"Y": v},
        )
        xr.testing.assert_allclose(u_result, u_expected)

        # test v
        v_result = pad(
            {"Y": v},
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component={"X": u},
        )
        xr.testing.assert_allclose(v_result, v_expected)

    def test_vector_face_connections_right_right_same_axis(
        self, boundary_width, ds_faces, fill_value
    ):
        face_connections = {
            "face": {
                0: {"X": (None, (1, "X", True))},
                1: {"X": (None, (0, "X", True))},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        u = ds_faces.u
        v = ds_faces.v

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        # # restrict data here, so its easier to see the output
        # data = data.isel(y=slice(0, 2), x=slice(0, 2))
        u = u.reset_coords(drop=True).reset_index(u.dims, drop=True)
        v = v.reset_coords(drop=True).reset_index(v.dims, drop=True)

        u_face_0_padded, u_face_1_padded = _prepad_right_right_same_axis(
            u, boundary_width, fill_value, x="xl", y="y"
        )
        v_face_0_padded, v_face_1_padded = _prepad_right_right_same_axis(
            v,
            boundary_width,
            fill_value,
            x="x",
            y="yl",
        )

        # Slice the appropriate portion of the source face to concat
        u_face_0_addition = u_face_1_padded.isel(
            xl=slice(
                -boundary_width["X"][1],
                None if boundary_width["X"][1] > 0 else 0,
            )
        )
        u_face_1_addition = u_face_0_padded.isel(
            xl=slice(
                -boundary_width["X"][1],
                None if boundary_width["X"][1] > 0 else 0,
            )
        )
        v_face_0_addition = v_face_1_padded.isel(
            x=slice(
                -boundary_width["X"][1],
                None if boundary_width["X"][1] > 0 else 0,
            )
        )
        v_face_1_addition = v_face_0_padded.isel(
            x=slice(
                -boundary_width["X"][1],
                None if boundary_width["X"][1] > 0 else 0,
            )
        )

        # do a parallel flip since this connection is reversed (and sign change for u)
        u_face_0_addition = -u_face_0_addition.isel(xl=slice(None, None, -1))
        u_face_1_addition = -u_face_1_addition.isel(xl=slice(None, None, -1))

        v_face_0_addition = v_face_0_addition.isel(x=slice(None, None, -1))
        v_face_1_addition = v_face_1_addition.isel(x=slice(None, None, -1))

        # then simply add the corresponding slice to each face according to the connection
        u_face_0_expected = xr.concat(
            [u_face_0_padded, u_face_0_addition],
            dim="xl",
        )
        u_face_1_expected = xr.concat(
            [u_face_1_padded, u_face_1_addition],
            dim="xl",
        )

        v_face_0_expected = xr.concat(
            [v_face_0_padded, v_face_0_addition],
            dim="x",
        )
        v_face_1_expected = xr.concat(
            [v_face_1_padded, v_face_1_addition],
            dim="x",
        )

        u_expected = xr.concat([u_face_0_expected, u_face_1_expected], dim="face")
        v_expected = xr.concat([v_face_0_expected, v_face_1_expected], dim="face")

        # test u
        u_result = pad(
            {"X": u},
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component={"Y": v},
        )
        xr.testing.assert_allclose(u_result, u_expected)

        # test v
        v_result = pad(
            {"Y": v},
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component={"X": u},
        )
        xr.testing.assert_allclose(v_result, v_expected)

    def test_vector_face_connections_right_left_swap_axis(
        self, boundary_width, ds_faces, fill_value
    ):
        face_connections = {
            "face": {
                0: {"X": (None, (1, "Y", False))},
                1: {"Y": ((0, "X", False), None)},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        u = ds_faces.u
        v = ds_faces.v

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        # # restrict data here, so its easier to see the output
        u = u.reset_coords(drop=True).reset_index(u.dims, drop=True)
        v = v.reset_coords(drop=True).reset_index(v.dims, drop=True)

        (
            u_face_0_padded,
            u_face_1_padded,
            u_face_0_padded_swapped,
            u_face_1_padded_swapped,
        ) = _prepad_right_left_swap_axis(u, boundary_width, fill_value, x="xl", y="y")

        (
            v_face_0_padded,
            v_face_1_padded,
            v_face_0_padded_swapped,
            v_face_1_padded_swapped,
        ) = _prepad_right_left_swap_axis(v, boundary_width, fill_value, x="x", y="yl")

        # Put together the additions for each face
        u_face_0_addition = v_face_1_padded_swapped.isel(
            yl=slice(0, boundary_width["X"][1])
        ).rename({"x": "xl", "yl": "y"})
        # Tangential flip (u doesnt need sign change)
        u_face_0_addition = u_face_0_addition.isel(xl=slice(None, None, -1))
        u_face_0_addition = _maybe_swap_dimension_names(u_face_0_addition, "y", "xl")

        u_face_1_addition = v_face_0_padded_swapped.isel(
            x=slice(
                -boundary_width["Y"][0],
                None if boundary_width["Y"][0] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        ).rename({"x": "xl", "yl": "y"})
        u_face_1_addition = -u_face_1_addition.isel(y=slice(None, None, -1))
        u_face_1_addition = _maybe_swap_dimension_names(u_face_1_addition, "y", "xl")

        # now v (this one needs a sign change)
        v_face_0_addition = u_face_1_padded_swapped.isel(
            y=slice(0, boundary_width["X"][1])
        ).rename({"xl": "x", "y": "yl"})
        # Tangential flip (v DOES need sign change)
        v_face_0_addition = -v_face_0_addition.isel(x=slice(None, None, -1))
        v_face_0_addition = _maybe_swap_dimension_names(v_face_0_addition, "yl", "x")

        v_face_1_addition = u_face_0_padded_swapped.isel(
            xl=slice(
                -boundary_width["Y"][0],
                None if boundary_width["Y"][0] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        ).rename({"xl": "x", "y": "yl"})
        v_face_1_addition = v_face_1_addition.isel(yl=slice(None, None, -1))
        v_face_1_addition = _maybe_swap_dimension_names(v_face_1_addition, "yl", "x")

        # then simply add the corresponding slice to each face according to the connection
        u_face_0_expected = xr.concat(
            [u_face_0_padded, u_face_0_addition],
            dim="xl",
        )
        u_face_1_expected = xr.concat(
            [u_face_1_addition, u_face_1_padded],
            dim="y",
        )

        v_face_0_expected = xr.concat(
            [v_face_0_padded, v_face_0_addition],
            dim="x",
        )
        v_face_1_expected = xr.concat(
            [v_face_1_addition, v_face_1_padded],
            dim="yl",
        )

        u_expected = xr.concat([u_face_0_expected, u_face_1_expected], dim="face")
        v_expected = xr.concat([v_face_0_expected, v_face_1_expected], dim="face")

        u_result = pad(
            {"X": u},
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component={"Y": v},
        )

        v_result = pad(
            {"Y": v},
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component={"X": u},
        )

        xr.testing.assert_allclose(u_result, u_expected)
        xr.testing.assert_allclose(v_result, v_expected)

    def test_vector_face_connections_right_right_swap_axis(
        self, boundary_width, ds_faces, fill_value
    ):
        face_connections = {
            "face": {
                0: {"X": (None, (1, "Y", True))},
                1: {"Y": (None, (0, "X", True))},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        u = ds_faces.u
        v = ds_faces.v

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        # # restrict data here, so its easier to see the output
        u = u.reset_coords(drop=True).reset_index(u.dims, drop=True)
        v = v.reset_coords(drop=True).reset_index(v.dims, drop=True)

        (
            u_face_0_padded,
            u_face_1_padded,
            u_face_0_padded_swapped,
            u_face_1_padded_swapped,
        ) = _prepad_right_right_swap_axis(u, boundary_width, fill_value, x="xl", y="y")

        (
            v_face_0_padded,
            v_face_1_padded,
            v_face_0_padded_swapped,
            v_face_1_padded_swapped,
        ) = _prepad_right_right_swap_axis(v, boundary_width, fill_value, x="x", y="yl")

        # Put together the additions for each face
        u_face_0_addition = v_face_1_padded_swapped.isel(
            yl=slice(
                -boundary_width["X"][1],
                None if boundary_width["X"][1] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        ).rename({"x": "xl", "yl": "y"})
        u_face_0_addition = -u_face_0_addition.isel(y=slice(None, None, -1))
        u_face_0_addition = _maybe_swap_dimension_names(u_face_0_addition, "y", "xl")

        u_face_1_addition = v_face_0_padded_swapped.isel(
            x=slice(
                -boundary_width["Y"][1],
                None if boundary_width["Y"][1] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        ).rename({"x": "xl", "yl": "y"})
        u_face_1_addition = u_face_1_addition.isel(xl=slice(None, None, -1))
        u_face_1_addition = _maybe_swap_dimension_names(u_face_1_addition, "y", "xl")

        # now v (this one needs a sign change)
        v_face_0_addition = u_face_1_padded_swapped.isel(
            y=slice(
                -boundary_width["X"][1],
                None if boundary_width["X"][1] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        ).rename({"xl": "x", "y": "yl"})
        # Tangential flip (v DOES need sign change)
        v_face_0_addition = v_face_0_addition.isel(yl=slice(None, None, -1))
        v_face_0_addition = _maybe_swap_dimension_names(v_face_0_addition, "yl", "x")

        v_face_1_addition = u_face_0_padded_swapped.isel(
            xl=slice(
                -boundary_width["Y"][1],
                None if boundary_width["Y"][1] > 0 else 0,
                # this is a bit annoying. if the boundary width on this side is
                # 0 I want nothing to be padded. but slice(0,None) pads the whole array...
            )
        ).rename({"xl": "x", "y": "yl"})
        v_face_1_addition = -v_face_1_addition.isel(x=slice(None, None, -1))
        v_face_1_addition = _maybe_swap_dimension_names(v_face_1_addition, "yl", "x")

        # then simply add the corresponding slice to each face according to the connection
        u_face_0_expected = xr.concat(
            [u_face_0_padded, u_face_0_addition],
            dim="xl",
        )
        u_face_1_expected = xr.concat(
            [u_face_1_padded, u_face_1_addition],
            dim="y",
        )

        v_face_0_expected = xr.concat(
            [v_face_0_padded, v_face_0_addition],
            dim="x",
        )
        v_face_1_expected = xr.concat(
            [v_face_1_padded, v_face_1_addition],
            dim="yl",
        )

        u_expected = xr.concat([u_face_0_expected, u_face_1_expected], dim="face")
        v_expected = xr.concat([v_face_0_expected, v_face_1_expected], dim="face")

        u_result = pad(
            {"X": u},
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component={"Y": v},
        )

        v_result = pad(
            {"Y": v},
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component={"X": u},
        )

        xr.testing.assert_allclose(u_result, u_expected)
        xr.testing.assert_allclose(v_result, v_expected)

    @pytest.mark.xfail(
        reason="Figuring out how to preserve the indicies with padding is super hard.",
        strict=True,
    )
    @pytest.mark.parametrize("boundary", ["constant", "wrap"])
    def test_vector_face_connections_coord_padding(
        self, boundary_width, ds_faces, fill_value, boundary
    ):
        # make sure that the complex padding acts like xarray.pad when it comes to dimension coordinates
        face_connections = {
            "face": {
                0: {"X": (None, (1, "Y", True))},
                1: {"Y": (None, (0, "X", True))},
            }
        }
        grid = Grid(ds_faces, face_connections=face_connections)
        u = ds_faces.u
        v = ds_faces.v

        # fill in zeros for y boundary width if not given
        boundary_width["Y"] = boundary_width.get("Y", (0, 0))

        padded_complex = pad(
            {"X": u},
            grid,
            boundary_width=boundary_width,
            boundary="fill",
            fill_value=fill_value,
            other_component={"Y": v},
        )
        padded_simple = u.pad(
            {"xl": boundary_width["X"], "y": boundary_width["Y"]},
            "constant",
            constant_values=fill_value,
        )  # TODO: add constant coord padding here once implemented in xarray.
        for di in u.dims:
            assert (di in padded_simple.coords and di in padded_complex.coords) or (
                di not in padded_simple.coords and di not in padded_complex.coords
            )
            if di in padded_simple.coords:
                xr.testing.assert_allclose(padded_complex[di], padded_simple[di])
