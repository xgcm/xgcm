from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np
from dask.array import from_array

from xgcm.grid import Grid, Axis


@pytest.fixture(scope="module")
def ds():
    N = 25
    return xr.Dataset(
        {
            "data_c": (["face", "y", "x"], np.random.rand(2, N, N)),
            "u": (["face", "xl", "y"], np.random.rand(2, N, N)),
            "v": (["face", "x", "yl"], np.random.rand(2, N, N)),
        },
        coords={
            "x": (("x",), np.arange(N), {"axis": "X"}),
            "xl": (
                ("xl"),
                np.arange(N) - 0.5,
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "y": (("y",), np.arange(N), {"axis": "Y"}),
            "yl": (
                ("yl"),
                np.arange(N) - 0.5,
                {"axis": "Y", "c_grid_axis_shift": -0.5},
            ),
            "face": (("face",), [0, 1]),
        },
    )


# ---- structure of face_connections dictionaries ----
# key: index of face
# value: another dictionary
#   key: axis name
#   value: a tuple of link specifiers
#        neighbor face index,
#          neighboring axis to connect to,
#            whether to reverse the connection


@pytest.fixture(scope="module")
def ds_face_connections_x_to_x():
    return {
        "face": {0: {"X": (None, (1, "X", False))}, 1: {"X": ((0, "X", False), None)}}
    }


@pytest.fixture(scope="module")
def ds_face_connections_x_to_y():
    return {
        "face": {0: {"X": (None, (1, "Y", False))}, 1: {"Y": ((0, "X", False), None)}}
    }


@pytest.fixture(scope="module")
def cs():
    # cubed-sphere
    N = 25
    ds = xr.Dataset(
        {"data_c": (["face", "y", "x"], np.random.rand(6, N, N))},
        coords={
            "x": (("x",), np.arange(N), {"axis": "X"}),
            "xl": (
                ("xl"),
                np.arange(N) - 0.5,
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "y": (("y",), np.arange(N), {"axis": "Y"}),
            "yl": (
                ("yl"),
                np.arange(N) - 0.5,
                {"axis": "Y", "c_grid_axis_shift": -0.5},
            ),
            "face": (("face",), np.arange(6)),
        },
    )
    return ds


# TODO: consider revising this to avoid any reversed connections, which
# can cause problems for vector interpolation
@pytest.fixture(scope="module")
def cubed_sphere_connections():
    return {
        "face": {
            0: {
                "X": ((3, "X", False), (1, "X", False)),
                "Y": ((4, "Y", False), (5, "Y", False)),
            },
            1: {
                "X": ((0, "X", False), (2, "X", False)),
                "Y": ((4, "X", False), (5, "X", True)),
            },
            2: {
                "X": ((1, "X", False), (3, "X", False)),
                "Y": ((4, "Y", True), (5, "Y", True)),
            },
            3: {
                "X": ((2, "X", False), (0, "X", False)),
                "Y": ((4, "X", True), (5, "X", False)),
            },
            4: {
                "X": ((3, "Y", True), (1, "Y", False)),
                "Y": ((2, "Y", True), (0, "Y", False)),
            },
            5: {
                "X": ((3, "Y", False), (1, "Y", True)),
                "Y": ((0, "Y", False), (2, "Y", True)),
            },
        }
    }


def test_create_periodic_grid(ds):
    ds = ds.isel(face=0)
    grid = Grid(ds, periodic=True)
    for axis_name in grid.axes:
        axis = grid.axes[axis_name]
        assert axis._facedim is None
        connect_left, connect_right = axis._connections[None]
        assert connect_left[0] is None
        assert connect_right[0] is None
        assert connect_left[1] is axis
        assert connect_right[1] is axis
        assert connect_left[2] is False
        assert connect_right[2] is False


def test_get_periodic_grid_edge(ds):
    ds = ds.isel(face=0)
    grid = Grid(ds, periodic=True)

    xaxis = grid.axes["X"]
    left_edge_data_x = xaxis._get_edge_data(ds.data_c)
    np.testing.assert_allclose(left_edge_data_x, ds.data_c.isel(x=-1).data[:, None])
    right_edge_data_x = xaxis._get_edge_data(ds.data_c, is_left_edge=False)
    np.testing.assert_allclose(right_edge_data_x, ds.data_c.isel(x=0).data[:, None])

    yaxis = grid.axes["Y"]
    left_edge_data_y = yaxis._get_edge_data(ds.data_c)
    np.testing.assert_allclose(left_edge_data_y, ds.data_c.isel(y=-1).data[None, :])
    right_edge_data_y = yaxis._get_edge_data(ds.data_c, is_left_edge=False)
    np.testing.assert_allclose(right_edge_data_y, ds.data_c.isel(y=0).data[None, :])


def test_connection_errors(ds):
    pass


def test_create_connected_grid(ds, ds_face_connections_x_to_x):
    # simplest scenario with one face connection
    grid = Grid(ds, face_connections=ds_face_connections_x_to_x)

    xaxis = grid.axes["X"]
    yaxis = grid.axes["Y"]

    # make sure we have actual axis objects in the connection dict
    # this is a bad test because it tests the details of the implementation,
    # not the behavior. But it is useful for now
    assert xaxis._facedim == "face"
    assert xaxis._connections[0][1][0] == 1
    assert xaxis._connections[0][1][1] is xaxis
    assert xaxis._connections[1][0][0] == 0
    assert xaxis._connections[1][0][1] is xaxis


def test_diff_interp_connected_grid_x_to_x(ds, ds_face_connections_x_to_x):
    # simplest scenario with one face connection
    grid = Grid(ds, face_connections=ds_face_connections_x_to_x)
    diff_x = grid.diff(ds.data_c, "X", boundary="fill")
    interp_x = grid.interp(ds.data_c, "X", boundary="fill")

    # make sure the face connection got applied correctly
    np.testing.assert_allclose(
        diff_x[1, :, 0], ds.data_c[1, :, 0] - ds.data_c[0, :, -1]
    )
    np.testing.assert_allclose(
        interp_x[1, :, 0], 0.5 * (ds.data_c[1, :, 0] + ds.data_c[0, :, -1])
    )

    # make sure the left boundary got applied correctly
    np.testing.assert_allclose(diff_x[0, :, 0], ds.data_c[0, :, 0] - 0.0)
    np.testing.assert_allclose(interp_x[0, :, 0], 0.5 * (ds.data_c[0, :, 0] + 0.0))


def test_diff_interp_connected_grid_x_to_y(ds, ds_face_connections_x_to_y):
    # one face connection, rotated
    grid = Grid(ds, face_connections=ds_face_connections_x_to_y)

    diff_x = grid.diff(ds.data_c, "X", boundary="fill")
    interp_x = grid.interp(ds.data_c, "X", boundary="fill")
    diff_y = grid.diff(ds.data_c, "Y", boundary="fill")
    interp_y = grid.interp(ds.data_c, "Y", boundary="fill")

    # make sure the face connection got applied correctly
    # non-same axis connections require rotation
    # ravel everything to avoid dealing with broadcasting
    np.testing.assert_allclose(
        diff_y.data[1, 0, :].ravel(),
        ds.data_c.data[1, 0, :].ravel() - ds.data_c.data[0, ::-1, -1].ravel(),
    )

    np.testing.assert_allclose(
        interp_y.data[1, 0, :].ravel(),
        0.5 * (ds.data_c.data[1, 0, :].ravel() + ds.data_c.data[0, ::-1, -1].ravel()),
    )

    # TODO: checking all the other boundaries


def test_vector_diff_interp_connected_grid_x_to_y(ds, ds_face_connections_x_to_y):
    # simplest scenario with one face connection
    grid = Grid(ds, face_connections=ds_face_connections_x_to_y)

    # interp u and v to cell center
    vector_center = grid.interp_2d_vector(
        {"X": ds.u, "Y": ds.v}, to="center", boundary="fill"
    )
    u_c_interp, v_c_interp = vector_center["X"], vector_center["Y"]

    vector_diff = grid.diff_2d_vector(
        {"X": ds.u, "Y": ds.v}, to="center", boundary="fill"
    )
    u_c_diff, v_c_diff = vector_diff["X"], vector_diff["Y"]

    # first point should be normal
    np.testing.assert_allclose(
        u_c_interp.data[0, 0, :], 0.5 * (ds.u.data[0, 0, :] + ds.u.data[0, 1, :])
    )
    np.testing.assert_allclose(
        u_c_diff.data[0, 0, :], ds.u.data[0, 1, :] - ds.u.data[0, 0, :]
    )

    # last point should be fancy
    np.testing.assert_allclose(
        u_c_interp.data[0, -1, :], 0.5 * (ds.u.data[0, -1, :] + ds.v.data[1, ::-1, 0])
    )
    np.testing.assert_allclose(
        u_c_diff.data[0, -1, :], -ds.u.data[0, -1, :] + ds.v.data[1, ::-1, 0]
    )

    # TODO: figure out tangent vectors
    with pytest.raises(NotImplementedError):
        vector_corner = grid.interp_2d_vector(
            {"X": ds.v, "Y": ds.u}, to="left", boundary="fill"
        )
    with pytest.raises(NotImplementedError):
        vector_corner = grid.interp_2d_vector({"X": ds.v, "Y": ds.u}, boundary="fill")


def test_create_cubed_sphere_grid(cs, cubed_sphere_connections):
    grid = Grid(cs, face_connections=cubed_sphere_connections)


def test_diff_interp_cubed_sphere(cs, cubed_sphere_connections):
    grid = Grid(cs, face_connections=cubed_sphere_connections)
    face, _ = xr.broadcast(cs.face, cs.data_c)

    face_diff_x = grid.diff(face, "X")
    np.testing.assert_allclose(face_diff_x[:, 0, 0], [-3, 1, 1, 1, 1, 2])
    np.testing.assert_allclose(face_diff_x[:, -1, 0], [-3, 1, 1, 1, 1, 2])

    face_diff_y = grid.diff(face, "Y")
    np.testing.assert_allclose(face_diff_y[:, 0, 0], [-4, -3, -2, -1, 2, 5])
    np.testing.assert_allclose(face_diff_y[:, 0, -1], [-4, -3, -2, -1, 2, 5])
