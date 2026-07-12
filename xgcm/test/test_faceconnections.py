import numpy as np
import pytest
import xarray as xr

from xgcm.grid import Grid
from xgcm.metadata_parsers import parse_comodo


@pytest.fixture(scope="module")
def ds():
    N = 25  # TODO: Reduce the size here. I think something like 4 or 5 is sufficient
    return xr.Dataset(
        {
            "data_c": (["face", "y", "x"], np.random.rand(2, N, N)),
            "u": (
                ["face", "xl", "y"],
                np.random.rand(2, N, N),
            ),  # TODO: Will it make testing easier if I make these not random?
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


# TODO: These should be reused in padding tests
@pytest.fixture(scope="module")
def ds_face_connections_x_to_y_reverse():
    return {
        "face": {0: {"X": (None, (1, "Y", True))}, 1: {"Y": ((0, "X", True), None)}}
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


def test_connection_errors(ds):
    pass


@pytest.mark.parametrize("face_coord_dim", [True, False])
def test_create_connected_grid(ds, ds_face_connections_x_to_x, face_coord_dim):
    #
    if face_coord_dim:
        ds = ds.drop_vars("face")

    # simplest scenario with one face connection
    grid = Grid(ds, face_connections=ds_face_connections_x_to_x)

    xaxis = grid.axes["X"]

    # make sure we have actual axis objects in the connection dict
    # this is a bad test because it tests the details of the implementation,
    # not the behavior. But it is useful for now
    assert xaxis._facedim == "face"
    assert xaxis._face_connections[0][1][0] == 1
    assert xaxis._face_connections[0][1][1] is xaxis
    assert xaxis._face_connections[1][0][0] == 0
    assert xaxis._face_connections[1][0][1] is xaxis


def test_create_connected_grid_error_wrong_facedim(ds, ds_face_connections_x_to_x):
    # rename face dimension to trigger error
    ds = ds.rename({"face": "something_else"})
    with pytest.raises(
        ValueError, match="Face dimension face does not exist in the dataset."
    ):
        Grid(ds, face_connections=ds_face_connections_x_to_x)


def test_diff_interp_connected_grid_x_to_x(ds, ds_face_connections_x_to_x):
    # simplest scenario with one face connection
    grid = Grid(ds, face_connections=ds_face_connections_x_to_x, padding="fill")
    diff_x = grid.diff(ds.data_c, "X", padding="fill")
    interp_x = grid.interp(ds.data_c, "X", padding="fill")

    # make sure the face connection got applied correctly
    np.testing.assert_allclose(
        diff_x[1, :, 0], ds.data_c[1, :, 0] - ds.data_c[0, :, -1]
    )
    np.testing.assert_allclose(
        interp_x[1, :, 0], 0.5 * (ds.data_c[1, :, 0] + ds.data_c[0, :, -1])
    )

    # make sure the left padding got applied correctly
    np.testing.assert_allclose(diff_x[0, :, 0], ds.data_c[0, :, 0] - 0.0)
    np.testing.assert_allclose(interp_x[0, :, 0], 0.5 * (ds.data_c[0, :, 0] + 0.0))


def test_diff_interp_connected_grid_x_to_y(ds, ds_face_connections_x_to_y):
    # one face connection, rotated
    grid = Grid(ds, face_connections=ds_face_connections_x_to_y)

    diff_y = grid.diff(ds.data_c, "Y", padding="fill")
    interp_y = grid.interp(ds.data_c, "Y", padding="fill")

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


@pytest.mark.parametrize("padding", ["periodic", "fill"])
def test_vector_connected_grid_x_to_y(ds, ds_face_connections_x_to_y, padding):
    # one face connection, rotated
    grid = Grid(
        ds,
        face_connections=ds_face_connections_x_to_y,
        padding=padding,
        fill_value=1,
    )
    # ! Set padding on grid, so it is applied to all axes.
    # TODO: modify the non velocity tests too (after release)

    # modify the values of the dataset, so we know what to expect from the output
    # TODO: Maybe change this in the dataset definition?
    u_modifier = xr.DataArray([-2, -1], dims="face")
    v_modifier = xr.DataArray([1, 1], dims="face")
    u = ds.u * 0 + u_modifier
    v = ds.v * 0 + v_modifier

    # no need to check for diff vs interp. They all go through the same dispatch
    # v is the interesting variable here because it involves a sign change for this
    # connection (see https://github.com/xgcm/xgcm/issues/410#issue-1098348557)
    v_out = grid.interp({"Y": v}, "X", other_component={"X": u})
    # the test case is set up in a way that all interpolated values for u should be 1
    # if the face connection is done properly
    np.testing.assert_allclose(v_out.data, 1)


@pytest.mark.parametrize("no_coords", [True, False])
def test_vector_diff_interp_connected_grid_x_to_y(
    ds, ds_face_connections_x_to_y, no_coords
):
    # TODO: this is not elegant. This test should perhaps not use metadata parsing.
    # Instead we can use a dataset_factory fixture to create the dataset, and input kwargs with different input options (e.g. no coords)
    if no_coords:
        """Trigger error in https://github.com/xgcm/xgcm/issues/595 and https://github.com/xgcm/xgcm/issues/531 by removing coords from dataset."""
        # parse comodo metadata before removing coords
        ds, comodo_grid_kwargs = parse_comodo(ds)
        ds = ds.drop_vars(
            [di for di in ds.dims if di != "face"]
        )  # need to retain the face dimension coordinates here. I wonder if this should actually work without coords?
        grid = Grid(
            ds,
            **comodo_grid_kwargs,
            face_connections=ds_face_connections_x_to_y,
            autoparse_metadata=False,
        )
    else:
        # simplest scenario with one face connection
        grid = Grid(ds, face_connections=ds_face_connections_x_to_y)

    # interp u and v to cell center
    vector_center = grid.interp_2d_vector(
        {"X": ds.u, "Y": ds.v},
        to="center",
        padding="fill",
        fill_value=100,
    )
    u_c_interp = vector_center["X"]

    vector_diff = grid.diff_2d_vector(
        {"X": ds.u, "Y": ds.v},
        to="center",
        padding="fill",
        fill_value=100,
    )
    u_c_diff = vector_diff["X"]

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
        _ = grid.interp_2d_vector({"X": ds.v, "Y": ds.u}, to="left", padding="fill")
    with pytest.raises(NotImplementedError):
        _ = grid.interp_2d_vector({"X": ds.v, "Y": ds.u}, padding="fill")


@pytest.mark.parametrize("method", ["interp_2d_vector", "diff_2d_vector"])
def test_vector_diff_interp_connected_grid_x_to_y_dask(
    ds, ds_face_connections_x_to_y, method
):
    """Regression test for https://github.com/xgcm/xgcm/issues/704.

    When the vector components are dask-backed, padding splits the core
    dimension into boundary chunks, so the operation goes through
    ``_rechunk_to_merge_in_boundary_chunks``. That helper used to assume the
    argument was always a ``DataArray`` and raised
    ``AttributeError: 'dict' object has no attribute 'variable'`` when handed
    the ``{"X": u}`` vector-component dict (see the traceback in #704). Here we
    keep the inputs lazy (single chunk per core dim) so the result must match
    the numpy path exactly.
    """
    pytest.importorskip("dask")

    grid = Grid(ds, face_connections=ds_face_connections_x_to_y)

    # Keep the components lazy with a single chunk per core dim. This mirrors
    # the #704 scenario: map_overlap stays False, but padding still chunks the
    # core dim, exercising _rechunk_to_merge_in_boundary_chunks.
    u = ds.u.chunk()
    v = ds.v.chunk()

    vector_out = getattr(grid, method)(
        {"X": u, "Y": v},
        to="center",
        padding="fill",
        fill_value=100,
    )
    u_c = vector_out["X"]

    # The result should stay on the dask path.
    assert u_c.chunks is not None

    # Values must match the numpy path: first point is normal, last point picks
    # up the rotated neighbour across the face connection.
    if method == "interp_2d_vector":
        np.testing.assert_allclose(
            u_c.data[0, 0, :], 0.5 * (ds.u.data[0, 0, :] + ds.u.data[0, 1, :])
        )
        np.testing.assert_allclose(
            u_c.data[0, -1, :], 0.5 * (ds.u.data[0, -1, :] + ds.v.data[1, ::-1, 0])
        )
    else:
        np.testing.assert_allclose(
            u_c.data[0, 0, :], ds.u.data[0, 1, :] - ds.u.data[0, 0, :]
        )
        np.testing.assert_allclose(
            u_c.data[0, -1, :], -ds.u.data[0, -1, :] + ds.v.data[1, ::-1, 0]
        )


@pytest.mark.parametrize("method", ["interp_2d_vector", "diff_2d_vector"])
def test_vector_diff_interp_connected_grid_x_to_y_dask_multichunk(
    ds, ds_face_connections_x_to_y, method
):
    """Regression test for https://github.com/xgcm/xgcm/issues/708.

    When a vector component has more than one chunk along its core dimension,
    ``_1d_grid_ufunc_dispatch`` routes through the ``map_overlap`` path
    (``_map_func_over_core_dims``) instead of the single-chunk path exercised by
    ``test_vector_diff_interp_connected_grid_x_to_y_dask`` (#704). Padding a
    vector component across a face connection pulls in data from the connected
    face, which forces dask to rechunk the *non-core* dims (e.g. splitting the
    face dim). The ``adjust_chunks`` spec handed to dask used to be derived from
    the pre-pad array, so it still expected the original block count and the
    computation failed with ``ValueError: Dimension 0 has 2 blocks,
    adjust_chunks specified with 1 blocks``. The result must now match the numpy
    path exactly.
    """
    pytest.importorskip("dask")

    grid = Grid(ds, face_connections=ds_face_connections_x_to_y)

    # >1 chunk along the core dim flips the dispatch onto the map_overlap path.
    u = ds.u.chunk({"xl": 10})
    v = ds.v.chunk({"yl": 10})

    vector_out = getattr(grid, method)(
        {"X": u, "Y": v},
        to="center",
        padding="fill",
        fill_value=100,
    )
    u_c = vector_out["X"]

    # The result should stay on the dask path, with the core dim still chunked.
    assert u_c.chunks is not None
    assert len(u_c.variable.chunksizes["x"]) > 1

    # Values must match the numpy path: first point is normal, last point picks
    # up the rotated neighbour across the face connection.
    if method == "interp_2d_vector":
        np.testing.assert_allclose(
            u_c.data[0, 0, :], 0.5 * (ds.u.data[0, 0, :] + ds.u.data[0, 1, :])
        )
        np.testing.assert_allclose(
            u_c.data[0, -1, :], 0.5 * (ds.u.data[0, -1, :] + ds.v.data[1, ::-1, 0])
        )
    else:
        np.testing.assert_allclose(
            u_c.data[0, 0, :], ds.u.data[0, 1, :] - ds.u.data[0, 0, :]
        )
        np.testing.assert_allclose(
            u_c.data[0, -1, :], -ds.u.data[0, -1, :] + ds.v.data[1, ::-1, 0]
        )


def test_create_cubed_sphere_grid(cs, cubed_sphere_connections):
    _ = Grid(cs, face_connections=cubed_sphere_connections)


def test_diff_interp_cubed_sphere(cs, cubed_sphere_connections):
    # Note: no `boundary` is specified. On a fully connected topology like the
    # cubed sphere, every edge that requires padding gets its halo from a face
    # connection, so no boundary condition is needed and the padding-time
    # "no boundary condition was specified" error must not fire.
    grid = Grid(cs, face_connections=cubed_sphere_connections)
    face, _ = xr.broadcast(cs.face, cs.data_c)

    face_diff_x = grid.diff(face, "X")
    np.testing.assert_allclose(face_diff_x[:, 0, 0], [-3, 1, 1, 1, 1, 2])
    np.testing.assert_allclose(face_diff_x[:, -1, 0], [-3, 1, 1, 1, 1, 2])

    face_diff_y = grid.diff(face, "Y")
    np.testing.assert_allclose(face_diff_y[:, 0, 0], [-4, -3, -2, -1, 2, 5])
    np.testing.assert_allclose(face_diff_y[:, 0, -1], [-4, -3, -2, -1, 2, 5])

    # interp must work without a boundary condition as well
    face_interp_x = grid.interp(face, "X")
    np.testing.assert_allclose(face_interp_x[:, 0, 0], [1.5, 0.5, 1.5, 2.5, 3.5, 4.0])


def test_unconnected_edge_without_boundary_raises(ds, ds_face_connections_x_to_x):
    # Regression test for the no-boundary guard on partially connected grids:
    # faces 0 and 1 are joined along X in the middle, but the outer X edges
    # (left edge of face 0, right edge of face 1) are unconnected. With no
    # boundary condition specified anywhere, padding along X genuinely needs a
    # boundary for those outer edges, so the informative error must still fire.
    grid = Grid(ds, face_connections=ds_face_connections_x_to_x)
    with pytest.raises(ValueError, match="No boundary condition was specified"):
        grid.diff(ds.data_c, "X")

    # ...but supplying the padding at operation time must work.
    _ = grid.diff(ds.data_c, "X", padding="fill")


def test_cubed_sphere_scalar_pad_connected_halos(cs, cubed_sphere_connections):
    # Regression test for GH #712. Padding a field whose connected dims carry no
    # coordinate variables used to leave the source slice 1-D (the dim-restoring
    # ``expand_dims`` was gated on the dim having a coordinate), so
    # ``xr.concat(..., join="override")`` transposed and clobbered the orthogonal
    # connected edge. Because the axes were iterated in ``set`` (hash-seed) order,
    # which edge came out wrong varied from one Python process to the next.
    #
    # Here we pad a coordinate-less field equal to the face index, so every
    # connected halo cell must read the neighbor face that the connections declare.
    from xgcm.padding import pad as _pad

    grid = Grid(cs, face_connections=cubed_sphere_connections)
    nf, ny, nx = cs.sizes["face"], cs.sizes["y"], cs.sizes["x"]
    face_field = xr.DataArray(
        np.broadcast_to(np.arange(nf)[:, None, None], (nf, ny, nx)).astype(float),
        dims=("face", "y", "x"),
    )
    padded = _pad(
        face_field,
        grid,
        {"X": (1, 1), "Y": (1, 1)},
        padding={"X": "fill", "Y": "fill"},
        fill_value=np.nan,
    ).values  # (face, y+2, x+2)

    for f in range(nf):
        conn = cubed_sphere_connections["face"][f]
        (left_x, right_x), (down_y, up_y) = conn["X"], conn["Y"]
        # interior of each edge (exclude the two corner cells) reads the source face
        np.testing.assert_array_equal(padded[f, 1:-1, 0], left_x[0])
        np.testing.assert_array_equal(padded[f, 1:-1, -1], right_x[0])
        np.testing.assert_array_equal(padded[f, 0, 1:-1], down_y[0])
        np.testing.assert_array_equal(padded[f, -1, 1:-1], up_y[0])


class TestErrors:
    def test_vector_missing_other_component(self, ds, ds_face_connections_x_to_y):
        grid = Grid(ds, face_connections=ds_face_connections_x_to_y)
        msg = "Padding vector components requires `other_component` input"
        with pytest.raises(ValueError, match=msg):
            grid.diff(
                {"X": ds.u},
                "X",
                other_component=None,
            )
