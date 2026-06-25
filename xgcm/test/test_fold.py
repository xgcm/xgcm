"""Tests for the bipolar north-fold boundary.

The global grids these serve are *tripolar* (South Pole + two Arctic poles); the
north fold itself is *bipolar* -- its seam is the line joining the two northern
poles. A fold boundary is requested as a per-axis ``boundary`` value on the
(single tile) fold axis -- the meridional "Y" axis -- e.g.
``boundary={"X": "periodic", "Y": {"fold": "corner"}}``. The northern edge of
the grid folds onto itself: the seam (zonal "X") axis is mirrored about the pole
and vector components reverse sign. See ``xgcm/padding.py`` for the pivot/offset
conventions.
"""

import numpy as np
import pytest
import xarray as xr

from xgcm import Grid
from xgcm.padding import _resolve_pivot, pad

Nx, Ny = 8, 5


def _make_ds():
    ds = xr.Dataset(
        coords={
            "xh": np.arange(Nx),
            "xl": np.arange(Nx),
            "yh": np.arange(Ny),
            "yl": np.arange(Ny),
        }
    )

    def fld(dy, dx):
        n = ds.sizes[dy] * ds.sizes[dx]
        return xr.DataArray(
            np.arange(n).reshape(ds.sizes[dy], ds.sizes[dx]).astype(float),
            dims=[dy, dx],
        )

    ds["c"] = fld("yh", "xh")  # tracer  (seam=center, fold=center)
    ds["u"] = fld("yh", "xl")  # u-point (seam=edge,   fold=center)
    ds["v"] = fld("yl", "xh")  # v-point (seam=center, fold=edge)
    ds["q"] = fld("yl", "xl")  # corner  (seam=edge,   fold=edge)
    return ds


def _grid(ds, pivot):
    return Grid(
        ds,
        coords={
            "X": {"center": "xh", "left": "xl"},
            "Y": {"center": "yh", "left": "yl"},
        },
        boundary={"X": "periodic", "Y": {"fold": pivot}},
        autoparse_metadata=False,
    )


# ---------------------------------------------------------------------------
# Pivot parsing / aliases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "alias, seam, fold",
    [
        ("center", "center", "center"),
        ("T", "center", "center"),
        ("corner", "edge", "edge"),
        ("F", "edge", "edge"),
        ("U", "edge", "center"),
        ("V", "center", "edge"),
    ],
)
def test_pivot_aliases(alias, seam, fold):
    roles = _resolve_pivot(alias, fold_axis="Y", seam_axis="X")
    assert roles == {"seam": seam, "fold": fold}


def test_explicit_pivot_pair_and_left_right_equivalence():
    # explicit {axis: position} mapping resolves by role ...
    assert _resolve_pivot({"X": "right", "Y": "center"}, "Y", "X") == {
        "seam": "edge",
        "fold": "center",
    }
    # ... and left vs right name the same (edge) sublattice -> identical pivot
    left = _resolve_pivot({"X": "left", "Y": "left"}, "Y", "X")
    right = _resolve_pivot({"X": "right", "Y": "right"}, "Y", "X")
    assert left == right == {"seam": "edge", "fold": "edge"}


# ---------------------------------------------------------------------------
# Grid-level validation
# ---------------------------------------------------------------------------
def test_fold_requires_periodic_seam():
    ds = _make_ds()
    with pytest.raises(ValueError, match="periodic seam axis"):
        Grid(
            ds,
            coords={"X": {"center": "xh"}, "Y": {"center": "yh"}},
            boundary={"X": "fill", "Y": {"fold": "corner"}},
            autoparse_metadata=False,
        )


def test_fold_seam_axis_inferred():
    ds = _make_ds()
    grid = _grid(ds, "corner")
    assert grid._folds["Y"]["seam_axis"] == "X"


def test_bad_pivot_raises():
    ds = _make_ds()
    with pytest.raises(ValueError, match="Unknown fold pivot"):
        _grid(ds, "banana")


# ---------------------------------------------------------------------------
# Per-position halo (explicit expected, the Oceananigans-kernel conventions)
# ---------------------------------------------------------------------------
def test_corner_pivot_all_positions():
    ds = _make_ds()
    grid = _grid(ds, "corner")  # seam=edge, fold=edge

    # scalar c: seam=center -> no roll; fold=center, pivot fold=edge -> no skip
    out = pad(ds.c, grid, boundary_width={"Y": (0, 1)})
    np.testing.assert_allclose(out.isel(yh=-1).values, ds.c.isel(yh=-1).values[::-1])

    # u (vector): seam=edge -> reverse+roll(1); fold=center skip 0; sign flip
    out = pad(
        {"X": ds.u}, grid, boundary_width={"Y": (0, 1)}, other_component={"Y": ds.v}
    )
    np.testing.assert_allclose(
        out.isel(yh=-1).values, -np.roll(ds.u.isel(yh=-1).values[::-1], 1)
    )

    # v (vector): seam=center -> no roll; fold=edge skip 1; sign flip
    out = pad(
        {"Y": ds.v}, grid, boundary_width={"Y": (0, 1)}, other_component={"X": ds.u}
    )
    np.testing.assert_allclose(out.isel(yl=-1).values, -ds.v.isel(yl=-2).values[::-1])

    # q (scalar corner): seam=edge roll(1); fold=edge skip 1
    out = pad(ds.q, grid, boundary_width={"Y": (0, 1)})
    np.testing.assert_allclose(
        out.isel(yl=-1).values, np.roll(ds.q.isel(yl=-2).values[::-1], 1)
    )


def test_u_pivot_redundant_row():
    ds = _make_ds()
    grid = _grid(ds, "U")  # seam=edge, fold=center

    # tracer fold=center == pivot fold -> skip the duplicated top row
    out = pad(ds.c, grid, boundary_width={"Y": (0, 1)})
    np.testing.assert_allclose(out.isel(yh=-1).values, ds.c.isel(yh=-2).values[::-1])
    # v fold=edge != pivot fold -> no skip (sources the top row)
    out = pad(ds.v, grid, boundary_width={"Y": (0, 1)})
    np.testing.assert_allclose(out.isel(yl=-1).values, ds.v.isel(yl=-1).values[::-1])


# ---------------------------------------------------------------------------
# Independent geometric consistency: tracer (center) and u (edge) must mirror
# about the SAME physical pole line. We sample a generic field of physical x and
# check the fold halo equals that field evaluated at the mirrored location.
# ---------------------------------------------------------------------------
def test_center_and_edge_mirror_same_pole():
    ds = xr.Dataset(
        coords={"xh": np.arange(Nx), "xl": np.arange(Nx), "yh": np.arange(Ny)}
    )
    # physical seam coordinate: center i at i+0.5, left(edge) i at i
    F = lambda x: np.sin(2 * np.pi * x / Nx) + 0.3 * np.cos(6 * np.pi * x / Nx)
    xc = np.arange(Nx) + 0.5
    xe = np.arange(Nx).astype(float)
    # constant in y, so the top row carries the F profile
    ds["c"] = (("yh", "xh"), np.tile(F(xc), (Ny, 1)))
    ds["u"] = (("yh", "xl"), np.tile(F(xe), (Ny, 1)))
    grid = Grid(
        ds,
        coords={"X": {"center": "xh", "left": "xl"}, "Y": {"center": "yh"}},
        boundary={"X": "periodic", "Y": {"fold": "corner"}},
        autoparse_metadata=False,
    )
    # corner pivot -> pole at x=0 -> mirror(x) = -x (periodic on [0, Nx))
    halo_c = pad(ds.c, grid, boundary_width={"Y": (0, 1)}).isel(yh=-1).values
    halo_u = pad(ds.u, grid, boundary_width={"Y": (0, 1)}).isel(yh=-1).values
    np.testing.assert_allclose(halo_c, F((-xc) % Nx), atol=1e-12)
    np.testing.assert_allclose(halo_u, F((-xe) % Nx), atol=1e-12)


def test_outer_symmetric_memory():
    # MOM6 "symmetric" memory uses the `outer` position (length N+1) for the
    # cell-edge dims xq/yq. Fold must mirror those about the pole correctly
    # despite the duplicated periodic endpoint.
    ds = xr.Dataset(
        coords={
            "xh": np.arange(Nx),
            "xq": np.arange(Nx + 1),
            "yh": np.arange(Ny),
            "yq": np.arange(Ny + 1),
        }
    )
    # corner pivot -> X pole at x=0 (mirror -x mod Nx), Y pole at top edge y=Ny
    F = lambda x, y: np.sin(2 * np.pi * x / Nx) + 0.5 * y
    xq, yq = np.arange(Nx + 1), np.arange(Ny + 1)
    ds["q"] = (("yq", "xq"), F(xq[None, :], yq[:, None]))  # both outer
    ds["v"] = (("yq", "xh"), F((np.arange(Nx) + 0.5)[None, :], yq[:, None]))
    grid = Grid(
        ds,
        coords={
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
        },
        boundary={"X": "periodic", "Y": {"fold": "corner"}},
        autoparse_metadata=False,
    )
    xc = np.arange(Nx) + 0.5
    halo_v = pad(ds.v, grid, boundary_width={"Y": (0, 1)}).isel(yq=-1).values
    np.testing.assert_allclose(halo_v, F((-xc) % Nx, Ny - 1), atol=1e-12)
    halo_q = pad(ds.q, grid, boundary_width={"Y": (0, 1)}).isel(yq=-1).values
    np.testing.assert_allclose(
        halo_q, np.array([F((-j) % Nx, Ny - 1) for j in range(Nx + 1)]), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Vector sign: scalar does not flip, vector does
# ---------------------------------------------------------------------------
def test_vector_flips_scalar_does_not():
    ds = _make_ds()
    grid = _grid(ds, "corner")
    # same array, once as scalar, once as a vector component
    scal = pad(ds.v, grid, boundary_width={"Y": (0, 1)}).isel(yl=-1).values
    vec = (
        pad(
            {"Y": ds.v}, grid, boundary_width={"Y": (0, 1)}, other_component={"X": ds.u}
        )
        .isel(yl=-1)
        .values
    )
    np.testing.assert_allclose(vec, -scal)


# ---------------------------------------------------------------------------
# End-to-end operators + dask
# ---------------------------------------------------------------------------
def test_diff_across_seam_runs():
    ds = _make_ds()
    grid = _grid(ds, "corner")
    # left->center diff pads the north edge -> exercises the fold
    d = grid.diff(ds.q, "Y")  # q lives on yl -> shifts to yh (center), pads north
    assert d.dims == ("yh", "xl")
    assert np.isfinite(d.values).all()


def test_interp_diff_across_seam_known_answer():
    """`interp` and `diff` across the north fold must equal the same operation on
    a hand-folded (pure-numpy) halo -- a known-answer check that the *operators*,
    not just `pad`, are correct across the seam, for both a scalar and a vector.

    ``v`` lives on ``(yl, xh)``; a left->center shift in Y pads the north edge, so
    the top output row straddles the last interior row and the folded halo row.
    For the ``corner`` pivot a field at this position folds by: mirror in x
    (``[::-1]``, seam=center), skip the redundant seam row (fold=edge -> source
    ``yl=-2``), and -- only as a vector -- flip sign. We build that halo by hand
    and finite-difference/average against it.
    """
    ds = _make_ds()
    grid = _grid(ds, "corner")
    v = ds.v.values  # (Ny, Nx), on (yl, xh)

    def expect(halo_row):
        vp = np.vstack([v, halo_row[None, :]])  # append halo as yl=Ny
        return 0.5 * (vp[:-1] + vp[1:]), vp[1:] - vp[:-1]  # interp(yh), diff(yh)

    # scalar: mirror + skip, no sign change
    exp_i_s, exp_d_s = expect(v[-2][::-1])
    np.testing.assert_allclose(grid.interp(ds.v, "Y").values, exp_i_s)
    np.testing.assert_allclose(grid.diff(ds.v, "Y").values, exp_d_s)

    # vector: same mirror + skip, plus the 180deg sign flip
    exp_i_v, exp_d_v = expect(-v[-2][::-1])
    oc = {"X": ds.u}
    np.testing.assert_allclose(
        grid.interp({"Y": ds.v}, "Y", other_component=oc).values, exp_i_v
    )
    np.testing.assert_allclose(
        grid.diff({"Y": ds.v}, "Y", other_component=oc).values, exp_d_v
    )


def test_multi_row_halo():
    ds = _make_ds()
    grid = _grid(ds, "corner")  # center field: skip 0, no roll
    out = pad(ds.c, grid, boundary_width={"Y": (0, 2)})
    # consecutive halo rows source consecutive interior rows from the top down
    np.testing.assert_allclose(out.isel(yh=-2).values, ds.c.isel(yh=-1).values[::-1])
    np.testing.assert_allclose(out.isel(yh=-1).values, ds.c.isel(yh=-2).values[::-1])


def test_fold_with_simultaneous_seam_padding():
    ds = _make_ds()
    grid = _grid(ds, "corner")
    out = pad(ds.c, grid, boundary_width={"X": (1, 1), "Y": (0, 1)})
    assert out.shape == (Ny + 1, Nx + 2)
    # the folded north row is itself wrapped periodically along the seam axis
    base = ds.c.isel(yh=-1).values[::-1]
    expected = np.concatenate([[base[-1]], base, [base[0]]])
    np.testing.assert_allclose(out.isel(yh=-1).values, expected)


@pytest.mark.parametrize("chunks", [{"xh": Nx, "xl": Nx}, {"xh": 3, "xl": 3}])
def test_fold_dask_matches_numpy(chunks):
    ds = _make_ds()
    grid_np = _grid(ds, "corner")
    expected = pad(ds.c, grid_np, boundary_width={"Y": (0, 1)})

    dsc = ds.chunk(chunks)
    grid_da = _grid(dsc, "corner")
    out = pad(dsc.c, grid_da, boundary_width={"Y": (0, 1)})
    import dask

    assert dask.is_dask_collection(out.data)
    np.testing.assert_allclose(out.compute().values, expected.values)
