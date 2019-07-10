from __future__ import print_function
from future.utils import iteritems
import pytest
import xarray as xr
import numpy as np
from xarray.testing import assert_allclose, assert_equal

from xgcm.autogenerate import (
    generate_axis,
    generate_grid_ds,
    _parse_boundary_params,
    _parse_position,
    _position_to_relative,
    _fill_attrs,
)

# create test datasets
pad_value = 1000
dx = 5.0
dy = 2.5
dz = 0.5
a = np.random.rand(int(180 / dy), int(360 / dx), int(10 / dz))
x = np.arange(-180 + dx, 180 + dx, dx)
y = np.arange(-90 + dy, 90 + dy, dy)
z = np.arange(0 + dz, 10 + dz, dz)
x_outer = np.arange(-180 + dx / 2, 180 + dx, dx)
y_outer = np.arange(-90 + dy / 2, 90 + dy, dy)
z_outer = np.arange(0 + dz / 2, 10 + dz, dz)
z_1D = z[0:3]
z_1D_padded = np.hstack([z[0:2] + (dz / 2.0), (z[2] + pad_value) / 2])

xx, yy = np.meshgrid(x, y)
_, _, zz = np.meshgrid(x, y, z)

xx_outer, _ = np.meshgrid(x_outer, y)
_, yy_outer = np.meshgrid(x, y_outer)
_, _, zz_outer = np.meshgrid(x, y, z_outer)

ds_original = xr.Dataset(
    {"somedata": (["lat", "lon", "z"], a)},
    coords={
        "lon": (["lon"], x + (dx / 2.0)),
        "lat": (["lat"], y + (dy / 2.0)),
        "z": (["z"], z + (dz / 2.0)),
        "llon": (["lat", "lon"], xx + (dx / 2.0)),
        "llat": (["lat", "lon"], yy + (dy / 2.0)),
        "zz": (["lat", "lon", "z"], zz + (dz / 2.0)),
    },
)

ds_original_left = xr.Dataset(
    {"somedata": (["lat", "lon", "z"], a)},
    coords={
        "lon": (["lon"], x),
        "lat": (["lat"], y),
        "z": (["z"], z),
        "llon": (["lat", "lon"], xx),
        "llat": (["lat", "lon"], yy),
        "zz": (["lat", "lon", "z"], zz),
    },
)

ds_original_1D = xr.Dataset(
    {"somedata": (["z"], np.array([1, 2, 3]))}, coords={"z": (["z"], z_1D)}
)
ds_original_1D_padded = xr.Dataset(
    {"somedata": (["z"], np.array([1, 2, 3]))},
    coords={"z": (["z"], z[0:3]), "test": (["test"], z_1D_padded)},
)
ds_out_left = xr.Dataset(
    {"somedata": (["lat", "lon", "z"], a)},
    coords={
        "lon": (["lon"], x + (dx / 2.0), {"axis": "X"}),
        "lat": (["lat"], y + (dy / 2.0), {"axis": "Y"}),
        "z": (["z"], z + (dz / 2.0), {"axis": "Z"}),
        "llon": (["lat", "lon"], xx + (dx / 2.0), {"axis": "X"}),
        "llat": (["lat", "lon"], yy + (dy / 2.0), {"axis": "Y"}),
        "zz": (["lat", "lon", "z"], zz + (dz / 2.0), {"axis": "Z"}),
        "lon_left": (["lon_left"], x, {"axis": "X", "c_grid_axis_shift": -0.5}),
        "lat_left": (["lat_left"], y, {"axis": "Y", "c_grid_axis_shift": -0.5}),
        "z_left": (["z_left"], z, {"axis": "Z", "c_grid_axis_shift": -0.5}),
        "llon_left": (
            ["lat", "lon_left"],
            xx,
            {"axis": "X", "c_grid_axis_shift": -0.5},
        ),
        "llat_left": (
            ["lat_left", "lon"],
            yy,
            {"axis": "Y", "c_grid_axis_shift": -0.5},
        ),
        "zz_left": (
            ["lat", "lon", "z_left"],
            zz,
            {"axis": "Z", "c_grid_axis_shift": -0.5},
        ),
    },
)

ds_out_right = xr.Dataset(
    {"somedata": (["lat", "lon", "z"], a)},
    coords={
        "lon": (["lon"], x + (dx / 2.0), {"axis": "X", "c_grid_axis_shift": -0.5}),
        "lat": (["lat"], y + (dy / 2.0), {"axis": "Y", "c_grid_axis_shift": -0.5}),
        "z": (["z"], z + (dz / 2.0), {"axis": "Z", "c_grid_axis_shift": -0.5}),
        "llon": (
            ["lat", "lon"],
            xx + (dx / 2.0),
            {"axis": "X", "c_grid_axis_shift": -0.5},
        ),
        "llat": (
            ["lat", "lon"],
            yy + (dy / 2.0),
            {"axis": "Y", "c_grid_axis_shift": -0.5},
        ),
        "zz": (
            ["lat", "lon", "z"],
            zz + (dz / 2.0),
            {"axis": "Z", "c_grid_axis_shift": -0.5},
        ),
        "lon_right": (["lon_right"], x + dx, {"axis": "X"}),
        "lat_right": (["lat_right"], y + dy, {"axis": "Y"}),
        "z_right": (["z_right"], z + dz, {"axis": "Z"}),
        "llon_right": (["lat", "lon_right"], xx + dx, {"axis": "X"}),
        "llat_right": (["lat_right", "lon"], yy + dy, {"axis": "Y"}),
        "zz_right": (["lat", "lon", "z_right"], zz + dz, {"axis": "Z"}),
    },
)

ds_out_center = xr.Dataset(
    {"somedata": (["lat", "lon", "z"], a)},
    coords={
        "lon": (["lon"], x, {"axis": "X", "c_grid_axis_shift": -0.5}),
        "lat": (["lat"], y, {"axis": "Y", "c_grid_axis_shift": -0.5}),
        "z": (["z"], z, {"axis": "Z", "c_grid_axis_shift": -0.5}),
        "llon": (["lat", "lon"], xx, {"axis": "X", "c_grid_axis_shift": -0.5}),
        "llat": (["lat", "lon"], yy, {"axis": "Y", "c_grid_axis_shift": -0.5}),
        "zz": (["lat", "lon", "z"], zz, {"axis": "Z", "c_grid_axis_shift": -0.5}),
        "lon_center": (["lon_center"], x + (dx / 2.0), {"axis": "X"}),
        "lat_center": (["lat_center"], y + (dy / 2.0), {"axis": "Y"}),
        "z_center": (["z_center"], z + (dz / 2.0), {"axis": "Z"}),
        "llon_center": (["lat", "lon_center"], xx + (dx / 2.0), {"axis": "X"}),
        "llat_center": (["lat_center", "lon"], yy + (dy / 2.0), {"axis": "Y"}),
        "zz_center": (["lat", "lon", "z_center"], zz + (dz / 2.0), {"axis": "Z"}),
    },
)

ds_out_outer = xr.Dataset(
    {"somedata": (["lat", "lon", "z"], a)},
    coords={
        "lon": (["lon"], x, {"axis": "X"}),
        "lat": (["lat"], y, {"axis": "Y"}),
        "z": (["z"], z, {"axis": "Z"}),
        "llon": (["lat", "lon"], xx, {"axis": "X"}),
        "llat": (["lat", "lon"], yy, {"axis": "Y"}),
        "zz": (["lat", "lon", "z"], zz, {"axis": "Z"}),
        "lon_outer": (["lon_outer"], x_outer, {"axis": "X", "c_grid_axis_shift": -0.5}),
        "lat_outer": (["lat_outer"], y_outer, {"axis": "Y", "c_grid_axis_shift": -0.5}),
        "z_outer": (["z_outer"], z_outer, {"axis": "Z", "c_grid_axis_shift": -0.5}),
        "llon_outer": (
            ["lat", "lon_outer"],
            xx_outer,
            {"axis": "X", "c_grid_axis_shift": -0.5},
        ),
        "llat_outer": (
            ["lat_outer", "lon"],
            yy_outer,
            {"axis": "Y", "c_grid_axis_shift": -0.5},
        ),
        "zz_outer": (
            ["lat", "lon", "z_outer"],
            zz_outer,
            {"axis": "Z", "c_grid_axis_shift": -0.5},
        ),
    },
)


def test_generate_axis():
    a = generate_axis(
        ds_original,
        "X",
        "lon",
        "lon",
        pos_from="center",
        pos_to="right",
        pad=None,
        boundary_discontinuity=360,
    )
    b = generate_axis(
        ds_original,
        "Y",
        "lat",
        "lat",
        pos_from="center",
        pos_to="left",
        pad=None,
        boundary_discontinuity=180,
    )
    c = generate_axis(
        ds_original, "Z", "z", "z", pos_from="center", pos_to="left", pad="auto"
    )
    d = generate_axis(
        ds_original_1D,
        "Z",
        "z",
        "z",
        pos_from="left",
        pos_to="center",
        pad=pad_value,
        new_name="test",
    )
    e = generate_axis(
        ds_original_left, "Z", "z", "z", pos_from="left", pos_to="center", pad="auto"
    )
    f = generate_axis(
        ds_original_left, "Z", "z", "z", pos_from="center", pos_to="outer", pad="auto"
    )
    g = generate_axis(
        ds_original_left,
        "X",
        "lon",
        "lon",
        pos_from="center",
        pos_to="outer",
        pad=None,
        boundary_discontinuity=360,
    )

    assert_allclose(a["lon_right"], ds_out_right["lon_right"])
    assert_allclose(b["lat_left"], ds_out_left["lat_left"])
    assert_allclose(c["z_left"], ds_out_left["z_left"])
    assert_allclose(d["test"], ds_original_1D_padded["test"])
    assert_allclose(e["z_center"], ds_out_center["z_center"])
    assert_allclose(f["z_outer"], ds_out_outer["z_outer"])
    assert_allclose(g["lon_outer"], ds_out_outer["lon_outer"])

    # Mulitdim cases
    aa = generate_axis(
        a,
        "X",
        "llon",
        "lon",
        pos_from="center",
        pos_to="right",
        pad=None,
        boundary_discontinuity=360,
        attrs_from_scratch=False,
    )
    bb = generate_axis(
        b,
        "Y",
        "llat",
        "lat",
        pos_from="center",
        pos_to="left",
        pad=None,
        boundary_discontinuity=180,
        attrs_from_scratch=False,
    )
    ee = generate_axis(
        e,
        "Z",
        "zz",
        "z",
        pos_from="left",
        pos_to="center",
        pad="auto",
        attrs_from_scratch=False,
    )
    ff = generate_axis(
        f,
        "Z",
        "zz",
        "z",
        pos_from="center",
        pos_to="outer",
        pad="auto",
        attrs_from_scratch=False,
    )

    gg = generate_axis(
        g,
        "X",
        "llon",
        "lon",
        pos_from="center",
        pos_to="outer",
        pad=None,
        boundary_discontinuity=360,
        attrs_from_scratch=False,
    )

    assert_allclose(aa["llon_right"], ds_out_right["llon_right"])
    assert_allclose(bb["llat_left"], ds_out_left["llat_left"])
    assert_allclose(ee["zz_center"], ds_out_center["zz_center"])
    assert_allclose(ff["zz_outer"], ds_out_outer["zz_outer"])
    assert_allclose(gg["llon_outer"], ds_out_outer["llon_outer"])

    with pytest.raises(ValueError):
        # Check if generate axis fails when a DataArray is passed instead of
        # Dataset
        generate_axis(
            c["somedata"],
            "Z",
            "zz",
            "z",
            pos_from="left",
            pos_to="center",
            pad="auto",
            attrs_from_scratch=False,
        )
    with pytest.raises(ValueError):
        generate_axis(c, "Z", "zz", "z", pad="auto", boundary_discontinuity=360)
    with pytest.raises(ValueError):
        generate_axis(c, "Z", "zz", "z", pad=None, boundary_discontinuity=None)


def test_generate_grid_ds():
    # This needs more cases
    axis_dims = {"X": "lon", "Y": "lat", "Z": "z"}
    axis_coords = {"X": "llon", "Y": "llat", "Z": "zz"}
    position = ("center", "outer")
    boundary_discontinuity = {"lon": 360, "llon": 360, "lat": 180, "llat": 180}
    pad = {"z": "auto", "zz": "auto"}
    ds = generate_grid_ds(
        ds_original_left, axis_dims, axis_coords, position, boundary_discontinuity, pad
    )
    assert_equal(ds, ds_out_outer)


def test_parse_boundary_params():
    assert _parse_boundary_params(360, "anything") == 360
    assert _parse_boundary_params({"something": 360}, "something") == 360
    assert _parse_boundary_params({"something": 360}, "something_else") is None


@pytest.mark.parametrize(
    "p_f, p_t",
    [("left", "center"), ("center", "left"), ("center", "right"), ("right", "center")],
)
def test_parse_position(p_f, p_t):
    default = ("center", "left")
    assert _parse_position((p_f, p_t), "anything") == (p_f, p_t)
    assert _parse_position({"a": (p_f, p_t)}, "a") == (p_f, p_t)
    assert _parse_position({"a": (p_f, p_t)}, "b") == default
    assert _parse_position({"a": (p_f, p_t), "b": (p_t, p_f)}, "a") == (p_f, p_t)
    assert _parse_position({"a": (p_f, p_t), "b": (p_t, p_f)}, "b") == (p_t, p_f)


@pytest.mark.parametrize(
    "p, relative",
    [
        (("left", "center"), "right"),
        (("center", "left"), "left"),
        (("center", "right"), "right"),
        (("right", "center"), "left"),
        (("center", "outer"), "outer"),
        (("center", "inner"), "inner"),
    ],
)
def test_position_to_relative(p, relative):
    assert _position_to_relative(p[0], p[1]) == relative
    error_test = [
        ("left", "right"),
        ("inner", "outer"),
        ("left", "outer"),
        ("right", "outer"),
        ("left", "inner"),
        ("right", "inner"),
    ]
    for et in error_test:
        with pytest.raises(RuntimeError):
            _position_to_relative(et[0], et[1])
        with pytest.raises(RuntimeError):
            _position_to_relative(et[1], et[0])


def test_fill_attrs():
    a = _fill_attrs(ds_out_right["lon"], "right", "X")
    assert a.attrs["axis"] == "X"
    assert a.attrs["c_grid_axis_shift"] == 0.5

    a = _fill_attrs(ds_out_right["lon"], "left", "Z")
    assert a.attrs["axis"] == "Z"
    assert a.attrs["c_grid_axis_shift"] == -0.5

    a = _fill_attrs(ds_out_right["lon"], "center", "Y")
    assert a.attrs["axis"] == "Y"
    assert "c_grid_axis_shift" not in a.attrs.keys()
