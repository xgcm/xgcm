import re

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from xgcm.grid import Grid
from xgcm.grid_ufunc import _parse_grid_ufunc_signature, apply_grid_ufunc, as_grid_ufunc


class TestParseGridUfuncSignature:
    @pytest.mark.parametrize(
        "signature, exp_in_ax_names, exp_out_ax_names, exp_in_ax_pos, exp_out_ax_pos",
        [
            ("()->()", [()], [()], [()], [()]),
            ("(X:center)->()", [("X",)], [()], [("center",)], [()]),
            ("()->(X:left)", [()], [("X",)], [()], [("left",)]),
            ("(X:center)->(X:left)", [("X",)], [("X",)], [("center",)], [("left",)]),
            ("(X:left)->(Y:center)", [("X",)], [("Y",)], [("left",)], [("center",)]),
            ("(X:left)->(Y:center)", [("X",)], [("Y",)], [("left",)], [("center",)]),
            (
                "(X:left),(X:right)->(Y:center)",
                [("X",), ("X",)],
                [("Y",)],
                [("left",), ("right",)],
                [("center",)],
            ),
            (
                "(X:center)->(Y:inner),(Y:outer)",
                [("X",)],
                [("Y",), ("Y",)],
                [("center",)],
                [("inner",), ("outer",)],
            ),
            (
                "(X:center,Y:center)->(Z:center)",
                [("X", "Y")],
                [("Z",)],
                [("center", "center")],
                [("center",)],
            ),
        ],
    )
    def test_parse_valid_signatures(
        self,
        signature,
        exp_in_ax_names,
        exp_out_ax_names,
        exp_in_ax_pos,
        exp_out_ax_pos,
    ):
        in_ax_names, out_ax_names, in_ax_pos, out_ax_pos = _parse_grid_ufunc_signature(
            signature
        )
        assert in_ax_names == exp_in_ax_names
        assert out_ax_names == exp_out_ax_names
        assert in_ax_pos == exp_in_ax_pos
        assert out_ax_pos == exp_out_ax_pos

    @pytest.mark.parametrize(
        "signature",
        [
            "(x:left)(y:left)->()",
            "(x:left),(y:left)->",
            "((x:left))->(x:left)",
            "((x:left))->(x:left)",
            "(x:left)->(x:left)," "(i)->(i)",
            "(X:centre)->()",
        ],
    )
    def test_invalid_signatures(self, signature):
        with pytest.raises(ValueError):
            _parse_grid_ufunc_signature(signature)


def create_1d_test_grid():

    grid_ds = xr.Dataset(
        coords={
            "x_c": (
                [
                    "x_c",
                ],
                np.arange(1, 10),
            ),
            "x_g": (
                [
                    "x_g",
                ],
                np.arange(0.5, 9),
            ),
            "x_r": (
                [
                    "x_r",
                ],
                np.arange(1.5, 10),
            ),
            "x_i": (
                [
                    "x_i",
                ],
                np.arange(1.5, 9),
            ),
        }
    )

    return Grid(
        grid_ds,
        coords={"X": {"center": "x_c", "left": "x_g", "right": "x_r", "inner": "x_i"}},
    )


def create_2d_test_grid():
    # TODO refactor this to combine with create_1d_test_grid
    grid_ds = xr.Dataset(
        coords={
            "x_c": (
                [
                    "x_c",
                ],
                np.arange(1, 10),
            ),
            "x_g": (
                [
                    "x_g",
                ],
                np.arange(0.5, 9),
            ),
            "x_i": (
                [
                    "x_i",
                ],
                np.arange(1.5, 9),
            ),
            "y_c": (
                [
                    "y_c",
                ],
                np.arange(1, 10),
            ),
            "y_g": (
                [
                    "y_g",
                ],
                np.arange(0.5, 9),
            ),
            "y_i": (
                [
                    "y_i",
                ],
                np.arange(1.5, 9),
            ),
        }
    )

    return Grid(
        grid_ds,
        coords={
            "X": {"center": "x_c", "left": "x_g", "inner": "x_i"},
            "Y": {"center": "y_c", "left": "y_g", "inner": "y_i"},
        },
    )


class TestGridUFunc:
    def test_input_on_wrong_positions(self):
        grid = create_1d_test_grid()
        da = np.sin(grid._ds.x_g * 2 * np.pi / 9)

        with pytest.raises(ValueError, match=re.escape("(Y:center) does not exist")):
            apply_grid_ufunc(lambda x: x, da, grid=grid, signature="(Y:center)->()")

        with pytest.raises(ValueError, match="coordinate x_c does not appear"):
            apply_grid_ufunc(lambda x: x, da, grid=grid, signature="(X:center)->()")

    def test_1d_unchanging_size_no_dask(self):
        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        grid = create_1d_test_grid()
        da = np.sin(grid._ds.x_c * 2 * np.pi / 9)
        da.coords["x_c"] = grid._ds.x_c

        diffed = (da - da.roll(x_c=-1, roll_coords=False)).data
        expected = xr.DataArray(diffed, dims=["x_g"], coords={"x_g": grid._ds.x_g})

        # Test direct application
        result = apply_grid_ufunc(
            diff_center_to_left, da, grid=grid, signature="(X:center)->(X:left)"
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc(grid, "(X:center)->(X:left)")
        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        result = diff_center_to_left(da)
        assert_equal(result, expected)

    def test_1d_changing_size_dask_parallelized(self):
        def interp_center_to_inner(a):
            return 0.5 * (a[:-1] + a[1:])

        grid = create_1d_test_grid()
        da = xr.DataArray(
            np.arange(10, 19), dims=["x_c"], coords={"x_c": grid._ds.x_c}
        ).chunk()

        expected = da.interp(x_c=np.arange(1.5, 9), method="linear").rename(x_c="x_i")

        # Test direct application
        result = apply_grid_ufunc(
            interp_center_to_inner,
            da,
            grid=grid,
            signature="(X:center)->(X:inner)",
            dask="parallelized",
        ).compute()
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc(grid, "(X:center)->(X:inner)", dask="parallelized")
        def interp_center_to_inner(a):
            return 0.5 * (a[:-1] + a[1:])

        result = interp_center_to_inner(da).compute()
        assert_equal(result, expected)

    def test_1d_overlap_dask_allowed(self):
        from dask.array import map_overlap

        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        def diff_overlap(a):
            return map_overlap(diff_center_to_left, a, depth=1, boundary="periodic")

        grid = create_1d_test_grid()
        da = np.sin(grid._ds.x_c * 2 * np.pi / 9).chunk(1)
        da.coords["x_c"] = grid._ds.x_c

        diffed = (da - da.roll(x_c=-1, roll_coords=False)).data
        expected = xr.DataArray(
            diffed, dims=["x_g"], coords={"x_g": grid._ds.x_g}
        ).compute()

        # Test direct application
        result = apply_grid_ufunc(
            diff_center_to_left,
            da,
            grid=grid,
            signature="(X:center)->(X:left)",
            dask="allowed",
        ).compute()
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc(grid, "(X:center)->(X:left)", dask="allowed")
        def diff_overlap(a):
            return map_overlap(diff_center_to_left, a, depth=1, boundary="periodic")

        result = diff_overlap(da).compute()
        assert_equal(result, expected)

    def test_multiple_inputs(self):
        def inner_product_left_right(a, b):
            return np.inner(a, b)

        grid = create_1d_test_grid()
        a = np.sin(grid._ds.x_g * 2 * np.pi / 9)
        a.coords["x_g"] = grid._ds.x_g
        b = np.cos(grid._ds.x_r * 2 * np.pi / 9)
        b.coords["x_r"] = grid._ds.x_r

        expected = xr.DataArray(np.inner(a, b))

        # Test direct application
        result = apply_grid_ufunc(
            inner_product_left_right,
            a,
            b,
            grid=grid,
            signature="(X:left),(X:right)->()",
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc(grid, "(X:left),(X:right)->()")
        def inner_product_left_right(a, b):
            return np.inner(a, b)

        result = inner_product_left_right(a, b)
        assert_equal(result, expected)

    def test_multiple_outputs(self):
        def diff_center_to_left(a, axis):
            return a - np.roll(a, shift=-1, axis=axis)

        def grad_to_left(a):
            return diff_center_to_left(a, axis=0), diff_center_to_left(a, axis=1)

        grid = create_2d_test_grid()

        a = grid._ds.x_c ** 2 + grid._ds.y_c ** 2

        expected_u = 2 * grid._ds.x_c
        expected_u = expected_u.swap_dims(x_c="x_g")
        expected_u.coords["x_c"] = grid._ds.x_g
        expected_v = 2 * grid._ds.y_c
        expected_v = expected_v.swap_dims(y_c="y_g")
        expected_v.coords["y_c"] = grid._ds.y_g

        # Test direct application
        u, v = apply_grid_ufunc(
            grad_to_left,
            a,
            grid=grid,
            signature="(X:center,Y:center)->(X:left,Y:center),(X:center,Y:left)",
        )
        assert_equal(u, expected_u)
        assert_equal(v, expected_v)

        # Test decorator
        @as_grid_ufunc(grid, "(X:center,Y:center)->(X:left,Y:center),(X:center,Y:left)")
        def grad_to_left(a):
            return diff_center_to_left(a, axis=0), diff_center_to_left(a, axis=1)

        u, v = grad_to_left(a)
        assert_equal(u, expected_u)
        assert_equal(v, expected_v)
