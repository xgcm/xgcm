import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from xgcm.grid import Grid
from xgcm.grid_ufunc import _parse_grid_ufunc_signature, as_grid_ufunc, grid_ufunc


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
            "x_i": (
                [
                    "x_i",
                ],
                np.arange(1.5, 9),
            ),
        }
    )

    return Grid(grid_ds, coords={"X": {"center": "x_c", "left": "x_g", "inner": "x_i"}})


# TODO parametrize with using/not using dask
class TestGridUFunc:
    @pytest.mark.xfail
    def test_input_on_wrong_positions(self):
        grid = create_1d_test_grid()
        da = np.sin(grid._ds.x_g * 2 * np.pi / 9)

        # TODO raise more informative error
        with pytest.raises(ValueError):
            grid_ufunc(lambda x: x, grid, "(X:center)->()", da)

    def test_1d_unchanging_size_no_dask(self):
        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        grid = create_1d_test_grid()
        da = np.sin(grid._ds.x_c * 2 * np.pi / 9)
        da.coords["x_c"] = grid._ds.x_c

        diffed = (da - da.roll(x_c=-1, roll_coords=False)).data
        expected = xr.DataArray(diffed, dims=["x_g"], coords={"x_g": grid._ds.x_g})

        # Test direct application
        result = grid_ufunc(
            diff_center_to_left, da, grid=grid, signature="(X:center)->(X:left)"
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc(grid, "(X:center)->(X:left)")
        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        result = diff_center_to_left(da)
        assert_equal(result, expected)

    def test_1d_changing_size_no_dask(self):
        def interp_center_to_inner(a):
            return 0.5 * (a[:-1] + a[1:])

        grid = create_1d_test_grid()
        da = xr.DataArray(np.arange(10, 19), dims=["x_c"], coords={"x_c": grid._ds.x_c})

        expected = da.interp(x_c=np.arange(1.5, 9), method="linear").rename(x_c="x_i")

        # Test direct application
        result = grid_ufunc(
            interp_center_to_inner, da, grid=grid, signature="(X:center)->(X:inner)"
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc(grid, "(X:center)->(X:inner)")
        def interp_center_to_inner(a):
            return 0.5 * (a[:-1] + a[1:])

        result = interp_center_to_inner(da)
        assert_equal(result, expected)

    def test_1d_dask_overlap(self):
        from dask.array import map_overlap

        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        def diff_overlap(a):
            return map_overlap(diff_center_to_left, a, depth=1, boundary=0)

        grid = create_1d_test_grid()
        da = np.sin(grid._ds.x_c * 2 * np.pi / 9).chunk(1)
        da.coords["x_c"] = grid._ds.x_c

        diffed = (da - da.roll(x_c=-1, roll_coords=False)).data
        expected = xr.DataArray(diffed, dims=["x_g"], coords={"x_g": grid._ds.x_g})

        # Test direct application
        result = grid_ufunc(
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
            return map_overlap(diff_center_to_left, a, depth=1, boundary=0)

        result = diff_overlap(da).compute()
        assert_equal(result, expected)

    def test_multiple_inputs(self):
        ...

    def test_multiple_outputs(self):
        ...
