import re

import dask.array  # type: ignore
import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from xgcm.grid import Grid, _select_grid_ufunc
from xgcm.grid_ufunc import (
    GridUFunc,
    _parse_grid_ufunc_signature,
    _signatures_equivalent,
    apply_as_grid_ufunc,
    as_grid_ufunc,
)


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


def create_1d_test_grid_ds(ax_name, length):

    grid_ds = xr.Dataset(
        coords={
            f"{ax_name}_c": (
                [
                    f"{ax_name}_c",
                ],
                np.arange(1, length + 1),
            ),
            f"{ax_name}_g": (
                [
                    f"{ax_name}_g",
                ],
                np.arange(0.5, length),
            ),
            f"{ax_name}_r": (
                [
                    f"{ax_name}_r",
                ],
                np.arange(1.5, length + 1),
            ),
            f"{ax_name}_i": (
                [
                    f"{ax_name}_i",
                ],
                np.arange(1.5, length),
            ),
            f"{ax_name}_o": (
                [
                    f"{ax_name}_o",
                ],
                np.arange(0.5, 10),
            ),
        }
    )

    return grid_ds


def create_1d_test_grid(ax_name, length=9):
    grid_ds = create_1d_test_grid_ds(ax_name, length)
    return Grid(
        grid_ds,
        coords={
            f"{ax_name}": {
                "center": f"{ax_name}_c",
                "left": f"{ax_name}_g",
                "right": f"{ax_name}_r",
                "inner": f"{ax_name}_i",
                "outer": f"{ax_name}_o",
            }
        },
    )


def create_2d_test_grid(ax_name_1, ax_name_2, length1=9, length2=11):
    grid_ds_1 = create_1d_test_grid_ds(ax_name_1, length1)
    grid_ds_2 = create_1d_test_grid_ds(ax_name_2, length2)

    return Grid(
        ds=xr.merge([grid_ds_1, grid_ds_2]),
        coords={
            f"{ax_name_1}": {
                "center": f"{ax_name_1}_c",
                "left": f"{ax_name_1}_g",
                "right": f"{ax_name_1}_r",
                "inner": f"{ax_name_1}_i",
                "outer": f"{ax_name_1}_o",
            },
            f"{ax_name_2}": {
                "center": f"{ax_name_2}_c",
                "left": f"{ax_name_2}_g",
                "right": f"{ax_name_2}_r",
                "inner": f"{ax_name_2}_i",
                "outer": f"{ax_name_2}_o",
            },
        },
    )


class TestGridUFunc:
    def test_stores_ufunc_kwarg_info(self):
        signature = "(X:center)->(X:left)"

        @as_grid_ufunc(signature)
        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        assert isinstance(diff_center_to_left, GridUFunc)
        assert diff_center_to_left.signature == signature

        with pytest.raises(TypeError, match="Unsupported keyword argument"):

            @as_grid_ufunc(signature, junk="useless")
            def diff_center_to_left(a):
                return a - np.roll(a, shift=-1)

    # TODO change test so that this passes
    @pytest.mark.xfail(reason="changed the test fixture")
    def test_input_on_wrong_positions(self):
        grid = create_1d_test_grid("depth")
        grid._ds.drop_vars("depth_o")
        da = np.sin(grid._ds.depth_g * 2 * np.pi / 9)

        with pytest.raises(ValueError, match=re.escape("(depth:outer) does not exist")):
            apply_as_grid_ufunc(
                lambda x: x, da, axis=[("depth",)], grid=grid, signature="(X:outer)->()"
            )

        with pytest.raises(ValueError, match="coordinate depth_c does not appear"):
            apply_as_grid_ufunc(
                lambda x: x,
                da,
                axis=[("depth",)],
                grid=grid,
                signature="(X:center)->()",
            )

    def test_1d_unchanging_size_no_dask(self):
        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        grid = create_1d_test_grid("depth")
        da = np.sin(grid._ds.depth_c * 2 * np.pi / 9)
        da.coords["depth_c"] = grid._ds.depth_c

        diffed = (da - da.roll(depth_c=-1, roll_coords=False)).data
        expected = xr.DataArray(
            diffed, dims=["depth_g"], coords={"depth_g": grid._ds.depth_g}
        )

        # Test direct application
        result = apply_as_grid_ufunc(
            diff_center_to_left,
            da,
            axis=[("depth",)],
            grid=grid,
            signature="(X:center)->(X:left)",
        )
        assert_equal(result, expected)

        # Test Grid method
        result = grid.apply_as_grid_ufunc(
            diff_center_to_left, da, axis=[("depth",)], signature="(X:center)->(X:left)"
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc("(X:center)->(X:left)")
        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        result = diff_center_to_left(grid, da, axis=[("depth",)])
        assert_equal(result, expected)

    def test_1d_unchanging_size_but_padded_dask_parallelized(self):
        """
        This test checks that the process of padding a non-chunked core dimension doesn't turn it into a chunked core
        dimension. See GH #430.
        """

        def diff_center_to_left(a):
            return a[..., 1:] - a[..., :-1]

        grid = create_1d_test_grid("depth")
        da = np.sin(grid._ds.depth_c * 2 * np.pi / 9).chunk()
        da.coords["depth_c"] = grid._ds.depth_c

        diffed = (da - da.roll(depth_c=1, roll_coords=False)).data
        expected = xr.DataArray(
            diffed, dims=["depth_g"], coords={"depth_g": grid._ds.depth_g}
        ).compute()

        # Test direct application
        result = apply_as_grid_ufunc(
            diff_center_to_left,
            da,
            axis=[("depth",)],
            grid=grid,
            signature="(X:center)->(X:left)",
            boundary_width={"X": (1, 0)},
            dask="parallelized",
        ).compute()
        assert_equal(result, expected)

        # Test Grid method
        result = grid.apply_as_grid_ufunc(
            diff_center_to_left,
            da,
            axis=[("depth",)],
            signature="(X:center)->(X:left)",
            boundary_width={"X": (1, 0)},
            dask="parallelized",
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc(
            "(X:center)->(X:left)",
            boundary_width={"X": (1, 0)},
            dask="parallelized",
        )
        def diff_center_to_left(a):
            return a[..., 1:] - a[..., :-1]

        result = diff_center_to_left(
            grid,
            da,
            axis=[("depth",)],
        ).compute()
        assert_equal(result, expected)

    def test_1d_changing_size_dask_parallelized(self):
        def interp_center_to_inner(a):
            return 0.5 * (a[:-1] + a[1:])

        grid = create_1d_test_grid("depth")
        da = xr.DataArray(
            np.arange(10, 19), dims=["depth_c"], coords={"depth_c": grid._ds.depth_c}
        ).chunk()

        expected = da.interp(depth_c=np.arange(1.5, 9), method="linear").rename(
            depth_c="depth_i"
        )

        # Test direct application
        result = apply_as_grid_ufunc(
            interp_center_to_inner,
            da,
            axis=[("depth",)],
            grid=grid,
            signature="(X:center)->(X:inner)",
            dask="parallelized",
        ).compute()
        assert_equal(result, expected)

        # Test Grid method
        result = grid.apply_as_grid_ufunc(
            interp_center_to_inner,
            da,
            axis=[("depth",)],
            signature="(X:center)->(X:inner)",
            dask="parallelized",
        ).compute()
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc("(X:center)->(X:inner)", dask="parallelized")
        def interp_center_to_inner(a):
            return 0.5 * (a[:-1] + a[1:])

        result = interp_center_to_inner(grid, da, axis=[("depth",)]).compute()
        assert_equal(result, expected)

    def test_1d_overlap_dask_allowed(self):
        from dask.array import map_overlap

        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1)

        def diff_overlap(a):
            return map_overlap(diff_center_to_left, a, depth=1, boundary="periodic")

        grid = create_1d_test_grid("depth")
        da = np.sin(grid._ds.depth_c * 2 * np.pi / 9).chunk(1)
        da.coords["depth_c"] = grid._ds.depth_c

        diffed = (da - da.roll(depth_c=-1, roll_coords=False)).data
        expected = xr.DataArray(
            diffed, dims=["depth_g"], coords={"depth_g": grid._ds.depth_g}
        ).compute()

        # Test direct application
        result = apply_as_grid_ufunc(
            diff_overlap,
            da,
            axis=[("depth",)],
            grid=grid,
            signature="(X:center)->(X:left)",
            dask="allowed",
        ).compute()
        assert_equal(result, expected)

        # Test Grid method
        result = grid.apply_as_grid_ufunc(
            diff_overlap,
            da,
            axis=[("depth",)],
            signature="(X:center)->(X:left)",
            dask="allowed",
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc("(X:center)->(X:left)", dask="allowed")
        def diff_overlap(a):
            return map_overlap(diff_center_to_left, a, depth=1, boundary="periodic")

        result = diff_overlap(
            grid,
            da,
            axis=[("depth",)],
        ).compute()
        assert_equal(result, expected)

    def test_apply_along_one_axis(self):
        grid = create_2d_test_grid("lon", "lat")

        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1, axis=-1)

        da = grid._ds.lat_c ** 2 + grid._ds.lon_c ** 2

        diffed = (da - da.roll(lon_c=-1, roll_coords=False)).data
        expected = xr.DataArray(
            diffed,
            dims=["lat_c", "lon_g"],
            coords={"lat_c": grid._ds.lat_c, "lon_g": grid._ds.lon_g},
        )

        # Test direct application
        result = apply_as_grid_ufunc(
            diff_center_to_left,
            da,
            axis=[("lon",)],
            grid=grid,
            signature="(X:center)->(X:left)",
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc("(X:center)->(X:left)")
        def diff_center_to_left(a):
            return a - np.roll(a, shift=-1, axis=-1)

        result = diff_center_to_left(grid, da, axis=[("lon",)])
        assert_equal(result, expected)

    # TODO test a function with padding

    def test_multiple_inputs(self):
        def inner_product_left_right(a, b):
            return np.inner(a, b)

        grid = create_1d_test_grid("depth")
        a = np.sin(grid._ds.depth_g * 2 * np.pi / 9)
        a.coords["depth_g"] = grid._ds.depth_g
        b = np.cos(grid._ds.depth_r * 2 * np.pi / 9)
        b.coords["depth_r"] = grid._ds.depth_r

        expected = xr.DataArray(np.inner(a, b))

        # Test direct application
        result = apply_as_grid_ufunc(
            inner_product_left_right,
            a,
            b,
            axis=[("depth",), ("depth",)],
            grid=grid,
            signature="(X:left),(X:right)->()",
        )
        assert_equal(result, expected)

        # Test Grid method
        result = grid.apply_as_grid_ufunc(
            inner_product_left_right,
            a,
            b,
            axis=[("depth",), ("depth",)],
            signature="(X:left),(X:right)->()",
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc("(X:left),(X:right)->()")
        def inner_product_left_right(a, b):
            return np.inner(a, b)

        result = inner_product_left_right(grid, a, b, axis=[("depth",), ("depth",)])
        assert_equal(result, expected)

    def test_multiple_outputs(self):
        def diff_center_to_inner(a, axis):
            result = a - np.roll(a, shift=1, axis=axis)
            return np.delete(result, 0, axis)  # remove first element along axis

        def grad_to_inner(a):
            return diff_center_to_inner(a, axis=0), diff_center_to_inner(a, axis=1)

        grid = create_2d_test_grid("lon", "lat")

        a = grid._ds.lon_c ** 2 + grid._ds.lat_c ** 2

        expected_u = 2 * grid._ds.lon_i.expand_dims(dim={"lat_c": len(grid._ds.lat_c)})
        expected_u.coords["lat_c"] = grid._ds.lat_c
        expected_v = 2 * grid._ds.lat_i.expand_dims(dim={"lon_c": len(grid._ds.lon_c)})
        expected_v.coords["lon_c"] = grid._ds.lon_c

        # Test direct application
        u, v = apply_as_grid_ufunc(
            grad_to_inner,
            a,
            axis=[("lon", "lat")],
            grid=grid,
            signature="(X:center,Y:center)->(X:inner,Y:center),(X:center,Y:inner)",
        )
        assert_equal(u.T, expected_u)
        assert_equal(v, expected_v)

        # Test Grid method
        u, v = grid.apply_as_grid_ufunc(
            grad_to_inner,
            a,
            axis=[("lon", "lat")],
            signature="(X:center,Y:center)->(X:inner,Y:center),(X:center,Y:inner)",
        )
        assert_equal(u.T, expected_u)
        assert_equal(v, expected_v)

        # Test decorator
        @as_grid_ufunc("(X:center,Y:center)->(X:inner,Y:center),(X:center,Y:inner)")
        def grad_to_inner(a):
            return diff_center_to_inner(a, axis=0), diff_center_to_inner(a, axis=1)

        u, v = grad_to_inner(grid, a, axis=[("lon", "lat")])
        assert_equal(u.T, expected_u)
        assert_equal(v, expected_v)


class TestDaskNoOverlap:
    def test_chunked_non_core_dims(self):
        # Create 2D test data
        ...

    def test_chunked_core_dims_overlap_turned_off(self):
        ...


class TestDaskOverlap:
    def test_chunked_core_dims_unchanging_chunksize(self):
        def diff_center_to_left(a):
            return a[..., 1:] - a[..., :-1]

        grid = create_1d_test_grid("depth")
        da = np.sin(grid._ds.depth_c * 2 * np.pi / 9).chunk(3)
        da.coords["depth_c"] = grid._ds.depth_c

        diffed = (da - da.roll(depth_c=1, roll_coords=False)).data
        expected = xr.DataArray(
            diffed, dims=["depth_g"], coords={"depth_g": grid._ds.depth_g}
        ).compute()

        # Test direct application
        result = apply_as_grid_ufunc(
            diff_center_to_left,
            da,
            axis=[("depth",)],
            grid=grid,
            signature="(X:center)->(X:left)",
            boundary_width={"X": (1, 0)},
            dask="allowed",
            map_overlap=True,
        ).compute()
        assert_equal(result, expected)

        # Test Grid method
        result = grid.apply_as_grid_ufunc(
            diff_center_to_left,
            da,
            axis=[("depth",)],
            signature="(X:center)->(X:left)",
            boundary_width={"X": (1, 0)},
            dask="allowed",
            map_overlap=True,
        )
        assert_equal(result, expected)

        # Test decorator
        @as_grid_ufunc(
            "(X:center)->(X:left)",
            boundary_width={"X": (1, 0)},
            dask="allowed",
            map_overlap=True,
        )
        def diff_center_to_left(a):
            return a[..., 1:] - a[..., :-1]

        result = diff_center_to_left(
            grid,
            da,
            axis=[("depth",)],
        ).compute()
        assert_equal(result, expected)

    @pytest.mark.xfail
    def test_num_tasks_regression(self):
        # Assert numbr of tasks in optimized graph is <= some hardcoded number
        # Obtain that number from the old performance initially
        raise NotImplementedError

    @pytest.mark.xfail
    def test_gave_axis_but_no_corresponding_boundary_width(self):
        # TODO this should default to zero
        raise NotImplementedError

    def test_zero_width_boundary(self):
        def increment(x):
            """Mocking up a function which can only act on in-memory arrays, and requires no padding"""
            if isinstance(x, np.ndarray):
                return np.add(x, 1)
            else:
                raise TypeError

        grid = create_1d_test_grid("depth")
        a = np.sin(grid._ds.depth_g * 2 * np.pi / 9).chunk(2)
        a.coords["depth_g"] = grid._ds.depth_g

        expected = a + 1
        result = apply_as_grid_ufunc(
            increment,
            a,
            axis=[("depth",)],
            grid=grid,
            signature="(X:left)->(X:left)",
            boundary_width=None,
            dask="allowed",
            map_overlap=True,
        ).compute()
        assert_equal(result, expected)

        # in this case the result should be the same as using just map_blocks
        expected_data = dask.array.map_blocks(increment, a.data)
        np.testing.assert_equal(result.data, expected_data)

    @pytest.mark.skip
    def test_only_some_core_dims_are_chunked(self):
        raise NotImplementedError

    def test_raise_when_ufunc_changes_chunksize(self):
        @as_grid_ufunc(
            "(X:outer)->(X:center)",
            boundary_width={"X": (1, 0)},
            dask="allowed",
            map_overlap=True,
        )
        def diff_outer_to_center(a):
            """Mocking up a function which can only act on in-memory arrays, and requires no padding"""
            if isinstance(a, np.ndarray):
                return a[..., 1:] - a[..., :-1]
            else:
                raise TypeError

        grid = create_1d_test_grid("depth")
        da = np.sin(grid._ds.depth_o * 2 * np.pi / 9).chunk(3)
        da.coords["depth_o"] = grid._ds.depth_o

        with pytest.raises(
            NotImplementedError, match="includes one of the axis positions"
        ):
            diff_outer_to_center(
                grid,
                da,
                axis=[("depth",)],
            ).compute()

    def test_multiple_inputs(self):
        @as_grid_ufunc(
            "(X:left),(X:right)->(X:center)",
            boundary_width=None,
            map_overlap=True,
            dask="allowed",
        )
        def multiply_left_right(a, b):
            """Mocking up a function which can only act on in-memory arrays, and requires no padding"""
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return np.multiply(a, b)
            else:
                raise TypeError

        grid = create_1d_test_grid("depth")
        a = np.sin(grid._ds.depth_g * 2 * np.pi / 9).chunk(2)
        a.coords["depth_g"] = grid._ds.depth_g
        b = np.cos(grid._ds.depth_r * 2 * np.pi / 9).chunk(2)
        b.coords["depth_r"] = grid._ds.depth_r

        depth_c_coord = xr.DataArray(np.arange(1, 10), dims="depth_c")
        expected = xr.DataArray(
            np.multiply(a.data, b.data),
            dims=["depth_c"],
            coords={"depth_c": depth_c_coord},
        )

        result = multiply_left_right(grid, a, b, axis=[("depth",), ("depth",)])
        assert_equal(result, expected)

    def test_multiple_outputs(self):
        def diff_center_to_inner(a, axis):
            result = a - np.roll(a, shift=1, axis=axis)
            return np.delete(result, 0, axis)  # remove first element along axis

        def grad_to_inner(a):
            return diff_center_to_inner(a, axis=0), diff_center_to_inner(a, axis=1)

        grid = create_2d_test_grid("lon", "lat")

        a = (grid._ds.lon_c ** 2 + grid._ds.lat_c ** 2).chunk(1)

        # Test direct application
        with pytest.raises(NotImplementedError, match="multiple outputs"):
            apply_as_grid_ufunc(
                grad_to_inner,
                a,
                axis=[("lon", "lat")],
                grid=grid,
                signature="(X:center,Y:center)->(X:inner,Y:center),(X:center,Y:inner)",
                map_overlap=True,
                dask="allowed",
            )


# TODO tests for handling dask in gri.diff etc. should eventually live in test_grid.py
def test_grid_diff_center_to_right_1d():
    grid = create_1d_test_grid("depth")
    da = np.sin(grid._ds.depth_c * 2 * np.pi / 9).chunk(1)
    da.coords["depth_c"] = grid._ds.depth_c

    diffed = (da.roll(depth_c=-1, roll_coords=False) - da).data
    expected = xr.DataArray(
        diffed, dims=["depth_r"], coords={"depth_r": grid._ds.depth_r}
    ).compute()

    result = grid.diff(da, axis="depth", to="right").compute()
    assert_equal(result, expected)


class TestSignaturesEquivalent:
    def test_equivalent(self):
        sig1 = "(X:center)->(X:left)"
        sig2 = "(X:center)->(X:left)"
        assert _signatures_equivalent(sig1, sig2)

        sig3 = "(Y:center)->(Y:left)"
        assert _signatures_equivalent(sig1, sig3)

    def test_not_equivalent(self):
        sig1 = "(X:center)->(X:left)"
        sig2 = "(X:center)->(X:center)"
        assert not _signatures_equivalent(sig1, sig2)

        sig3 = "(X:center)->(Y:left)"
        assert not _signatures_equivalent(sig1, sig3)

        sig4 = "(X:center,X:center)->(X:left)"
        assert not _signatures_equivalent(sig1, sig4)

    def test_no_indices(self):
        sig = "()->()"
        assert _signatures_equivalent(sig, sig)


class GridOpsMockUp:
    """
    Container that stores some mocked-up grid ufuncs to look through.
    Intended to be used as if it were the gridops.py module file.
    """

    @staticmethod
    @as_grid_ufunc(signature="(X:center)->(X:left)")
    def diff_center_to_left(a):
        return a - np.roll(a, -1)

    @staticmethod
    @as_grid_ufunc(signature="(X:center)->(X:right)")
    def diff_center_to_right_fill(a):
        return np.roll(a, 1) - a

    @staticmethod
    @as_grid_ufunc(signature="(X:center)->(X:right)")
    def diff_center_to_right_extend(a):
        return np.roll(a, 1) - a

    @staticmethod
    @as_grid_ufunc(signature="()->()")
    def pass_through_kwargs(**kwargs):
        return kwargs


class TestGridUFuncDispatch:
    def test_select_ufunc(self):
        gridufunc, _ = _select_grid_ufunc(
            "diff", "(X:center)->(X:left)", module=GridOpsMockUp
        )
        assert gridufunc is GridOpsMockUp.diff_center_to_left

    def test_select_ufunc_equivalent_signature(self):
        gridufunc, _ = _select_grid_ufunc(
            "diff", "(Y:center)->(Y:left)", module=GridOpsMockUp
        )
        assert gridufunc is GridOpsMockUp.diff_center_to_left

        with pytest.raises(NotImplementedError):
            _select_grid_ufunc("diff", "(X:center)->(Y:left)", module=GridOpsMockUp)

    def test_select_ufunc_wrong_signature(self):
        with pytest.raises(NotImplementedError):
            _select_grid_ufunc("diff", "(X:center)->(X:center)", module=GridOpsMockUp)

    @pytest.mark.xfail(reason="currently no need for this")
    def test_select_ufunc_by_kwarg(self):
        gridufunc, _ = _select_grid_ufunc(
            "diff", "(X:center)->(X:right)", module=GridOpsMockUp, boundary="fill"
        )
        assert gridufunc is GridOpsMockUp.diff_center_to_right_fill

        with pytest.raises(NotImplementedError):
            _select_grid_ufunc(
                "diff",
                "(X:center)->(X:right)",
                module=GridOpsMockUp,
                boundary="nonsense",
            )

    @pytest.mark.xfail
    def test_pass_through_other_kwargs(self):
        # TODO put this in test_grid.py instead?
        gridufunc, _ = _select_grid_ufunc(
            "pass", "()->()", module=GridOpsMockUp, boundary="fill"
        )
        assert gridufunc(a=1) == {"a": 1}
