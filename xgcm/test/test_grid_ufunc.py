import re
from typing import Annotated, Tuple

import dask.array  # type: ignore
import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from xgcm.grid import Grid, _select_grid_ufunc
from xgcm.grid_ufunc import (
    GridUFunc,
    _GridUFuncSignature,
    _parse_signature_from_string,
    apply_as_grid_ufunc,
    as_grid_ufunc,
)


class TestParseSignatureFromString:
    @pytest.mark.parametrize(
        "sig_str, exp_in_ax_names, exp_in_ax_pos, exp_out_ax_names, exp_out_ax_pos",
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
        sig_str,
        exp_in_ax_names,
        exp_out_ax_names,
        exp_in_ax_pos,
        exp_out_ax_pos,
    ):
        in_ax_names, out_ax_names, in_ax_pos, out_ax_pos = _parse_signature_from_string(
            sig_str
        )
        assert in_ax_names == exp_in_ax_names
        assert in_ax_pos == exp_in_ax_pos
        assert out_ax_names == exp_out_ax_names
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
            _parse_signature_from_string(signature)

    @pytest.mark.parametrize(
        "sig_str",
        [
            "()->()",
            "(X:center)->()",
            "()->(X:left)",
            "(X:center)->(X:left)",
            "(X:left)->(Y:center)",
            "(X:left)->(Y:center)",
            "(X:left),(X:right)->(Y:center)",
            "(X:center)->(Y:inner),(Y:outer)",
            "(X:center,Y:center)->(Z:center)",
        ],
    )
    def test_roundtrip_from_string(self, sig_str):
        """Checks that the __str__ method of the signature class works"""
        sig = _GridUFuncSignature.from_string(sig_str)
        assert str(sig) == sig_str


class TestParseSignatureFromTypeHints:
    def test_no_args_to_annotate(self):
        with pytest.raises(ValueError, match="Must specify axis positions"):

            @as_grid_ufunc()
            def ufunc():
                ...

    # TODO test hints without annotations
    # TODO test hints with annotations that don't conform to Xgcm

    def test_annotated_args(self):
        @as_grid_ufunc()
        def ufunc(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Annotated[np.ndarray, "X:center"]:
            ...

        assert str(ufunc.signature) == "(X:center)->(X:center)"

        @as_grid_ufunc()
        def ufunc(
            a: Annotated[np.ndarray, "X:center,Y:center"]
        ) -> Annotated[np.ndarray, "X:center"]:
            ...

        assert str(ufunc.signature) == "(X:center,Y:center)->(X:center)"

        @as_grid_ufunc()
        def ufunc(
            a: Annotated[np.ndarray, "X:left"],
            b: Annotated[np.ndarray, "Y:right"],
        ) -> Annotated[np.ndarray, "X:center"]:
            ...

        assert str(ufunc.signature) == "(X:left),(Y:right)->(X:center)"

        @as_grid_ufunc()
        def ufunc(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Annotated[np.ndarray, "X:left,Y:right"]:
            ...

        assert str(ufunc.signature) == "(X:center)->(X:left,Y:right)"

        @as_grid_ufunc()
        def ufunc(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Tuple[Annotated[np.ndarray, "X:left"], Annotated[np.ndarray, "Y:right"]]:
            ...

        assert str(ufunc.signature) == "(X:center)->(X:left),(Y:right)"

    @pytest.mark.xfail(reason="signature regex will assume nonsense==no inputs")
    def test_invalid_arg_annotation(self):
        # TODO how to get it to realise this is wrong?

        with pytest.raises(ValueError, match="Not a valid grid ufunc signature"):

            @as_grid_ufunc()
            def ufunc(
                a: Annotated[np.ndarray, "nonsense"]  # type: ignore
            ) -> Annotated[np.ndarray, "X:center"]:
                ...

        with pytest.raises(ValueError, match="Not a valid grid ufunc signature"):

            @as_grid_ufunc()
            def ufunc(
                a: Annotated[np.ndarray, "X:Mars"]
            ) -> Annotated[np.ndarray, "X:center"]:
                ...

    @pytest.mark.xfail(reason="signature regex will assume nonsense==no inputs")
    def test_invalid_return_arg_annotation(self):
        # TODO how to get it to realise this is wrong?

        with pytest.raises(ValueError, match="Not a valid grid ufunc signature"):

            @as_grid_ufunc()
            def ufunc(
                a: Annotated[np.ndarray, "X:center"]
            ) -> Annotated[np.ndarray, "X:Venus"]:
                ...

    def test_both_sig_kwarg_and_hints_given(self):
        with pytest.raises(
            ValueError, match="only one of either type hints or signature kwarg"
        ):

            @as_grid_ufunc(signature="(X:center)->(X:left)")
            def ufunc(
                a: Annotated[np.ndarray, "X:center"]
            ) -> Annotated[np.ndarray, "X:left"]:
                ...

    def test_type_hint_as_numpy_ndarray(self):

        # This should raise a mypy error, which is then ignored
        @as_grid_ufunc()
        def ufunc1(a: Annotated[str, "X:center"]) -> Annotated[np.ndarray, "X:center"]:
            # np.ndarray has a .strides method but str doesn't (and nor does xr.DataArray)
            print(a.strides)  # type: ignore
            return a  # type: ignore

        # This should pass mypy without raising any errors
        @as_grid_ufunc()
        def ufunc3(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Annotated[np.ndarray, "X:center"]:
            print(a.strides)
            return a


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


class TestGridUFuncNoPadding:
    def test_stores_ufunc_kwarg_info(self):
        signature = "(X:center)->(X:left)"

        @as_grid_ufunc()
        def diff_center_to_left(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Annotated[np.ndarray, "X:left"]:
            return a - np.roll(a, shift=-1)

        assert isinstance(diff_center_to_left, GridUFunc)
        assert str(diff_center_to_left.signature) == signature

        with pytest.raises(TypeError, match="Unsupported keyword argument"):

            @as_grid_ufunc(junk="useless")
            def diff_center_to_left(a):
                return a - np.roll(a, shift=-1)

    # TODO change test so that this passes
    @pytest.mark.xfail(reason="changed the test fixture")
    def test_input_on_wrong_positions(self):
        grid = create_1d_test_grid("depth")
        grid._ds.drop_vars("depth_o")
        da = np.sin(grid._ds.depth_g * 2 * np.pi / 9)

        with pytest.raises(
            ValueError,
            match=re.escape("Axis:positions pair depth:outer does not exist"),
        ):
            da: Annotated[np.ndarray, "X:outer"]
            apply_as_grid_ufunc(
                lambda x: x, da, axis=[("depth",)], grid=grid, signature="(X:outer)->()"
            )

        with pytest.raises(ValueError, match="coordinate depth_c does not appear"):
            da: Annotated[np.ndarray, "X:center"]
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
        @as_grid_ufunc()
        def diff_center_to_left(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Annotated[np.ndarray, "X:left"]:
            return a - np.roll(a, shift=-1)

        result = diff_center_to_left(grid, da, axis=[("depth",)])
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
        @as_grid_ufunc(dask="parallelized")
        def interp_center_to_inner(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Annotated[np.ndarray, "X:inner"]:
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
        @as_grid_ufunc(dask="allowed")
        def diff_overlap(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Annotated[np.ndarray, "X:left"]:
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

        da = grid._ds.lat_c**2 + grid._ds.lon_c**2

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
        @as_grid_ufunc()
        def diff_center_to_left(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Annotated[np.ndarray, "X:left"]:
            return a - np.roll(a, shift=-1, axis=-1)

        result = diff_center_to_left(grid, da, axis=[("lon",)])
        assert_equal(result, expected)

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
        @as_grid_ufunc()
        def inner_product_left_right(
            a: Annotated[np.ndarray, "X:left"], b: Annotated[np.ndarray, "X:right"]
        ):
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

        a = grid._ds.lon_c**2 + grid._ds.lat_c**2

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
        @as_grid_ufunc()
        def grad_to_inner(
            a: Annotated[np.ndarray, "X:center,Y:center"]
        ) -> Tuple[
            Annotated[np.ndarray, "X:inner,Y:center"],
            Annotated[np.ndarray, "X:center,Y:inner"],
        ]:
            return diff_center_to_inner(a, axis=0), diff_center_to_inner(a, axis=1)

        u, v = grad_to_inner(grid, a, axis=[("lon", "lat")])
        assert_equal(u.T, expected_u)
        assert_equal(v, expected_v)


class TestGridUfuncWithPadding:
    def test_1d_padded_but_no_change_in_grid_position(self):
        def diff_center_to_center_second_order(a):
            return 0.5 * (a[..., 2:] - a[..., :-2])

        grid = create_1d_test_grid("depth")
        da = np.sin(grid._ds.depth_c * 2 * np.pi / 9)
        da.coords["depth_c"] = grid._ds.depth_c

        diffed = 0.5 * (da - da.roll(depth_c=2, roll_coords=False)).data
        expected = xr.DataArray(
            diffed, dims=["depth_c"], coords={"depth_c": grid._ds.depth_c}
        )

        # Test direct application
        result = apply_as_grid_ufunc(
            diff_center_to_center_second_order,
            da,
            axis=[("depth",)],
            grid=grid,
            signature="(X:center)->(X:center)",
            boundary_width={"X": (2, 0)},
        )
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

    def test_2d_padding(self):
        def diff(a, axis):
            def _diff(a):
                return a[..., 1:] - a[..., :-1]

            return np.apply_along_axis(_diff, axis, a)

        def vort(u, v):
            """This needs to return an array 1 element smaller along both axis -1 & -2."""

            u_trimmed = u[..., 1:, :]
            v_trimmed = v[..., 1:]

            v_diff = diff(v_trimmed, axis=-2)
            u_diff = diff(u_trimmed, axis=-1)
            return v_diff - u_diff

        grid = create_2d_test_grid("lon", "lat")

        U = grid._ds.lon_g**2 + grid._ds.lat_c**3
        V = grid._ds.lon_c**3 + grid._ds.lat_g**2

        diffed_v = (V - V.roll(lon_c=1, roll_coords=False)).data
        diffed_u = (U - U.roll(lat_c=1, roll_coords=False)).data
        expected = xr.DataArray(
            diffed_v - diffed_u,
            dims=["lon_g", "lat_g"],
            coords={"lon_g": grid._ds.lon_g, "lat_g": grid._ds.lat_g},
        ).compute()

        result = grid.apply_as_grid_ufunc(
            vort,
            U,
            V,
            axis=2 * [("lon", "lat")],
            signature="(lon:left,lat:center),(lon:center,lat:left)->(lon:left,lat:left)",
            boundary_width={"lon": (1, 0), "lat": (1, 0)},
            dask="parallelized",  # data isn't chunked along lat/lon
        )
        assert_equal(result, expected)


class TestPadManuallyInsideUfunc:
    """Tests that we can set boundary_wdith=None and instead manually pad inside the applied ufunc."""

    def test_1d_padded_but_no_change_in_grid_position(self):
        def diff_center_to_center_second_order(a):
            b = a[..., 2:]
            c = a[..., :-2]
            return 0.5 * (b - c)

        grid = create_1d_test_grid("depth")
        da = grid._ds.depth_c**2
        da.coords["depth_c"] = grid._ds.depth_c

        diffed = 0.5 * (da - da.roll(depth_c=2, roll_coords=False)).data
        expected = xr.DataArray(
            diffed, dims=["depth_c"], coords={"depth_c": grid._ds.depth_c}
        )

        def pad_args(func, pad_width):
            def padding_version_of_func(*args):
                padded_args = [
                    np.pad(a, pad_width=pad_width, mode="wrap") for a in args
                ]
                return func(*padded_args)

            return padding_version_of_func

        # Test direct application
        result = apply_as_grid_ufunc(
            pad_args(diff_center_to_center_second_order, pad_width=[(2, 0)]),
            da,
            axis=[("depth",)],
            grid=grid,
            signature="(X:center)->(X:center)",
            boundary_width=None,
        )
        assert_equal(result, expected)


class TestPadAfterUFunc:
    def test_cumsum(self):
        def cumsum_center_to_left(a):
            return np.cumsum(a, axis=-1)[..., :-1]

        grid = create_1d_test_grid("depth")
        da = grid._ds.depth_c**2
        da.coords["depth_c"] = grid._ds.depth_c

        cum = da.cumsum(dim="depth_c").roll(depth_c=1, roll_coords=False)
        cum[0] = 0
        expected = xr.DataArray(
            cum.data, dims=["depth_g"], coords={"depth_g": grid._ds.depth_g}
        )

        result = apply_as_grid_ufunc(
            cumsum_center_to_left,
            da,
            axis=[("depth",)],
            grid=grid,
            signature="(X:center)->(X:left)",
            boundary_width={"X": (1, 0)},
            boundary="fill",
            fill_value=0,
            pad_before_func=False,
        )
        assert_equal(result, expected)


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
            boundary_width={"X": (1, 0)},
            dask="allowed",
            map_overlap=True,
        )
        def diff_center_to_left(
            a: Annotated[np.ndarray, "X:center"]
        ) -> Annotated[np.ndarray, "X:left"]:
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
            boundary_width={"X": (1, 0)},
            dask="allowed",
            map_overlap=True,
        )
        def diff_outer_to_center(
            a: Annotated[np.ndarray, "X:outer"]
        ) -> Annotated[np.ndarray, "X:center"]:
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
            boundary_width=None,
            map_overlap=True,
            dask="allowed",
        )
        def multiply_left_right(
            a: Annotated[np.ndarray, "X:left"], b: Annotated[np.ndarray, "X:right"]
        ) -> Annotated[np.ndarray, "X:center"]:
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

        a = (grid._ds.lon_c**2 + grid._ds.lat_c**2).chunk(1)

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


class TestBoundary:
    def test_boundary_constant(self):
        def interp(a):
            return 0.5 * (a[..., :-1] + a[..., 1:])

        @as_grid_ufunc(
            signature="(X:center)->(X:left)",
            boundary_width={"X": (1, 0)},
            boundary="fill",
            fill_value=0,
        )
        def interp_center_to_left(a):
            return interp(a)

        grid = create_1d_test_grid("lat")
        arr = np.arange(9)
        da = grid._ds.lat_c.copy(data=arr)

        # test that bound kwargs are used
        result = interp_center_to_left(grid, da, axis=[["lat"]])
        interped_arr_padded_with_zero = interp(np.concatenate([[0], arr]))
        expected = grid._ds.lat_g.copy(data=interped_arr_padded_with_zero)
        assert_equal(result, expected)

        # test that bound kwargs can be overridden at call time
        result = interp_center_to_left(
            grid, da, axis=[["lat"]], boundary="fill", fill_value=1
        )
        interped_arr_padded_with_one = interp(np.concatenate([[1], arr]))
        expected = grid._ds.lat_g.copy(data=interped_arr_padded_with_one)
        assert_equal(result, expected)


# TODO tests for handling dask in gri.diff etc. should eventually live in test_grid.py
class TestMapOverlapGridops:
    def test_chunked_core_dims_unchanging_chunksize_center_to_right(self):
        # attempt to debug GH #438

        grid = create_1d_test_grid("depth")
        da = np.sin(grid._ds.depth_c * 2 * np.pi / 9).chunk(1)
        da.coords["depth_c"] = grid._ds.depth_c

        diffed = (da.roll(depth_c=-1, roll_coords=False) - da).data
        expected = xr.DataArray(
            diffed, dims=["depth_r"], coords={"depth_r": grid._ds.depth_r}
        ).compute()

        result = grid.diff(da, axis="depth", to="right").compute()
        assert_equal(result, expected)

    def test_chunked_core_dims_unchanging_chunksize_center_to_right_2d(self):
        # attempt to debug GH #440

        grid = create_2d_test_grid("depth", "y")

        da = (grid._ds.depth_c**2 + grid._ds.y_c**2).chunk(3)
        da.coords["depth_c"] = grid._ds.depth_c
        da.coords["y_c"] = grid._ds.y_c

        diffed = (da.roll(depth_c=-1, roll_coords=False) - da).data
        expected = xr.DataArray(
            diffed,
            dims=["depth_r", "y_c"],
            coords={"depth_r": grid._ds.depth_r, "y_c": grid._ds.y_c},
        ).compute()

        result = grid.diff(da, axis="depth", to="right").compute()
        assert_equal(result, expected)


class TestSignaturesEquivalent:
    def test_equivalent(self):
        sig1 = _GridUFuncSignature.from_string("(X:center)->(X:left)")
        sig2 = _GridUFuncSignature.from_string("(X:center)->(X:left)")
        assert sig1.equivalent(sig2)

        sig3 = _GridUFuncSignature.from_string("(Y:center)->(Y:left)")
        assert sig1.equivalent(sig3)

    def test_not_equivalent(self):
        sig1 = _GridUFuncSignature.from_string("(X:center)->(X:left)")
        sig2 = _GridUFuncSignature.from_string("(X:center)->(X:center)")
        assert not sig1.equivalent(sig2)

        sig3 = _GridUFuncSignature.from_string("(X:center)->(Y:left)")
        assert not sig1.equivalent(sig3)

        sig4 = _GridUFuncSignature.from_string("(X:center,X:center)->(X:left)")
        assert not sig1.equivalent(sig4)

    def test_no_indices(self):
        sig = _GridUFuncSignature.from_string("()->()")
        assert sig.equivalent(sig)


# TODO Is there a way to prevent this running at test module import time?
# TODO (test by adding a raise in here)
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


class TestGridUFuncDispatch:
    def test_select_ufunc(self):
        gridufunc, _ = _select_grid_ufunc(
            "diff",
            _GridUFuncSignature.from_string("(X:center)->(X:left)"),
            module=GridOpsMockUp,
        )
        assert gridufunc is GridOpsMockUp.diff_center_to_left

    def test_select_ufunc_equivalent_signature(self):
        gridufunc, _ = _select_grid_ufunc(
            "diff",
            _GridUFuncSignature.from_string("(Y:center)->(Y:left)"),
            module=GridOpsMockUp,
        )
        assert gridufunc is GridOpsMockUp.diff_center_to_left

        with pytest.raises(NotImplementedError):
            _select_grid_ufunc(
                "diff",
                _GridUFuncSignature.from_string("(X:center)->(Y:left)"),
                module=GridOpsMockUp,
            )

    def test_select_ufunc_wrong_signature(self):
        with pytest.raises(NotImplementedError):
            _select_grid_ufunc(
                "diff",
                _GridUFuncSignature.from_string("(X:center)->(X:center)"),
                module=GridOpsMockUp,
            )

    @pytest.mark.xfail(reason="currently no need for this")
    def test_select_ufunc_by_kwarg(self):
        gridufunc, _ = _select_grid_ufunc(
            "diff",
            _GridUFuncSignature.from_string("(X:center)->(X:right)"),
            module=GridOpsMockUp,
            boundary="fill",
        )
        assert gridufunc is GridOpsMockUp.diff_center_to_right_fill

        with pytest.raises(NotImplementedError):
            _select_grid_ufunc(
                "diff",
                _GridUFuncSignature.from_string("(X:center)->(X:right)"),
                module=GridOpsMockUp,
                boundary="nonsense",
            )

    @pytest.mark.xfail
    def test_pass_through_other_kwargs(self):
        # TODO put this in test_grid.py instead?
        gridufunc, _ = _select_grid_ufunc(
            "pass",
            _GridUFuncSignature.from_string("()->()"),
            module=GridOpsMockUp,
            boundary="fill",
        )
        assert gridufunc(a=1) == {"a": 1}
