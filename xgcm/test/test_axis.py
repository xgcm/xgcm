import numpy as np
import pytest
import xarray as xr

from xgcm.axis import Axis

from .datasets import all_datasets, datasets  # noqa


def periodic_1d():
    # TODO get this from datasets.py

    N = 5
    ds = xr.Dataset(
        {"data_g": (["XG"], np.random.rand(N)), "data_c": (["XC"], np.random.rand(N))},
        coords={
            "XG": (
                ["XG"],
                2 * np.pi / N * np.arange(0, N),
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "XC": (["XC"], 2 * np.pi / N * (np.arange(0, N) + 0.5), {"axis": "X"}),
        },
    )
    return ds


class TestInit:
    def test_default_init(self):

        # test initialisation
        axis = Axis(name="X", ds=periodic_1d(), coords={"center": "XC", "left": "XG"})

        # test attributes
        assert axis.name == "X"
        assert axis.coords == {"center": "XC", "left": "XG"}

        # test default values of attributes
        assert axis.default_shifts == {"left": "center", "center": "left"}
        assert axis.boundary == "periodic"

    def test_override_defaults(self):
        # test initialisation
        axis = Axis(
            name="foo",
            ds=periodic_1d(),
            coords={"center": "XC", "left": "XG"},
            # TODO does this make sense as default shifts?
            default_shifts={"left": "inner", "center": "outer"},
            boundary="fill",
        )

        # test attributes
        assert axis.name == "foo"
        assert axis.coords == {"center": "XC", "left": "XG"}

        # test default values of attributes
        # TODO (these deafult shift values make no physical sense)
        assert axis.default_shifts == {"left": "inner", "center": "outer"}
        assert axis.boundary == "fill"

    def test_inconsistent_dims(self):
        """Test when xgcm coord names are not present in dataset dims"""
        with pytest.raises(ValueError, match="Could not find dimension"):
            Axis(
                name="X",
                ds=periodic_1d(),
                coords={"center": "lat", "left": "lon"},
            )

    def test_invalid_args(self):

        # invalid defaults
        with pytest.raises(ValueError, match="Can't set the default"):
            Axis(
                name="foo",
                ds=periodic_1d(),
                coords={"center": "XC", "left": "XG"},
                default_shifts={"left": "left", "center": "center"},
            )

        with pytest.raises(ValueError, match="boundary must be one of"):
            Axis(
                name="foo",
                ds=periodic_1d(),
                coords={"center": "XC", "left": "XG"},
                boundary="blargh",
            )

    def test_repr(self):
        axis = Axis(name="X", ds=periodic_1d(), coords={"center": "XC", "left": "XG"})
        repr = axis.__repr__()

        assert repr.startswith("<xgcm.Axis 'X'")


def test_get_position_name():
    ds = periodic_1d()
    axis = Axis(name="X", ds=ds, coords={"center": "XC", "left": "XG"})

    da = ds["data_g"]
    pos, name = axis._get_position_name(da)
    assert pos == "left"
    assert name == "XG"


def test_get_axis_dim_num():
    ds = periodic_1d()
    axis = Axis(name="X", ds=ds, coords={"center": "XC", "left": "XG"})

    da = ds["data_g"]
    num = axis._get_axis_dim_num(da)
    assert num == da.get_axis_num("XG")


def test_assert_axes_equal():
    ...
