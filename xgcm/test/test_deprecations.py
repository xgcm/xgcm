import pytest

import xgcm
from xgcm.test.datasets import datasets

pytestmark = pytest.mark.filterwarnings("error")


def test_periodic_removed():
    # `periodic` was removed in v1.0.0 (#746); passing it now raises an
    # informative error pointing at `padding`.
    ds = datasets["2d_left"]
    with pytest.raises(
        ValueError,
        match="The `periodic` argument has been removed",
    ):
        xgcm.Grid(
            ds,
            coords={
                "X": {"left": "XG", "center": "XC"},
                "Y": {"left": "YG", "center": "YC"},
            },
            autoparse_metadata=False,
            periodic=True,
        )


def test_padding_deprecation():
    ds = datasets["2d_left"]
    with pytest.raises(
        ValueError, match="Argument 'boundary' has been renamed to 'padding'"
    ):
        xgcm.Grid(
            ds,
            coords={
                "X": {"left": "XG", "center": "XC"},
                "Y": {"left": "YG", "center": "YC"},
            },
            autoparse_metadata=False,
            boundary="periodic",
        )


def test_padding_width_deprecation():
    # `boundary_width` was renamed to `padding_width` (#696); the old name now
    # raises rather than warning.
    with pytest.raises(
        ValueError,
        match="Argument 'boundary_width' has been renamed to 'padding_width'",
    ):
        xgcm.as_grid_ufunc("(X:center)->(X:left)", boundary_width={"X": (1, 0)})


def test_axis_boundary_attribute_removed():
    # The `Axis.boundary` attribute was renamed to `Axis.padding` (#696).
    ds = datasets["2d_left"]
    grid = xgcm.Grid(
        ds,
        coords={
            "X": {"left": "XG", "center": "XC"},
            "Y": {"left": "YG", "center": "YC"},
        },
        autoparse_metadata=False,
        padding="periodic",
    )
    with pytest.raises(
        AttributeError, match="Attribute 'boundary' has been renamed to 'padding'"
    ):
        grid.axes["X"].boundary


def test_gridufunc_boundary_attributes_removed():
    # `GridUFunc.boundary` / `.boundary_width` were renamed to
    # `.padding` / `.padding_width` (#696).
    gf = xgcm.as_grid_ufunc("(X:center)->(X:left)")(lambda a: a)
    with pytest.raises(
        AttributeError, match="Attribute 'boundary' has been renamed to 'padding'"
    ):
        gf.boundary
    with pytest.raises(
        AttributeError,
        match="Attribute 'boundary_width' has been renamed to 'padding_width'",
    ):
        gf.boundary_width
