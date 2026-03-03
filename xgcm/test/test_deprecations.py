import pytest

import xgcm
from xgcm.test.datasets import datasets

pytestmark = pytest.mark.filterwarnings("error")


def test_periodic_true_deprecation():
    ds = datasets["2d_left"]
    with pytest.raises(
        DeprecationWarning,
        match="The `periodic` argument will be deprecated. To preserve previous behavior supply `padding = 'periodic'.",
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
