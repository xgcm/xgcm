import pytest
from xgcm import metadata_parsers, sgrid

from .datasets import all_sgrid  # noqa: F401
from .datasets import nonperiodic_1d  # noqa: F401


class TestSGRID:
    def test_valid_sgrid(self, all_sgrid):
        # Check valid SGRID datasets are identified as such
        ds, periodic, expected = all_sgrid
        assert sgrid.assert_valid_sgrid(ds)

    def test_invalid_sgrid(self, nonperiodic_1d):
        # Check non-valid SGRID datasets are identified as such
        # TODO: Is there a more rigorous way to check? Construct dataset with no conventions?
        ds, periodic, expected = nonperiodic_1d
        assert not sgrid.assert_valid_sgrid(ds)

    def test_valid_get_grid(self, all_sgrid):
        # Check valid SGRID datasets have correct variable returned
        ds, periodic, expected = all_sgrid
        assert ds[sgrid.get_sgrid_grid(ds)].attrs.get("cf_role") == "grid_topology"

    def test_invalid_get_grid(self, nonperiodic_1d):
        # Check invalid SGRID datasets raise error
        ds, periodic, expected = nonperiodic_1d
        msg = "Could not find identify SGRID grid in input dataset."
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            sgrid.get_sgrid_grid(ds)

    def test_get_all_axes(self, all_sgrid):
        # Check valid SGRID datasets generate expected axes for 2D, 2D+vert, 3D datasets
        ds, periodic, expected = all_sgrid
        assert sgrid.get_all_axes(ds) == expected["axes"].keys()

    def test_get_axis_positions_and_coords(self, all_sgrid):
        # Check valid SGRID datasets generate expected coordinates
        # for 2D, 2D+vert, 3D datasets
        ds, periodic, expected = all_sgrid
        for ax in sgrid.get_all_axes(ds):
            assert sgrid.get_axis_positions_and_coords(ds, ax) == expected["axes"][ax]

    def test_parse_sgrid(self, all_sgrid):
        # check parsing SGRID returns corect coordinates for 2D 2D+vert and 3D grids
        ds, periodic, expected = all_sgrid
        _, parsed_kwargs = metadata_parsers.parse_sgrid(ds)
        assert parsed_kwargs["coords"] == expected["axes"]


# TODO: Currently comodo datasets are tested in test_grid.py as per pre-refactor
#  Consider introducing a set of COMODO datasets and testing here using a
#  similar structure to the TestSGRID class above


# TODO: Add similar TestCF class once #568 is completed


# TODO: currently parse_metadata in metadata_parsers.py is not directly tested
#  This will need doing as a hierachy is developed and conflicting use cases found.
