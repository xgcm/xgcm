from xgcm import sgrid
from xgcm import metadata_parsers
from .datasets import all_sgrid  # noqa: F401
from .datasets import nonperiodic_1d  # noqa: F401


def test_valid_sgrid(all_sgrid):
    ds, periodic, expected = all_sgrid
    assert sgrid.assert_valid_sgrid(ds)


def test_invalid_sgrid(nonperiodic_1d):
    # TODO: Is there a more rigorous way to check? Construct dataset with no conventions?
    ds, periodic, expected = nonperiodic_1d
    assert not sgrid.assert_valid_sgrid(ds)


def test_parse_sgrid(all_sgrid):
    ds, periodic, expected = all_sgrid
    parsed_coords = metadata_parsers.parse_sgrid(ds)
    assert parsed_coords == expected["axes"]
    # assert 'X' in kwargs_parsed['coords'].keys()
    # ....


# TODO: Tests for: sgrid.get_sgrid_grid, sgrid.get_all_axes, sgrid.get_axis_positions_and_coords?

# TODO: Construct dict of comodo datasets to test comodo parsing as follows:
# def test_parse_comodo(all_comodo):
#     ds, periodic, expected = all_comodo
#     parsed_coords = metadata_parsers.parse_comodo(ds)
#     assert parsed_coords == expected["axes"]
