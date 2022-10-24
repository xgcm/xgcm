from collections import OrderedDict


def assert_valid_sgrid(ds):
    """Verify that the dataset meets SGRID conventions
    We know this if the dataset has an integer grid variable with the
    attribute 'cf_role'

    Parameters
    ----------
    ds : xarray.dataset
    """

    # TODO: Check that this really is a unique identifier for SGRID datasets
    for d in ds.data_vars:
        if "cf_role" in ds[d].attrs:
            return True
        else:
            pass
    return False
