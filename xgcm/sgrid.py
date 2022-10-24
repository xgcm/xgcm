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
    for var_name in ds.data_vars:
        if "cf_role" in ds[var_name].attrs:
            return True
        else:
            pass
    return False


def get_sgrid_grid(ds):
    """Extract the 

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    var_name : str
    """

    for var_name in ds.data_vars:
        if "cf_role" in ds[var_name].attrs:
            return var_name
        else:
            pass
    raise ValueError(
        "Could not find identify sgrid grid in input dataset. This should not happen."
    )


def get_all_axes(ds):
    """Works out how many axes there are and what the names of them are.
    Names are added to a set.

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------

    """
    axes = set()

    sgrid_grid_name = get_sgrid_grid(ds)
    ndims = ds[sgrid_grid_name].attrs["topology_dimension"]
    if ndims == 1:
        axes.update(["X"])
    elif ndims == 2:
        axes.update(["X", "Y"])
        # Check for a vertical dimension
        if "vertical_dimensions" in ds[sgrid_grid_name].attrs:
            axes.update(["Z"])
    elif ndims == 3:
        axes.update(["X", "Y", "Z"])
    else:
        raise ValueError(
            f"Sgrid dimensions {ndims} in variable {sgrid_grid_name} is > 3."
        )
    print(f"axes = {axes}")
    return axes
