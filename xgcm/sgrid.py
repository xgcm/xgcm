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
    #  i.e. is the existence of cf_role a sufficcient condition for SGRID?
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


def get_axis_coords(ds, axis_name):
    """Find the name of the coordinates associated with an sgrid axis.

    Parameters
    ----------
    ds : xarray.dataset or xarray.dataarray
    axis_name : str
        The name of the axis to find (e.g. 'X')

    Returns
    -------
    coord_name : list
        The names of the coordinate matching that axis
    """

    coord_names = []

    # Extract the name of the SGRID grid variable
    sgrid_grid_name = get_sgrid_grid(ds)

    # Select the index to use corresponding to X, Y, or Z
    if axis_name == "X":
        i_select = 0
    elif axis_name == "Y":
        i_select = 1
    elif axis_name == "Z":
        i_select = 2
    else:
        raise ValueError(
            f"{axis_name} not recognised as one of the default Sgrid values 'X', 'Y', 'Z'."
        )

    # If it's a 3D grid using vertical_dimensions catch this and treat appropriately
    if (axis_name == "Z") & ("vertical_dimensions" in ds[sgrid_grid_name].attrs):
        # vertical coordinates not specified for sgrid with vertical_dimensions.
        # Therefore need to generate a default for xgcm. Choose a grid spanning [0, 1]
        pass
    else:

        # Coordinates are not a required attribute for sgrid, only dimensions.
        # Therefore need to generate a default if these are not specified.
        # Choose a grid spanning [0, 1]
        if "node_coordinates" in ds[sgrid_grid_name].attrs:
            coord_names.append(ds[sgrid_grid_name].attrs["node_coordinates"].split()[i_select])
        else:
            pass

        if "face_coordinates" in ds[sgrid_grid_name].attrs:
            coord_names.append(ds[sgrid_grid_name].attrs["face_coordinates"].split()[i_select])
        else:
            pass

        if "volume_coordinates" in ds[sgrid_grid_name].attrs:
            coord_names.append(ds[sgrid_grid_name].attrs["volume_coordinates"].split()[i_select])
        else:
            pass

    return coord_names


def get_axis_positions_and_coords(ds, axis_name):
    coord_names = get_axis_coords(ds, axis_name)
    print(f"{axis_name}: {coord_names}")
    # ncoords = len(coord_names)
    # if ncoords == 0:
    #     # didn't find anything for this axis
    #     raise ValueError("Couldn't find any coordinates for axis %s" % axis_name)

    # # now figure out what type of coordinates these are:
    # # center, left, right, or outer

    # now we can start filling in the information about the different coords
    axis_coords = OrderedDict()

    return axis_coords
