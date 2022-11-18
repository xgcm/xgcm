from collections import OrderedDict

"""Some musings on concrete SGRID syntax.

Most example staggered grids on the SGRID spec are described using a Java-like
syntax (probably a NetCDF export/view of the metadata), which isn't part of the
spec. However, some dataset attributes use syntax that deserve commentary.

The `*_dimensions` attributes have different forms depending on grid
dimensionality.

  * The most simple form is a space-separated list of node dimensions.
    `node_dimensions` uses this in 2D and 3D.
  * The general form is a list where each entry is either `NODE_DIMENSION` or
    `FACE_DIMENSION: NODE_DIMENSION (padding: PADDING_TYPE)`. In 3D grids, these
    may be mixed and matched (i.e. the optional `edge1_dimensions` attribute
    looks like `face_dimension1:node_dimension1 (padding:type1) node_dimension2
    node_dimension3`: face-node connectivity is defined for only the first
    dimension
"""


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
    """Obtain the name of the dummy variable that stores various SGRID metadata.

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    var_name : str
    """

    # TODO 2022-11-17 raehik
    # re a conversation with xGCM devs: seems reasonable this is an omission in
    # the SGRID spec. Perhaps we simply assume `grid`.

    # look for a reasonably uniquely-identifying attribute in each variable, and
    # return the first one found (we assume there should only be one)
    # (checking for `grid_topology` is a bit extra)
    for var_name in ds.data_vars:
        if ds[var_name].attrs.get("cf_role") == "grid_topology":
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
    # set of strings

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


def get_axis_positions_and_coords(ds, axis_name):
    """
    SGRID version of related function in `comodo.py`.

    Returns a dictionary of position->coordinate entries, which is then stored
    in the `Axis.coords` instance variable.

    https://sgrid.github.io/sgrid/
    """

    # generate an ordered dict axis_coords
    # Populate with keywords 'center', 'inner', 'outer', 'left', 'right'
    # corresponding to the name of the coordinates on this axis
    # For SGRID this can be done based on whether it is node or face and looking at the padding attribute.
    #
    # center refers to the edge location in 1D, face locations in 2D, volume locations in 3D

    # 1) Guaranteed are node and face/volume dimensions, which include padding info so check for these first, and
    #      record the type of shift.
    # 2) Then check for coordinates and if absent generate.
    # 3) Finally store type of 'shift'associated with this coordinate.

    # To ask xgcm and sgrid:
    # - 'vertical_dimensions' are vertical coordinates ever supplied somewhere? Looks like no in sgrid docs?
    # - If coordinates not supplied do we need to generate some and add to dataset?
    # - If coordinates are no supplied is a range of [0, 1] appropriate?
    # - Can you have a 1D dataset?

    # Dictionary mapping SGRID padding types to xgcm node positions
    pad2pos = {
        "high": "left",
        "low": "right",
        "both": "inner",
        "none": "outer",
    }

    # Extract the name of the SGRID grid variable
    sgrid_grid_name = get_sgrid_grid(ds)
    sgrid_grid_dim = ds[sgrid_grid_name].attrs["topology_dimension"]

    # Generate an empty dictionary to store coordinates associated with this axis
    axis_coords = OrderedDict()

    # Select the index to use corresponding to X, Y, or Z axis
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

    # Coordinates are not a required attribute for sgrid, only dimensions.
    # Therefore need to generate a default if these are not specified.
    # Choose a grid spanning [0, 1]
    # TODO: Need to be careful, however, as if Volume AND Face coords are specified then
    #  it's over-constrained!

    # TODO 2022-11-18 raehik
    # re discussion earlier this week: following block should be rewritten to
    # consider primarily `face_dimensions` (required for 2D grid), ignore
    # `*_coordinates` attributes)

    if (axis_name == "Z") & ("vertical_dimensions" in ds[sgrid_grid_name].attrs):
        # vertical coordinates not specified for sgrid with vertical_dimensions.
        # Therefore need to generate a default for xgcm. Choose a grid spanning [0, 1]
        node_coord_name = "_".join([axis_name, "node", "coords"])
        cell_coord_name = "_".join([axis_name, "cell", "coords"])
        cell_pad = (
            ds[sgrid_grid_name].attrs["vertical_dimensions"].split()[2].replace(")", "")
        )
        # TODO: Generate vertical coordinates if none supplied

    else:
        # Nodes
        # node_dim = ds[sgrid_grid_name].attrs["node_dimensions"].split()[i_select]
        if "node_coordinates" in ds[sgrid_grid_name].attrs:
            node_coord_name = (
                ds[sgrid_grid_name].attrs["node_coordinates"].split()[i_select]
            )
        else:
            node_coord_name = "_".join([axis_name, "node", "coords"])
            # TODO: Generate node coordinates if none supplied

        # Edges/Faces/Volume
        if sgrid_grid_dim == 2:
            cell_pad = (
                ds[sgrid_grid_name]
                .attrs["face_dimensions"]
                .split()[3 * i_select + 2]
                .replace(")", "")
            )
            if "face_coordinates" in ds[sgrid_grid_name].attrs:
                cell_coord_name = (
                    ds[sgrid_grid_name].attrs["face_coordinates"].split()[i_select]
                )
            else:
                cell_coord_name = "_".join([axis_name, "cell", "coords"])
                # TODO: Generate face coordinates in ds if none supplied

        elif sgrid_grid_dim == 3:
            cell_pad = (
                ds[sgrid_grid_name]
                .attrs["volume_dimensions"]
                .split()[3 * i_select + 2]
                .replace(")", "")
            )

            if "volume_coordinates" in ds[sgrid_grid_name].attrs:
                cell_coord_name = (
                    ds[sgrid_grid_name].attrs["volume_coordinates"].split()[i_select]
                )
            else:
                cell_coord_name = "_".join([axis_name, "cell", "coords"])
                # TODO: Generate volume coordinates if none supplied
        else:
            raise ValueError(
                f"Sgrid dimensions {sgrid_grid_dim} in variable {sgrid_grid_name} is > 3."
            )

    # Set the padding type for the nodes accordingly. Cell padding is center.
    axis_coords["center"] = cell_coord_name
    try:
        axis_pos = pad2pos[cell_pad]
        axis_coords[axis_pos] = node_coord_name
    except KeyError as e:
        # TODO: Raise error properly
        pass

    # TODO: do we need to infer/generate edge_i/face_i coordinates here?
    #  Discuss with xgcm...
    # No... but we may need to infer cell coordinates from them if face/volume not provided... will be messy...
    # This would need adding to the above

    return axis_coords


def _assert_data_on_grid(da):
    pass
