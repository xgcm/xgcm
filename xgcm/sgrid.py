from collections import OrderedDict


def assert_valid_sgrid(ds):
    """Verify that the dataset meets SGRID conventions
    We know this if the dataset has 'SGRID' listed in its conventions

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    True/False : boolean
    """

    if "SGRID" in ds.Conventions:
        return True
    elif "sgrid" in ds.Conventions:
        return True
    elif "Sgrid" in ds.Conventions:
        return True

    return False


def get_sgrid_grid(ds):
    """Extract the name of the variable containing the SGRID grid data.
    This can be done by checking cf_role attribute for 'grid_topology'

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    var_name : str
        The name of the grid variable
    """

    for var_name in ds.data_vars:
        if ds[var_name].attrs.get("cf_role") == "grid_topology":
            return var_name
    raise ValueError("Could not find identify SGRID grid in input dataset.")


def get_all_axes(ds):
    """Works out how many axes there are and what the names of them are.
    Names are added to a set.

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    axes : set
        set of strings containing the names of the spatial axes in the dataset

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
            f"SGRID dimensions {ndims} in variable {sgrid_grid_name} is > 3."
        )
    return axes


def get_axis_positions_and_coords(ds, axis_name):
    """SGRID version of related function in `comodo.py`.
    Returns an ordered dictionary of position->coordinate dimension entries
    associated with a given axis, which is then stored in the `Axis.coords`
    instance variable.
    Populate with keywords 'center', 'inner', 'outer', 'left', 'right'
    corresponding to the name of the dimensions on this axis.
    For SGRID this can be done based on whether it is node or face and looking
    at the padding attribute.

    1) Node dimensions are guaranteed by basic SGRID so extract these first
    2) Face or Volume dimensions are guaranteed and are associated with the cell
       coordinates (center)
        a) Check grid dimension and extract face/vol appropriately
        b) Syntax match face/vol dimensions to node dimension
        c) Record Shift and assign to nodes as appropriate
    3) Facei and edgei are optional and depend on 2D vs 3D.
       They add no additional information about dimensions so ignored for now.

    SGRID cell padding types can be converted to xgcm node positions as follows:
      - padding low  refers to 'right'
      - padding high refers to 'left'
      - padding both refers to 'inner'
      - padding none refers to 'outer'

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    axis_name : str
        The name of the axis to find (e.g. 'X')

    References
    ----------
    https://sgrid.github.io/sgrid/
    """

    # Dictionary mapping SGRID padding types to xgcm node positions
    pad2pos = {
        "high": "left",
        "low": "right",
        "both": "inner",
        "none": "outer",
    }

    # Extract the name of the SGRID grid variable and number of dimensions
    sgrid_grid_name = get_sgrid_grid(ds)
    sgrid_grid_dim = ds[sgrid_grid_name].attrs["topology_dimension"]

    # Generate an empty dictionary to store coordinates associated with the axis
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
            f"Axis name '{axis_name}' not recognised as one of the default SGRID values 'X', 'Y', 'Z'."
        )

    # Due to the way vertical coordinates are represented with a 2D dataset
    # using SGRID catch this case and handle before general cases:
    if (axis_name == "Z") & ("vertical_dimensions" in ds[sgrid_grid_name].attrs):
        vert_dim = ds[sgrid_grid_name].attrs["vertical_dimensions"].split()
        node_dim_name = vert_dim[0].split(":")[1]
        cell_dim_name = vert_dim[0].split(":")[0]
        cell_pad = vert_dim[2].replace(")", "")

    else:
        # Nodes
        if "node_dimensions" in ds[sgrid_grid_name].attrs:
            try:
                node_dim_name = (
                    ds[sgrid_grid_name].attrs["node_dimensions"].split()[i_select]
                )
            except IndexError:
                raise IndexError(
                    f"Not enough 'node_dimensions'. Expecting {i_select} got {len(ds[sgrid_grid_name].attrs['node_dimensions'].split())}."
                )

        else:
            raise ValueError(
                f"'node_dimensions' attribute not found in grid variable '{sgrid_grid_name}''."
            )

        # Edges/Faces/Volume
        # If 2D dataset extract from face_dimensions
        if sgrid_grid_dim == 2:
            cell_dim = ds[sgrid_grid_name].attrs["face_dimensions"].split()

            # Find the face dimension that matches the node dimension
            dim = [s[0] for s in enumerate(cell_dim) if node_dim_name in s[1]]
            if len(dim) != 1:
                raise IndexError(
                    f"Found {len(dim)} face_dimensions corresponding to node_dimension '{node_dim_name}'. Expecting 1."
                )

            cell_dim_name = cell_dim[dim[0]].split(":")[0]
            cell_pad = cell_dim[dim[0] + 2].replace(")", "")

        # If 3D dataset extract from volume_dimensions
        elif sgrid_grid_dim == 3:
            cell_dim = ds[sgrid_grid_name].attrs["volume_dimensions"].split()

            # Find the face dimension that matches the node dimension
            dim = [s[0] for s in enumerate(cell_dim) if node_dim_name in s[1]]
            if len(dim) != 1:
                raise IndexError(
                    f"Found {len(dim)} face_dimensions corresponding to node_dimension '{node_dim_name}'. Expecting 1."
                )

            cell_dim_name = cell_dim[dim[0]].split(":")[0]
            cell_pad = cell_dim[dim[0] + 2].replace(")", "")

            # Check for face dimensions
            # This does not need doing as no additional information compared to
            # that in node and volume dimensions
            # ['face1_dimensions', 'face2_dimensions', 'face3_dimensions']

        else:
            raise ValueError(
                f"SGRID grid dimensions {sgrid_grid_dim} in variable {sgrid_grid_name} is > 3."
            )

        # Check for edge dimensions
        # This does not need doing as no additional information compared to that
        # in node, face, and volume dimensions
        # ['edge1_dimensions', 'edge2_dimensions', 'edge3_dimensions']

    # Set the padding type for the nodes accordingly. Cell padding is center.
    axis_coords["center"] = cell_dim_name
    try:
        axis_pos = pad2pos[cell_pad]
        axis_coords[axis_pos] = node_dim_name
    except KeyError:
        raise KeyError(f"Unexpected padding type '{cell_pad}' in SGRID data.")

    return axis_coords
