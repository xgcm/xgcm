from . import comodo, sgrid


def parse_metadata(ds):
    """Attempt to extract metadata from the dataset to provide coordinates
    and grid kwargs

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    (ds, grid_kwargs) : Tupule of xr.dataset and dict
        ds is the xarray dataset with any neccessary modifications
        grid_kwargs is a dictionary of kwargs for forming a Grid object
          extracted from metadata

    """

    # TODO (Julius in #568) full hierarchy of conventions here
    # but override with any user-given options

    # Placeholder for parsing CF metadata

    # try sgrid parsing
    if sgrid.assert_valid_sgrid(ds):
        parsed_coords = parse_sgrid(ds)
    # fall back on comodo
    else:
        parsed_coords = parse_comodo(ds)

    # TODO: Discuss at meeting:
    #   - Any other metadata to be extracted?
    #   - (real-world) coordinates from sgrid? Not sure xgcm uses.
    #   - Most is probably in CF metadata
    
    grid_kwargs = {
            "coords": parsed_coords
            }

    return (ds, grid_kwargs)


def parse_sgrid(ds):
    """Attempt to extract sgrid metadata from the dataset

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------

    """

    sgrid_ax_names = sgrid.get_all_axes(ds)
    parsed_coords = {}
    for ax_name in sgrid_ax_names:
        parsed_coords[ax_name] = sgrid.get_axis_positions_and_coords(ds, ax_name)
    return parsed_coords


def parse_comodo(ds):
    """Attempt to extract comodo metadata from the dataset

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------

    """

    comodo_ax_names = comodo.get_all_axes(ds)
    parsed_coords = {}
    for ax_name in comodo_ax_names:
        parsed_coords[ax_name] = comodo.get_axis_positions_and_coords(ds, ax_name)
    return parsed_coords


def cf_parser(ds):
    """Attempt to extract cf metadata from the dataset

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------

    """
    # TODO: To be completed as part of #568
    pass
    return None
