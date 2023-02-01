from . import sgrid, comodo


def parse_metadata(ds):
    """Attempt to extract metadata from the dataset to provide coordinates
    and grid kwargs

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------

    """

    # Placeholder for parsing CF metadata
    # TODO: add this in with #568

    # try sgrid parsing
    if sgrid.assert_valid_sgrid(ds):
        parsed_coords = parse_sgrid(ds)
    # fall back on comodo
    else:
        parsed_coords = parse_comodo(ds)

    return (ds, parsed_coords)


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
