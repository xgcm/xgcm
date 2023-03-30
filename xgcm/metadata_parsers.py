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
    # Follow sgrid approach below

    # try sgrid parsing
    if sgrid.assert_valid_sgrid(ds):
        ds_sgrid, grid_kwargs_sgrid = parse_sgrid(ds)
    # fall back on comodo
    else:
        ds_comodo, grid_kwargs_comodo = parse_comodo(ds)

    # Hierachy of conventions:
    # This will need expanding as more metadata conventions are added.
    # Currently use sgrid if available, otherwise comodo as in older version.
    # May seem superflous at present but in preparation for future developments.
    if sgrid.assert_valid_sgrid(ds):
        ds = ds_sgrid
        grid_kwargs = grid_kwargs_sgrid
    else:
        ds = ds_comodo
        grid_kwargs = grid_kwargs_comodo

    return (ds, grid_kwargs)


def parse_sgrid(ds):
    """Attempt to extract sgrid metadata from the dataset

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    (ds, sgrid_grid_kwargs) : Tupule of xr.dataset and dict
        ds is the xarray dataset with any neccessary modifications
        sgrid_grid_kwargs is a dictionary of kwargs for forming a Grid object
          extracted from metadata


    """

    sgrid_ax_names = sgrid.get_all_axes(ds)
    parsed_coords = {}
    for ax_name in sgrid_ax_names:
        parsed_coords[ax_name] = sgrid.get_axis_positions_and_coords(ds, ax_name)

    sgrid_grid_kwargs = {"coords": parsed_coords}
    return (ds, sgrid_grid_kwargs)


def parse_comodo(ds):
    """Attempt to extract comodo metadata from the dataset

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    (ds, comodo_grid_kwargs) : Tupule of xr.dataset and dict
        ds is the xarray dataset with any neccessary modifications
        comodo_grid_kwargs is a dictionary of kwargs for forming a Grid object
          extracted from metadata


    """

    comodo_ax_names = comodo.get_all_axes(ds)
    parsed_coords = {}
    for ax_name in comodo_ax_names:
        parsed_coords[ax_name] = comodo.get_axis_positions_and_coords(ds, ax_name)

    comodo_grid_kwargs = {"coords": parsed_coords}
    return (ds, comodo_grid_kwargs)


def cf_parser(ds):
    """Attempt to extract cf metadata from the dataset

    Parameters
    ----------
    ds : xarray.dataset

    Returns
    ----------
    (ds, cf_grid_kwargs) : Tupule of xr.dataset and dict
        ds is the xarray dataset with any neccessary modifications
        cf_grid_kwargs is a dictionary of kwargs for forming a Grid object
          extracted from metadata


    """
    # TODO: To be completed as part of #568
    cf_grid_kwargs = {}

    return (ds, cf_grid_kwargs)
