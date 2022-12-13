import cf_xarray


def cf_parser(ds):
    coords = {}
    cf_axes = (
        ds.cf.axes
    )  # TODO This contains lists of dims per axis. Can there be multiple dims associated with an axis in the CF conventions?

    # only allow a single dataset dimension per axis

    def only_one_dim(candidates: list) -> str:
        [dim] = candidates
        return dim

    cf_axes = {ax: only_one_dim(dims) for ax, dims in cf_axes.items()}

    for ax, [center_grid_dim] in ds.cf.axes.items():
        coords[ax] = {}
        coords[ax]["center"] = center_grid_dim
        # Check if bounds are available for dims
        [bounds_var] = ds.cf.bounds.get(
            ax, [None]
        )  # Make sure this is None or a single element list
        # TODO: What happens if there is more than one value given here?
        if bounds_var:

            #
            da_bounds = ds[bounds_var]
        else:
            # recreate bounds
            da_bounds = ds.cf.add_bounds(center_grid_dim)[center_grid_dim + "_bounds"]

        [bounds_dim] = set(da_bounds.dims) - set(
            cf_axes.values()
        )  # TODO: test with 2d bounds
        da_outer = cf_xarray.bounds_to_vertices(da_bounds, bounds_dim=bounds_dim)
        grid_dim_outer = center_grid_dim + "_outer"
        # assign new dimension to dataset
        ds = ds.assign_coords({grid_dim_outer: da_outer.data})
        coords[ax]["outer"] = grid_dim_outer

    # TODO:
    # TODO: We could create coordinate bounds here too and approximate metrics
    return {"ds": ds, "coords": coords}
