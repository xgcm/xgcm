import xarray as xr

from .grid import Grid, _get_global_grids


class GridError(Exception):
    pass


def _find_compatible_global_grid(da):
    global_grids = _get_global_grids()
    for grid in global_grids:
        try:
            xr.align(grid._ds, da, join="exact")
            return grid
        except ValueError:
            pass
    raise GridError("No compatible global grid found.")


@xr.register_dataarray_accessor("grid")
class GridAccessor:
    def __init__(self, xarray_obj):
        self._da = xarray_obj
        self._grid_obj = None
        self.init()

    def init(self, grid_obj=None):
        """Initialize xgcm.Grid object for the dataset."""
        if grid_obj:
            self._grid_obj = grid_obj
        else:
            self._grid_obj = _find_compatible_global_grid(self._da)

    def interp(self, *args, **kwargs):
        return self._grid_obj.interp(self._da, *args, **kwargs)

    def diff(self, *args, **kwargs):
        return self._grid_obj.diff(self._da, *args, **kwargs)
