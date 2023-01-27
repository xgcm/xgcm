try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .grid import Axis, Grid
from .grid_ufunc import apply_as_grid_ufunc, as_grid_ufunc
