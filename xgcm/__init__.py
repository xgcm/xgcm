try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .autogenerate import generate_grid_ds
from .grid import Axis, Grid
