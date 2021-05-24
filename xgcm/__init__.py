try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .grid import Grid, Axis
from .autogenerate import generate_grid_ds
from .accessor import GridAccessor
