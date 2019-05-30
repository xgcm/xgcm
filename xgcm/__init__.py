from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .grid import Grid, Axis
from .autogenerate import generate_grid_ds
