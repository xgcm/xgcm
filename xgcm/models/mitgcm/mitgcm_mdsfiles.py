"""
Functions for reading MITgcm mds files (.meta / .data)
"""
# python 3 compatiblity
from __future__ import print_function, division

import operator
from glob import glob
import os
import re
import warnings
import numpy as np
import dask.array as da
import xarray as xr
from xarray import Variable
from xarray import backends
from xarray import core

from .mitgcm_variables import _grid_variables, _state_variables, /
                            _grid_special_mapping, _ptracers
