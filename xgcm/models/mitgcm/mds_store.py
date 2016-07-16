"""
Class to represent MITgcm mds file storage format.
"""
# python 3 compatiblity
from __future__ import print_function, division

import operator
from glob import glob
import os
import warnings
import numpy as np
import dask.array as da
import xarray as xr
from xarray import Variable
from xarray import backends
from xarray import core

# we keep the metadata in its own module to keep this one cleaner
from .variables import dimensions, \
    horizontal_coordinates_spherical, horizontal_coordinates_cartesian, \
    vertical_coordinates, horizontal_grid_variables, vertical_grid_variables, \
    volume_grid_variables, state_variables
# would it be better to import mitgcm_variables and then automate the search
# for variable dictionaries

from .utils import parse_meta_file


class MDSDataStore(backends.common.AbstractDataStore):
    """Representation of MITgcm mds binary file storage format for a specific
    model instance."""
    def __init__(self, dirname, iters=None, deltaT=1,
                 prefix=None, ref_date=None, calendar=None,
                 ignore_pickup=True, geometry='cartesian',
                 skip_vars=[], endian='>'):
        """

        Parameters
        ----------
        dirname : string
            Location of the output files. Usually the "run" directory.
        iters : list
            The iteration numbers corresponding with the files to read
        deltaT : float
            Numerical timestep of MITgcm model. Used to infer actual time
        prefix : list
            Prefixes (string) of files to read. (If None, read all)
        ref_date : datetime or string
        calendar :
        ignore_pickup :
        skip_vars :
        endian :
        """

        self.geometry = geometry.lower()
        allowed_geometries = ['cartesian', 'sphericalpolar', 'llc']
        if self.geometry not in allowed_geometries:
            raise ValueError('Unexpected value for parameter `geometry`. '
                             'It must be one of the following: %s' %
                             allowed_geometries)


        # the directory where the files live
        self.dirname = dirname

        # storage dicts for variables and attributes
        self._variables = core.pycompat.OrderedDict()
        self._attributes = core.pycompat.OrderedDict()
        self._dimensions = []

        # the dimensions are theoretically the same for all datasets
        [self._dimensions.append(k) for k in dimensions]
        if self.geometry == 'llc':
            self._dimensions.append('face')

        # TODO: and maybe here a check for the presence of layers?

        # Now we need to figure out the dimensions of the numerical domain, i.e.
        # nx, ny, nz. We do this by peeking at the grid file metadata
        try:
            rc_meta = parse_meta_file(os.path.join(self.dirname, 'RC.meta'))
            self.nz = rc_meta['dimList'][2][2]
        except IOError:
            raise RuntimeError("Couldn't find RC.meta file to infer nz.")
        try:
            xc_meta = parse_meta_file(os.path.join(self.dirname, 'XC.meta'))
            self.nx = xc_meta['dimList'][0][0]
            self.ny = xc_meta['dimList'][1][0]
        except IOError:
            raise RuntimeError("Couldn't find XC.meta file to infer nx and ny.")

        # Now set up the corresponding coordinates.
        # Rather than assuming the dimension names, we use Comodo conventions to
        # parse the dimension metdata
        irange = np.arange(self.nx)
        jrange = np.arange(self.ny)
        krange = np.arange(self.nz)
        krange_p1 = np.arange(self.nz+1)
        # the keys are `standard_name` attribute
        dimension_data = {
            "x_grid_index": irange,
            "x_grid_index_at_u_location": irange,
            "x_grid_index_at_f_location": irange,
            "y_grid_index": jrange,
            "y_grid_index_at_v_location": jrange,
            "y_grid_index_at_f_location": jrange,
            "z_grid_index": krange,
            "z_grid_index_at_lower_w_location": krange,
            "z_grid_index_at_upper_w_location": krange,
            "z_grid_index_at_w_location": krange_p1,
        }

        for dim in self._dimensions:
            dim_meta = dimensions[dim]
            dims = dim_meta['dims']
            attrs = dim_meta['attrs']
            data = dimension_data[attrs['standard_name']]
            dim_variable = Variable(dims, data, attrs)
            self._variables[dim] = dim_variable

        # the rest of the data has to be read from disk
        # USE A SINGLE SYNTAX TO READ ALL VARIABLES... GRID OR OTHERWISE
        # Perhaps this is as simple as specifying the file prefixes

        prefixes = []
        for p in prefixes:
            if p not in grid_variables:
                pass
                # do something related to time
            dims, data, attrs = self.read_data_and_lookup_metadata(p)
            self._variables[p] = Variable(dims, data, attrs)

    def read_grid_data(self, name):
        """Read data and look up metadata for grid variable `name`.

        Parameters
        ----------
        name : string
            The name of the grid variable.

        Returns
        -------
        dims : list
            The dimension list
        data : arraylike
            The raw data
        attrs : dict
            The metadata attributes
        """
        pass

    def get_variables(self):
        return self._variables

    def get_attrs(self):
        return self._attributes

    def get_dimensions(self):
        return self._dimensions

    def close(self):
        pass
