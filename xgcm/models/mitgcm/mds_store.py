"""
Class to represent MITgcm mds file storage format.
"""
# python 3 compatiblity
from __future__ import print_function, division

from glob import glob
import os
import numpy as np
import xarray as xr
import warnings

# we keep the metadata in its own module to keep this one cleaner
from .variables import dimensions, \
    horizontal_coordinates_spherical, horizontal_coordinates_cartesian, \
    vertical_coordinates, horizontal_grid_variables, vertical_grid_variables, \
    volume_grid_variables, state_variables
# would it be better to import mitgcm_variables and then automate the search
# for variable dictionaries

from .utils import parse_meta_file, read_mds, parse_available_diagnostics


def open_mdsdataset(dirname, iters=None, delta_t=1, read_grid=True,
                    prefix=None, ref_date=None, calendar=None,
                    ignore_pickup=True, geometry='sphericalpolar',
                    grid_vars_to_coords=True, swap_dims=False,
                    skip_vars=[], endian=">", chunks=None,
                    ignore_unknown_vars=False):
    """Open MITgcm-style mds (.data / .meta) file output as xarray datset.

    Parameters
    ----------
    dirname : string
        Path to the directory where the mds .data and .meta files are stored
    iters : list, optional
        The iterations numbers of the files to be read
    deltaT : number, optional
        The timestep used in the model (can't be inferred)
    read_grid : bool, optional
        Whether to try to read the grid data
    prefix : list, optional
        List of different filename prefixes to read. Default is to read all
        files.
    ref_date : string, optional
        A date string corresponding to the zero timestep. See CF conventions
        [1]_
    calendar : string, optional
        A calendar allowed by CF conventions [1]_
    ignore_pickup : boolean, optional
        Whether to read the pickup files
    geometry : string
        MITgcm grid geometry. (Not really used yet.)
    grid_vars_to_coords : boolean, optional
        If `True`, all grid related variables will be promoted to coordinates
    swap_dims : boolean, optional
        Whether to swap the logical dimensions for physical ones.
    skip_vars : list
        Names of variables to ignore.
    endian : {'=', '>', '<'}, optional
        Endianness of variables. Default for MITgcm is ">" (big endian)

    Returns
    -------
    dset : xarray.Dataset
        Dataset object containing all coordinate and variables

    References
    ----------
    .. [1] http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch04s04.html
    """

    # We either have a single iter, in which case we create a fresh store,
    # or a list of iters, in which case we combine.
    if iters == 'all':
        iters = _get_all_iternums(dirname, file_prefixes=prefix)
    if iters is None:
        iternum = None
    else:
        try:
            iternum = int(iters)
        # if not we probably have some kind of list
        except TypeError:
            if len(iters) == 1:
                iternum = int(iters[0])
            else:
                # We have to check to make sure we have the same prefixes at
                # each timestep...otherwise we can't combine the datasets.
                first_prefixes = prefix or _get_all_matching_prefixes(
                                                        dirname, iters[0])
                for iternum in iters:
                    these_prefixes = _get_all_matching_prefixes(
                        dirname, iternum, prefix
                    )
                    # don't care about order
                    if set(these_prefixes) != set(first_prefixes):
                        raise IOError("Could not find the expected file "
                                      "prefixes %s at iternum %g. (Instead "
                                      "found %s)" % (repr(first_prefixes),
                                                     iternum,
                                                     repr(these_prefixes)))

                # chunk at least by time
                chunks = chunks or {}

                # recursively open each dataset at a time
                datasets = [open_mdsdataset(
                        dirname, iters=iternum, delta_t=delta_t,
                        read_grid=False, swap_dims=False,
                        prefix=prefix, ref_date=ref_date, calendar=calendar,
                        ignore_pickup=ignore_pickup, geometry=geometry,
                        grid_vars_to_coords=grid_vars_to_coords,
                        skip_vars=skip_vars, endian=endian, chunks=chunks,
                        ignore_unknown_vars=ignore_unknown_vars)
                    for iternum in iters]
                # now add the grid
                if read_grid:
                    datasets.insert(0, open_mdsdataset(
                        dirname, iters=None, delta_t=delta_t,
                        read_grid=True, swap_dims=False,
                        prefix=prefix, ref_date=ref_date, calendar=calendar,
                        ignore_pickup=ignore_pickup, geometry=geometry,
                        grid_vars_to_coords=grid_vars_to_coords,
                        skip_vars=skip_vars, endian=endian, chunks=chunks,
                        ignore_unknown_vars=ignore_unknown_vars))
                # apply chunking
                ds = xr.auto_combine(datasets)
                if swap_dims:
                    ds = _swap_dimensions(ds, geometry)
                return ds

    store = _MDSDataStore(dirname, iternum, delta_t, read_grid,
                          prefix, ref_date, calendar,
                          ignore_pickup, geometry, skip_vars, endian,
                          ignore_unknown_vars=ignore_unknown_vars)
    ds = xr.Dataset.load_store(store)

    if swap_dims:
        ds = _swap_dimensions(ds, geometry)

    # turn all the auxilliary grid variables into coordinates
    # if grid_vars_to_coords:
    #     for k in _grid_variables:
    #         ds.set_coords(k, inplace=True)
    #     ds.set_coords('iter', inplace=True)

    # do we need more fancy logic (like open_dataset), or is this enough
    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


def _swap_dimensions(ds, geometry, drop_old=True):
    """Replace logical coordinates with physical ones. Does not work for llc.
    """
    # TODO: handle metadata correctly such that the new dimension attributes
    # still conform to comodo conventions

    if geometry.lower() == 'llc':
        raise ValueError("Can't swap dimensions if geometry is `llc`")

    # first squeeze all the coordinates
    for orig_dim in ds.dims:
        #new_dim = dimensions[orig_dim]['swap_dim']
        if 'swap_dim' in ds[orig_dim].attrs:
            new_dim = ds[orig_dim].attrs['swap_dim']
            coord_var = ds[new_dim]
            for coord_dim in coord_var.dims:
                if coord_dim != orig_dim:
                    # dimension should be the same along all other axes, so just
                    # take the first row / column
                    coord_var = coord_var.isel(**{coord_dim: 0}).drop(coord_dim)
            ds[new_dim] = coord_var
    # then swap dims
    for orig_dim in ds.dims:
        if 'swap_dim' in ds[orig_dim].attrs:
            new_dim = ds[orig_dim].attrs['swap_dim']
            ds = ds.swap_dims({orig_dim: new_dim})
            if drop_old:
                ds = ds.drop(orig_dim)
    return ds


class _MDSDataStore(xr.backends.common.AbstractDataStore):
    """Representation of MITgcm mds binary file storage format for a specific
    model instance."""
    def __init__(self, dirname, iternum=None, delta_t=1, read_grid=True,
                 file_prefixes=None, ref_date=None, calendar=None,
                 ignore_pickup=True, geometry='sphericalpolar',
                 skip_vars=[], endian='>', ignore_unknown_vars=False):
        """

        Parameters
        ----------
        dirname : string
            Location of the output files. Usually the "run" directory.
        iternum : list
            The iteration numbers corresponding with the files to read. If
            None, don't read any data files.
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
        self._ignore_unknown_vars = ignore_unknown_vars

        # storage dicts for variables and attributes
        self._variables = xr.core.pycompat.OrderedDict()
        self._attributes = xr.core.pycompat.OrderedDict()
        self._dimensions = []

        # the dimensions are theoretically the same for all datasets
        [self._dimensions.append(k) for k in dimensions]
        if self.geometry == 'llc':
            self._dimensions.append('face')

        # TODO: and maybe here a check for the presence of layers?

        # Now we need to figure out the dimensions of the numerical domain,
        # i.e. nx, ny, nz. We do this by peeking at the grid file metadata
        self.nz, self.ny, self.nx = _guess_model_dimensions(dirname)
        self.layers = _guess_layers(dirname)

        # Now set up the corresponding coordinates.
        # Rather than assuming the dimension names, we use Comodo conventions
        # to parse the dimension metdata.
        # http://pycomodo.forge.imag.fr/norm.html
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
            dim_variable = xr.Variable(dims, data, attrs)
            self._variables[dim] = dim_variable

        # do the same for layers
        for layer_name, n_layer in self.layers.items():
            for suffix, offset in zip(['bounds', 'center', 'interface'],
                                      [0, -1, -2]):
                # e.g. "layer_1RHO_bounds"
                #dimname = 'layer_' + layer_name + '_' + suffix
                # e.g. "l1_b"
                dimname = 'l' + layer_name[0] + '_' + suffix[0]
                self._dimensions.append(dimname)
                data = np.arange(n_layer + offset)
                # we should figure out a way to properly populate the layers
                # attributes
                attrs = {'standard_name':
                         layer_name + '_layer_grid_index_at_layer_' + suffix,
                         'swap_dim': 'layer_' + layer_name + '_' + suffix}
                dim_variable = xr.Variable([dimname], data, attrs)
                self._variables[dimname] = dim_variable

        # maybe add a time dimension
        if iternum is not None:
            self.time_dim_name = 'time'
            self._dimensions.append(self.time_dim_name)
            # a variable for iteration number
            self._variables['iter'] = xr.Variable(
                        (self.time_dim_name,),
                        [iternum],
                        {'standard_name': 'timestep',
                         'long_name': 'model timestep number'})
            self._variables[self.time_dim_name] = _iternum_to_datetime_variable(
                iternum, delta_t, ref_date, self.time_dim_name
            )

        # build lookup tables for variable metadata
        self._all_grid_variables = _get_all_grid_variables(self.geometry,
                                                           self.layers)
        self._all_data_variables = _get_all_data_variables(self.dirname,
                                                           self.layers)

        # The rest of the data has to be read from disk.
        # The list `prefixes` specifies file prefixes from which to infer
        # The problem with this is that some prefixes are single variables
        # while some are multi-variable diagnostics files.
        prefixes = []
        if read_grid:
            prefixes = prefixes + self._all_grid_variables.keys()

        # add data files
        prefixes = (prefixes +
                    _get_all_matching_prefixes(
                                               dirname,
                                               iternum,
                                               file_prefixes))

        for p in prefixes:
            # use a generator to loop through the variables in each file
            for (vname, dims, data, attrs) in self.load_from_prefix(p, iternum):
                # print(vname, dims, data.shape)
                thisvar = xr.Variable(dims, data, attrs)
                self._variables[vname] = thisvar
                # print(type(data), type(thisvar._data), thisvar._in_memory)

    def load_from_prefix(self, prefix, iternum=None):
        """Read data and look up metadata for grid variable `name`.

        Parameters
        ----------
        name : string
            The name of the grid variable.
        iternume : int (optional)
            MITgcm iteration number

        Yields
        -------
        varname : string
            The name of the variable
        dims : list
            The dimension list
        data : arraylike
            The raw data
        attrs : dict
            The metadata attributes
        """

        fname_base = prefix

        # some special logic is required for grid variables
        if prefix in self._all_grid_variables:
            # grid variables don't have an iteration number suffix
            iternum = None
            # some grid variables have a different filename than their varname
            if 'filename' in self._all_grid_variables[prefix]:
                fname_base = self._all_grid_variables[prefix]['filename']
        else:
            assert iternum is not None

        # get a dict of variables and their data
        vardata = read_mds(os.path.join(self.dirname, fname_base), iternum)
        for vname, data in vardata.items():
            # we now have to revert to the original prefix once the file is read
            if fname_base != prefix:
                vname = prefix
            try:
                metadata = (self._all_grid_variables[vname]
                            if vname in self._all_grid_variables
                            else self._all_data_variables[vname])
            except KeyError:
                if self._ignore_unknown_vars:
                    # we didn't find any metadata, so we just skip this var
                    continue
                else:
                    raise KeyError("Couln't find metadata for variable %s "
                                   "and `ignore_unknown_vars`==False." % vname)

            # maybe slice and squeeze the data
            if 'slice' in metadata:
                sl = metadata['slice']
                # need to promote the variable to higher dimensions in the
                # to handle certain 2D model outputs
                if len(sl) == 3 and data.ndim == 2:
                    data.shape = (1,) + data.shape
                data = np.atleast_1d(data[sl])

            if 'transform' in metadata:
                # transform is a function to be called on the data
                data = metadata['transform'](data)

            dims = metadata['dims']
            attrs = metadata['attrs']

            # Some 2D output squeezes one of the dimensions out (e.g. hFacC).
            # How should we handle this? Can either eliminate one of the dims
            # or add an extra axis to the data. Let's try the former, on the
            # grounds that it is simpler for the user.
            if len(dims) == 3 and data.ndim == 2:
                # Deleting the first dimension (z) assumes that 2D data always
                # corresponds to x,y horizontal data. Is this really true?
                # The answer appears to be yes: 2D (x|y,z) data retains the
                # missing dimension as an axis of length 1.
                dims = dims[1:]
            elif len(dims) == 1 and (data.ndim == 2 or data.ndim == 3):
                # this is for certain profile data like RC, PHrefC, etc.
                data = np.atleast_1d(data.squeeze())

            # need to add an extra dimension at the beginning if we have a time
            # variable
            if iternum is not None:
                dims = [self.time_dim_name] + dims
                newshape = (1,) + data.shape
                data = data.reshape(newshape)

            yield vname, dims, data, attrs

    def get_variables(self):
        return self._variables

    def get_attrs(self):
        return self._attributes

    def get_dimensions(self):
        return self._dimensions

    def close(self):
        pass
        # do we actually need to close the memmaps?


def _guess_model_dimensions(dirname):
    try:
        rc_meta = parse_meta_file(os.path.join(dirname, 'RC.meta'))
        if len(rc_meta['dimList']) == 2:
            nz = 1
        else:
            nz = rc_meta['dimList'][2][2]
    except IOError:
        raise IOError("Couldn't find RC.meta file to infer nz.")
    try:
        xc_meta = parse_meta_file(os.path.join(dirname, 'XC.meta'))
        nx = xc_meta['dimList'][0][0]
        ny = xc_meta['dimList'][1][0]
    except IOError:
        raise IOError("Couldn't find XC.meta file to infer nx and ny.")
    return nz, ny, nx


def _guess_layers(dirname):
    """Return a dict matching layers suffixes to dimension length."""
    layers_files = glob(os.path.join(dirname, 'layers*.meta'))
    all_layers = {}
    for fname in layers_files:
        # should turn "foo/bar/layers1RHO.meta" into "1RHO"
        layers_suf = os.path.splitext(os.path.basename(fname))[0][6:]
        layers_num = int(layers_suf[0])
        meta = parse_meta_file(fname)
        Nlayers = meta['dimList'][2][2]
        all_layers[layers_suf] = Nlayers
    return all_layers


def _get_all_grid_variables(geometry, layers={}):
    """"Put all the relevant grid metadata into one big dictionary."""
    hcoords = (horizontal_coordinates_cartesian if geometry == 'cartesian' else
               horizontal_coordinates_spherical)
    allvars = [hcoords, vertical_coordinates, horizontal_grid_variables,
               vertical_grid_variables, volume_grid_variables]

    # tortured logic to add layers grid variables
    layersvars = [_make_layers_variables(layer_name)
                  for layer_name in layers]
    allvars += layersvars

    metadata = _concat_dicts(allvars)
    return metadata


def _make_layers_variables(layer_name):
    """Translate metadata template to actual variable metadata."""
    from .variables import layers_grid_variables
    lvars = xr.core.pycompat.OrderedDict()
    layer_num = layer_name[0]
    # should always be int
    assert isinstance(int(layer_num), int)
    layer_id = 'l' + layer_num
    for key, vals in layers_grid_variables.items():
        # replace the name template with the actual name
        # e.g. layer_NAME_bounds -> layer_1RHO_bounds
        varname = key.replace('NAME', layer_name)
        metadata = _recursively_replace(vals, 'NAME', layer_name)
        # now fix dimension
        metadata['dims'] = [metadata['dims'][0].replace('l', layer_id)]
        lvars[varname] = metadata
    return lvars

def _recursively_replace(item, search, replace):
    """Recursively search and replace all strings in dictionary values."""
    if isinstance(item, dict):
        return {key: _recursively_replace(item[key], search, replace)
                for key in item}
    try:
        return item.replace(search, replace)
    except AttributeError:
        # probably no such method
        return item

def _get_all_data_variables(dirname, layers):
    """"Put all the relevant data metadata into one big dictionary."""
    allvars = [state_variables]
    # add others from available_diagnostics.log
    fname = os.path.join(dirname, 'available_diagnostics.log')
    if os.path.exists(fname):
        available_diags = parse_available_diagnostics(fname, layers)
        allvars.append(available_diags)
    metadata = _concat_dicts(allvars)

    # Now add the suffix '-T' to every diagnostic. This is a somewhat hacky
    # way to increase the coverage of possible output filenames.
    for name, val in metadata.items():
        newname = name + '-T'
        metadata[newname] = val

    return metadata


def _concat_dicts(list_of_dicts):
    result = xr.core.pycompat.OrderedDict()
    for eachdict in list_of_dicts:
        for k, v in eachdict.items():
            result[k] = v
    return result


def _get_all_iternums(dirname, file_prefixes=None):
    """Scan a directory for all iteration number suffixes."""
    iternums = set()
    all_datafiles = glob(os.path.join(dirname, '*.??????????.data'))
    for f in all_datafiles:
        iternum = int(f[-15:-5])
        prefix = os.path.split(f[:-16])[-1]
        if file_prefixes is None:
            iternums.add(iternum)
        else:
            if prefix in file_prefixes:
                iternums.add(iternum)
    iterlist = sorted(iternums)
    return iterlist


def _get_all_matching_prefixes(dirname, iternum, file_prefixes=None):
    """Scan a directory and return all file prefixes matching a certain
    iteration number."""
    if iternum is None:
        return []
    prefixes = set()
    all_datafiles = glob(os.path.join(dirname, '*.%010d.data' % iternum))
    for f in all_datafiles:
        iternum = int(f[-15:-5])
        prefix = os.path.split(f[:-16])[-1]
        if file_prefixes is None:
            prefixes.add(prefix)
        else:
            if prefix in file_prefixes:
                prefixes.add(prefix)
    return list(prefixes)


def _iternum_to_datetime_variable(iternum, delta_t, ref_date,
                                  calendar, time_dim_name='time'):
    # create time array
    timedata = np.atleast_1d(iternum)*delta_t
    time_attrs = {'standard_name': 'time', 'long_name': 'Time', 'axis': 'T'}
    if ref_date is not None:
        time_attrs['units'] = 'seconds since %s' % ref_date
    else:
        time_attrs['units'] = 'seconds'
    if calendar is not None:
        time_attrs['calendar'] = calendar
    timevar = xr.Variable((time_dim_name,), timedata, time_attrs)
    return timevar
