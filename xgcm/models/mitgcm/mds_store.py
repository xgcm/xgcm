"""
Class to represent MITgcm mds file storage format.
"""
# python 3 compatiblity
from __future__ import print_function, division

from glob import glob
import os
import re
import numpy as np
import inspect
import xarray as xr
import dask.array as da

# we keep the metadata in its own module to keep this one cleaner
from .variables import dimensions, \
    horizontal_coordinates_spherical, horizontal_coordinates_cartesian, \
    vertical_coordinates, horizontal_grid_variables, vertical_grid_variables, \
    volume_grid_variables, state_variables
# would it be better to import mitgcm_variables and then automate the search
# for variable dictionaries

from .utils import parse_meta_file, read_mds, parse_available_diagnostics

# should we hard code this?
LLC_NUM_FACES = 13
LLC_FACE_DIMNAME = 'face'

def open_mdsdataset(dirname, iters='all', prefix=None, read_grid=True,
                    delta_t=1, ref_date=None, calendar='gregorian',
                    geometry='sphericalpolar',
                    grid_vars_to_coords=True, swap_dims=False,
                    endian=">", chunks=None,
                    ignore_unknown_vars=False,):
    """Open MITgcm-style mds (.data / .meta) file output as xarray datset.

    Parameters
    ----------
    dirname : string
        Path to the directory where the mds .data and .meta files are stored
    iters : list, optional
        The iterations numbers of the files to be read. If `None`, no data
        files will be read.
    prefix : list, optional
        List of different filename prefixes to read. Default is to read all
        available files.
    read_grid : bool, optional
        Whether to read the grid data
    deltaT : number, optional
        The timestep used in the model. (Can't be inferred.)
    ref_date : string, optional
        A date string corresponding to the zero timestep. E.g. "1990-1-1 0:0:0".
        See CF conventions [1]_
    calendar : string, optional
        A calendar allowed by CF conventions [1]_
    geometry : {'sphericalpolar', 'cartesian', 'llc'}
        MITgcm grid geometry specifier.
    swap_dims : boolean, optional
        Whether to swap the logical dimensions for physical ones.
    endian : {'=', '>', '<'}, optional
        Endianness of variables. Default for MITgcm is ">" (big endian)
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask arrays.
    ignore_unknown_vars : boolean, optional
        Don't raise an error if unknown variables are encountered while reading
        the dataset.

    Returns
    -------
    dset : xarray.Dataset
        Dataset object containing all coordinates and variables.

    References
    ----------
    .. [1] http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch04s04.html
    """

    # get frame info for history
    frame = inspect.currentframe()
    _, _, _, arg_values = inspect.getargvalues(frame)
    del arg_values['frame']
    function_name = inspect.getframeinfo(frame)[2]

    # some checks for argument consistency
    if swap_dims and not read_grid:
        raise ValueError("If swap_dims==True, read_grid must be True.")

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
                        geometry=geometry,
                        grid_vars_to_coords=grid_vars_to_coords,
                        endian=endian, chunks=chunks,
                        ignore_unknown_vars=ignore_unknown_vars)
                    for iternum in iters]
                # now add the grid
                if read_grid:
                    datasets.insert(0, open_mdsdataset(
                        dirname, iters=None, delta_t=delta_t,
                        read_grid=True, swap_dims=False,
                        prefix=prefix, ref_date=ref_date, calendar=calendar,
                        geometry=geometry,
                        grid_vars_to_coords=grid_vars_to_coords,
                        endian=endian, chunks=chunks,
                        ignore_unknown_vars=ignore_unknown_vars))
                # apply chunking
                ds = xr.auto_combine(datasets)
                if swap_dims:
                    ds = _swap_dimensions(ds, geometry)
                return ds

    store = _MDSDataStore(dirname, iternum, delta_t, read_grid,
                          prefix, ref_date, calendar,
                          geometry, endian,
                          ignore_unknown_vars=ignore_unknown_vars)
    ds = xr.Dataset.load_store(store)

    if swap_dims:
        ds = _swap_dimensions(ds, geometry)

    if grid_vars_to_coords:
        ds = _set_coords(ds)

    # turn all the auxilliary grid variables into coordinates
    # if grid_vars_to_coords:
    #     for k in _grid_variables:
    #         ds.set_coords(k, inplace=True)
    #     ds.set_coords('iter', inplace=True)

    if ref_date:
        ds = xr.decode_cf(ds)

    # do we need more fancy logic (like open_dataset), or is this enough
    if chunks is not None:
        ds = ds.chunk(chunks)

    # set attributes for CF conventions
    ds.attrs['Conventions'] = "CF-1.6"
    ds.attrs['title'] = "netCDF wrapper of MITgcm MDS binary data"
    ds.attrs['source'] = "MITgcm"
    arg_string = ', '.join(['%s=%s' % (str(k), repr(v))
                            for (k, v) in arg_values.items()])
    ds.attrs['history'] = ('Created by calling '
                           '`%s(%s)`'% (function_name, arg_string))

    return ds


def _set_coords(ds):
    """Turn all variables without `time` dimensions into coordinates."""
    coords = set()
    for vname in ds:
        if ('time' not in ds[vname].dims) or (ds[vname].dims == ('time',)):
            coords.add(vname)
    return ds.set_coords(list(coords))


def _swap_dimensions(ds, geometry, drop_old=True):
    """Replace logical coordinates with physical ones. Does not work for llc.
    """
    # TODO: handle metadata correctly such that the new dimension attributes
    # still conform to comodo conventions

    if geometry.lower() == 'llc':
        raise ValueError("Can't swap dimensions if `geometry` is `llc`")

    # first squeeze all the coordinates
    for orig_dim in ds.dims:
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
    model instance and a specific timestep iteration number."""
    def __init__(self, dirname, iternum=None, delta_t=1, read_grid=True,
                 file_prefixes=None, ref_date=None, calendar=None,
                 geometry='sphericalpolar',
                 endian='>', ignore_unknown_vars=False):
        """
        This is not a user-facing class. See open_mdsdataset for argument
        documentation. The only ones which are distinct are.

        Parameters
        ----------
        iternum : int, optional
            The iteration timestep number to read.
        file_prefixes : list
            The prefixes of the data files to be read.
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

        # The endianness of the files
        # By default, MITgcm does big endian
        if endian not in ['>', '<', '=']:
            raise ValueError("Invalid byte order (endian=%s)" % endian)
        self.endian = endian

        # storage dicts for variables and attributes
        self._variables = xr.core.pycompat.OrderedDict()
        self._attributes = xr.core.pycompat.OrderedDict()
        self._dimensions = []

        # the dimensions are theoretically the same for all datasets
        [self._dimensions.append(k) for k in dimensions]
        self.llc = (self.geometry == 'llc')

        # TODO: and maybe here a check for the presence of layers?

        # Now we need to figure out the dimensions of the numerical domain,
        # i.e. nx, ny, nz. We do this by peeking at the grid file metadata
        self.nz, self.nface, self.ny, self.nx = (
            _guess_model_dimensions(dirname, self.llc))
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

        # possibly add the llc dimension
        # seems sloppy to hard code this here
        # TODO: move this metadata to variables.py
        if self.llc:
            self._dimensions.append(LLC_FACE_DIMNAME)
            data = np.arange(self.nface)
            attrs = {'standard_name': 'face_index'}
            dims = [LLC_FACE_DIMNAME]
            self._variables[LLC_FACE_DIMNAME] = xr.Variable(dims, data, attrs)

        # do the same for layers
        for layer_name, n_layer in self.layers.items():
            for suffix, offset in zip(['bounds', 'center', 'interface'],
                                      [0, -1, -2]):
                # e.g. "layer_1RHO_bounds"
                # dimname = 'layer_' + layer_name + '_' + suffix
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
                iternum, delta_t, ref_date, calendar, self.time_dim_name
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
            prefixes = prefixes + list(self._all_grid_variables.keys())

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
        vardata = read_mds(os.path.join(self.dirname, fname_base), iternum,
                           endian=self.endian)
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

            # make sure we copy these things
            dims = list(metadata['dims'])
            attrs = dict(metadata['attrs'])

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

            if self.llc:
                dims, data = _reshape_for_llc(dims, data)

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


def _guess_model_dimensions(dirname, is_llc=False):
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
    if is_llc:
        nface = LLC_NUM_FACES
        ny /= nface
    else:
        nface = None
    return nz, nface, ny, nx


def _guess_layers(dirname):
    """Return a dict matching layers suffixes to dimension length."""
    layers_files = glob(os.path.join(dirname, 'layers*.meta'))
    all_layers = {}
    for fname in layers_files:
        # make sure to exclude filenames such as
        # "layers_surfflux.01.0000000001.meta"
        if not re.search('\.\d{10}\.', fname):
            # should turn "foo/bar/layers1RHO.meta" into "1RHO"
            layers_suf = os.path.splitext(os.path.basename(fname))[0][6:]
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
    # But it doesn't work in python3!!!
    extra_metadata = xr.core.pycompat.OrderedDict()
    for name, val in metadata.items():
        newname = name + '-T'
        extra_metadata[newname] = val
    metadata = _concat_dicts([metadata, extra_metadata])

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


def _reshape_for_llc(dims, data):
    """Take dims and data and return modified / reshaped dims and data for
    llc geometry."""

    # this won't work otherwise
    assert len(dims)==data.ndim

    # the only dimensions that get expanded into faces
    expand_dims = ['j', 'j_g']
    for dim in expand_dims:
        if dim in dims:
            # add face dimension to dims
            jdim = dims.index(dim)
            dims.insert(jdim, LLC_FACE_DIMNAME)
            data = _reshape_llc_data(data, jdim)
    return dims, data


def _reshape_llc_data(data, jdim):
    """Fix the weird problem with llc data array order."""
    # Can we do this without copying any data?
    # If not, we need to go upstream and implement this at the MDS level
    # Or can we fudge it with dask?
    # this is all very specific to the llc file output
    # would be nice to generalize more, but how?
    nside = data.shape[jdim] / LLC_NUM_FACES
    # how the LLC data is laid out along the j dimension
    strides = ((0,3), (3,6), (6,7), (7,10), (10,13))
    # whether to reshape each face
    reshape = (False, False, False, True, True)
    # this will slice the data into 5 facets
    slices = [jdim * (slice(None),) + (slice(nside*st[0], nside*st[1]),)
              for st in strides]
    facet_arrays = [data[sl] for sl in slices]
    face_arrays = []
    for ar, rs, st in zip(facet_arrays, reshape, strides):
        nfaces_in_facet = st[1] - st[0]
        shape = list(ar.shape)
        if rs:
            # we assume the other horizontal dimension is immediately after jdim
            shape[jdim] = ar.shape[jdim+1]
            shape[jdim+1] = ar.shape[jdim]
        # insert a length-1 dimension along which to concatenate
        shape.insert(jdim, 1)
        # modify the array shape in place, no copies allowed
        ar.shape = shape
        # now ar is propery shaped, but we still need to slice it into faces
        face_slice_dim = jdim + 1 + rs
        for n in range(nfaces_in_facet):
            face_slice = (face_slice_dim * (slice(None),) +
                          (slice(nside*n, nside*(n+1)),))
            data_face = ar[face_slice]
            face_arrays.append(data_face)

    # We can't concatenate using numpy (hcat etc.) because it makes a copy,
    # presumably loading the memmaps into memory.
    # Using dask gets around this.
    # But what if we want different chunks, or already chunked the data
    # upstream? Doesn't seem like this is ideal
    # TODO: Refactor handling of dask arrays and chunking
    #return np.concatenate(face_arrays, axis=jdim)
    # the dask version doesn't work because of this:
    # https://github.com/dask/dask/issues/1645
    face_arrays_dask = [da.from_array(fa, chunks=fa.shape)
                        for fa in face_arrays]
    concat = da.concatenate(face_arrays_dask, axis=jdim)
    return concat
