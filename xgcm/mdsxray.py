import operator
from glob import glob
import os
import re
import warnings
import numpy as np

import dask.array as da

import xray
from xray import Variable
from xray import backends #.common import AbstractDataStore
from xray import core
#core.utils.NDArrayMixin
#core.pycompat.OrderedDict
#core.indexing.NumpyIndexingAdapter

#from ..conventions import pop_to, cf_encoder
#from ..core import indexing
#from ..core.utils import (FrozenOrderedDict, NDArrayMixin,
#                          close_on_error, is_remote_uri)
#from ..core.pycompat import iteritems, basestring, OrderedDict

#from .common import AbstractWritableDataStore, robust_getitem

# This lookup table maps from dtype.byteorder to a readable endian
# string used by netCDF4.
_endian_lookup = {'=': 'native',
                  '>': 'big',
                  '<': 'little',
                  '|': 'native'}

# the variable metadata will be stored in dicts of the form
#_variable[varname] = (dimensions, description, units)

_grid_variables = xray.core.pycompat.OrderedDict(
    # horizontal grid
    X=   (('X',), "X-coordinate of cell center", "meters"),
    Y=   (('Y',), "Y-coordinate of cell center", "meters"),
    Xp1= (('Xp1',), "X-coordinate of cell corner", "meters"),
    Yp1= (('Yp1',), "Y-coordinate of cell corner", "meters"),
    # 2d versions
    XC=  (('Y','X'), "X coordinate of cell center (T-P point)", "degree_east"),
    YC=  (('Y','X'), "Y coordinate of cell center (T-P point)", "degree_north"),
    XG=  (('Yp1','Xp1'), "X coordinate of cell corner (Vorticity point)", "degree_east"),
    YG=  (('Yp1','Xp1'), "Y coordinate of cell corner (Vorticity point)", "degree_north"),
    # vertical grid
    Z=   (('Z',), "vertical coordinate of cell center", "meters"),
    Zp1= (('Zp1',), "vertical coordinate of cell interface", "meters"),
    Zu=  (('Zu',), "vertical coordinate of lower cell interface", "meters"),
    Zl=  (('Zl',), "vertical coordinate of upper cell interface", "meters"),
    # (for some reason, the netCDF files use both R and Z notation )
#    'RC':  (('Z',), "R coordinate of cell center", "m"),
#    'RF':  (('Zp1',), "R coordinate of cell interface", "m"),
#    'RU':  (('Zu',), "R coordinate of lower cell interface", "m"),
#    'RL':  (('Zl',), "R coordinate of upper cell interface", "m"),
    # horiz. differentials
    dxC= (('Y','Xp1'), "x cell center separation", "meters"),
    dyC= (('Yp1','X'), "y cell center separation", "meters"),
    dxG= (('Yp1','X'), "x cell corner separation", "meters"),
    dyG= (('Y','Xp1'), "y cell corner separation", "meters"),
    # vert. differentials
    drC= (('Zp1',), "r cell center separation", "m"),
    drF= (('Z',), "r cell face separation", "m"),
    # areas
    rA=  (('Y','X'), "r-face area at cell center", "m^2"),
    rAw= (('Y','Xp1'), "r-face area at U point", "m^2"),
    rAs= (('Yp1','X'), "r-face area at V point", "m^2"),
    rAz= (('Yp1','Xp1'), "r-face area at cell corner", "m^2"),
    # depth
    Depth=(('Y','X'), "fluid thickness in r coordinates (at rest)", "meters"),
    # thickness factors
    HFacC=(('Z','Y','X'),
             "vertical fraction of open cell at cell center", "none (0-1)"),
    HFacW=(('Z','Y','Xp1'),
             "vertical fraction of open cell at West face", "none (0-1)"),
    HFacS=(('Z','Yp1','X'),
             "vertical fraction of open cell at South face", "none (0-1)"),
    PHrefC=(('Z',), 'Reference Hydrostatic Pressure', 'm^2/s^2'),
    PHrefF=(('Zp1',), 'Reference Hydrostatic Pressure', 'm^2/s^2')
)

_grid_special_mapping = {
# name: (file_name, slice_to_extract, expecting_3D_field)
    'Z': ('RC', (slice(None),0,0), 3),
    'Zp1': ('RF', (slice(None),0,0), 3),
    'Zu': ('RF', (slice(1,None),0,0), 3),
    'Zl': ('RF', (slice(None,-1),0,0), 3),
    # this will create problems with some curvillinear grids
    # whate if X and Y need to be 2D?
    'X': ('XC', (0,slice(None)), 2),
    'Y': ('YC', (slice(None),0), 2),
    'Xp1': ('XG', (0,slice(None)), 2),
    'Yp1': ('YG', (slice(None),0), 2),
    'rA': ('RAC', (slice(None), slice(None)), 2),
    'HFacC': ('hFacC', 3*(slice(None),), 3),
    'HFacW': ('hFacW', 3*(slice(None),), 3),
    'HFacS': ('hFacS', 3*(slice(None),), 3),
}

_state_variables = xray.core.pycompat.OrderedDict(
    # state
    U=  (('Z','Y','Xp1'), 'Zonal Component of Velocity', 'm/s'),
    V=  (('Z','Yp1','X'), 'Meridional Component of Velocity', 'm/s'),
    W=  (('Zl','Y','X'), 'Vertical Component of Velocity', 'm/s'),
    T=  (('Z','Y','X'), 'Potential Temperature', 'degC'),
    S=  (('Z','Y','X'), 'Salinity', 'psu'),
    PH= (('Z','Y','X'), 'Hydrostatic Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    PHL=(('Y','X'), 'Bottom Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    Eta=(('Y','X'), 'Surface Height Anomaly', 'm'),
    # tave
    uVeltave=(('Z','Y','Xp1'), 'Zonal Component of Velocity', 'm/s'),
    vVeltave=(('Z','Yp1','X'), 'Meridional Component of Velocity', 'm/s'),
    wVeltave=(('Zl','Y','X'), 'Vertical Component of Velocity', 'm/s'),
    Ttave=(('Z','Y','X'), 'Potential Temperature', 'degC'),
    Stave=(('Z','Y','X'), 'Salinity', 'psu'),
    PhHytave=(('Z','Y','X'), 'Hydrostatic Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    PHLtave=(('Y','X'), 'Bottom Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    ETAtave=(('Y','X'), 'Surface Height Anomaly', 'm'),
    Convtave=(('Zl','Y','X'), "Convective Adjustment Index", "none [0-1]"),
    Eta2tave=(('Y','X'), "Square of Surface Height Anomaly", "m^2"),
    PHL2tave=(('Y','X'), 'Square of Hyd. Pressure Pot.(p/rho) Anomaly', 'm^4/s^4'),
    sFluxtave=(('Y','X'), 'total salt flux (match salt-content variations), >0 increases salt', 'g/m^2/s'),
    Tdiftave=(('Zl','Y','X'), "Vertical Diffusive Flux of Pot.Temperature", "degC.m^3/s"),
    tFluxtave=(('Y','X'), "Total heat flux (match heat-content variations), >0 increases theta", "W/m^2"),
    TTtave=(('Z','Y','X'), 'Squared Potential Temperature', 'degC^2'),
    uFluxtave=(('Y','Xp1'), 'surface zonal momentum flux, positive -> increase u', 'N/m^2'),
    UStave=(('Z','Y','Xp1'), "Zonal Transport of Salinity", "psu m/s"),
    UTtave=(('Z','Y','Xp1'), "Zonal Transport of Potenial Temperature", "degC m/s"),
    UUtave=(('Z','Y','Xp1'), "Zonal Transport of Zonal Momentum", "m^2/s^2"),
    UVtave=(('Z','Yp1','Xp1'), 'Product of meridional and zonal velocity', 'm^2/s^2'),
    vFluxtave=(('Yp1','X'), 'surface meridional momentum flux, positive -> increase v', 'N/m^2'),
    VStave=(('Z','Yp1','X'), "Meridional Transport of Salinity", "psu m/s"),
    VTtave=(('Z','Yp1','X'), "Meridional Transport of Potential Temperature", "degC m/s"),
    VVtave=(('Z','Yp1','X'), 'Zonal Transport of Zonal Momentum', 'm^2/s^2'),
    WStave=(('Zl','Y','X'), 'Vertical Transport of Salinity', "psu m/s"),
    WTtave=(('Zl','Y','X'), 'Vertical Transport of Potential Temperature', "degC m/s"),
)
# should find a better way to inlude the package variables
_state_variables['GM_Kwx-T'] = (
        ('Zl','Y','X'), 'K_31 element (W.point, X.dir) of GM-Redi tensor','m^2/s')
_state_variables['GM_Kwy-T'] = (
        ('Zl','Y','X'), 'K_33 element (W.point, X.dir) of GM-Redi tensor','m^2/s')
_state_variables['GM_Kwz-T'] = (
        ('Zl','Y','X'), 'K_33 element (W.point, X.dir) of GM-Redi tensor','m^2/s')


Nptracers=99
_ptracers = { 'PTRACER%02d' % n :
               (('Z','Y','X'), 'PTRACER%02d Concentration' % n, "tracer units/m^3")
               for n in range(Nptracers)}

def _read_and_shape_grid_data(k, dirname):
    if _grid_special_mapping.has_key(k):
        fname, sl, ndim_expected = _grid_special_mapping[k]
    else:
        fname = k
        sl = None
        ndim_expected = None
    data = None
    try:
        data = _read_mds(os.path.join(dirname, fname), force_dict=False)
    except IOError:
        try:
            data = _read_mds(os.path.join(dirname, fname.upper()),
                             force_dict=False)
        except IOError:
            warnings.warn("Couldn't load grid variable " + k)
    if data is not None:
        if sl is not None:
            # have to reslice and reshape the data
            if data.ndim != ndim_expected:
                # if there is only one vertical level, some variables
                # are squeeze at the mds level and need to get a dimension back
                if data.ndim==2 and ndim_expected==3:
                    data.shape = (1,) + data.shape
                else:
                    raise ValueError("Don't know how to handle data shape")
            # now apply the slice
            data = np.atleast_1d(data[sl])
        else:
            data = np.atleast_1d(data.squeeze())
        return data

def _force_native_endianness(var):
    # possible values for byteorder are:
    #     =    native
    #     <    little-endian
    #     >    big-endian
    #     |    not applicable
    # Below we check if the data type is not native or NA
    if var.dtype.byteorder not in ['=', '|']:
        # if endianness is specified explicitly, convert to the native type
        data = var.data.astype(var.dtype.newbyteorder('='))
        var = Variable(var.dims, data, var.attrs, var.encoding)
        # if endian exists, remove it from the encoding.
        var.encoding.pop('endian', None)
    # check to see if encoding has a value for endian its 'native'
    if not var.encoding.get('endian', 'native') is 'native':
        raise NotImplementedError("Attempt to write non-native endian type, "
                                  "this is not supported by the netCDF4 python "
                                  "library.")
    return var

def _parse_available_diagnostics(fname, Nlayers=None):
    """Examine the available_diagnostics.log file and translate it into
    useful variable metadata.

    PARAMETERS
    ----------
    fname : str
        the path to the diagnostics file
    Nlayers : int (optional)
        The size of the layers output. Used as a hint to decode vertical
        coordinate

    RETURNS
    -------
    all_diags : a dictionary keyed by variable names with values
        (coords, description, units)
    """
    all_diags = {}

    # mapping between the available_diagnostics.log codes and the actual
    # coordinate names
    # http://mitgcm.org/public/r2_manual/latest/online_documents/node268.html
    xcoords = {'U': 'Xp1', 'V': 'X', 'M': 'X', 'Z': 'Xp1'}
    ycoords = {'U': 'Y', 'V': 'Yp1', 'M': 'Y', 'Z': 'Yp1'}
    rcoords = {'M': 'Z', 'U': 'Zu', 'L': 'Zl'}

    with open(fname) as f:
        # will automatically skip first four header lines
        for l in f:
            c = re.split('\|',l)
            if len(c)==7 and c[0].strip()!='Num':

                # parse the line to extract the relevant variables
                key = c[1].strip()
                levs = int(c[2].strip())
                mate = c[3].strip()
                if mate: mate = int(mate)
                code = c[4]
                units = c[5].strip()
                desc = c[6].strip()

                # decode what those variables mean
                hpoint = code[1]
                rpoint = code[8]
                rlev = code[9]
                if rlev=='1' and levs==1:
                    coords = (ycoords[hpoint], xcoords[hpoint])
                elif rlev=='R':
                    coords = (rcoords[rpoint], ycoords[hpoint], xcoords[hpoint])
                elif rlev=='X' and (Nlayers is not None):
                    layers_suffix = key.ljust(8)[-4:].strip()
                    if levs==Nlayers:
                        lcoord = 'layers' + layers_suffix + '_bounds'
                    elif levs==(Nlayers-1):
                        lcoord = 'layers' + layers_suffix + '_center'
                    elif levs==(Nlayers-2):
                        lcoord = 'layers' + layers_suffix + '_interface'
                    else:
                        warnings.warn("Could not match rlev = %g to a layers"
                            "coordiante" % rlev)
                        lcoord = '_UNKNOWN_'
                    coords = (lcoord, ycoords[hpoint], xcoords[hpoint])
                else:
                    warnings.warn("Not sure what to do with rlev = " + rlev)
                    coords = (rcoords[rpoint], ycoords[hpoint], xcoords[hpoint])

                # don't need an object for this
                #dds = MITgcmDiagnosticDescription(
                #    key, code, units, desc, levs, mate)
                # return dimensions, description, units
                all_diags[key] = (coords, desc, units)
    return all_diags


def _decode_diagnostic_description(
        key, code, units=None, desc=None, levs=None, mate=None):
    """Convert parsed available_diagnostics line to tuple of
    coords, description, units."""

def _parse_meta(fname):
    """Get the metadata as a dict out of the mitGCM mds .meta file."""

    flds = {}
    basename = re.match('(^.+?)\..+', os.path.basename(fname)).groups()[0]
    flds['basename'] = basename
    with open(fname) as f:
        text = f.read()
    # split into items
    for item in re.split(';', text):
        # remove whitespace at beginning
        item = re.sub('^\s+', '', item)
        #match = re.match('(\w+) = ', item)
        match = re.match('(\w+) = (\[|\{)(.*)(\]|\})', item, re.DOTALL)
        if match:
            key, _, value, _ = match.groups()
            # remove more whitespace
            value = re.sub('^\s+', '', value)
            value = re.sub('\s+$', '', value)
            #print key,':', value
            flds[key] = value
    # now check the needed things are there
    needed_keys = ['dimList','nDims','nrecords','dataprec']
    for k in needed_keys:
        assert flds.has_key(k)
    # transform datatypes
    flds['nDims'] = int(flds['nDims'])
    flds['nrecords'] = int(flds['nrecords'])
    # use big endian always
    flds['dataprec'] = np.dtype(re.sub("'",'',flds['dataprec'])).newbyteorder('>')
    flds['dimList'] = [[int(h) for h in
                       re.split(',', g)] for g in
                       re.split(',\n',flds['dimList'])]
    if flds.has_key('fldList'):
        flds['fldList'] = [re.match("'*(\w+)",g).groups()[0] for g in
                           re.split("'\s+'",flds['fldList'])]
        assert flds['nrecords'] == len(flds['fldList'])
    return flds

def _read_mds(fname, iternum=None, use_mmap=True,
             force_dict=True, convert_big_endian=False):
    """Read an MITgcm .meta / .data file pair"""

    if iternum is None:
        istr = ''
    else:
        assert isinstance(iternum, int)
        istr = '.%010d' % iternum
    datafile = fname + istr + '.data'
    metafile = fname + istr + '.meta'

    # get metadata
    meta = _parse_meta(metafile)
    # why does the .meta file contain so much repeated info?
    # just get the part we need
    # and reverse order (numpy uses C order, mds is fortran)
    shape = [g[0] for g in meta['dimList']][::-1]
    assert len(shape) == meta['nDims']
    # now add an extra for number of recs
    nrecs = meta['nrecords']
    shape.insert(0, nrecs)

    # load and shape data
    if use_mmap:
        d = np.memmap(datafile, meta['dataprec'], 'r')
    else:
        d = np.fromfile(datafile, meta['dataprec'])
    if convert_big_endian:
        dtnew = d.dtype.newbyteorder('=')
        d = d.astype(dtnew)

    d.shape = shape

    if nrecs == 1:
        if meta.has_key('fldList'):
            name = meta['fldList'][0]
        else:
            name = meta['basename']
        if force_dict:
            return {name: d[0]}
        else:
            return d[0]
    else:
        # need record names
        out = {}
        for n, name in enumerate(meta['fldList']):
            out[name] = d[n]
        return out


class MDSArrayWrapper(xray.core.utils.NDArrayMixin):
    def __init__(self, array):
        self.array = array

    @property
    def dtype(self):
        dtype = self.array.dtype

def _list_all_mds_files(dirname):
    """Find all the meta / data files"""
    files = glob(os.path.join(dirname, '*.meta'))
    # strip the suffix
    return [f[:-5] for f in files]

_layers_desc_and_units = dict(
    TH = ('potential temperature layer', 'deg. C'),
    SA = ('salinity layer', 'PSU'),
    RHO = ('potential density layer', 'kg / m^3')
)

# varname, dims, desc, units, data in _get_layers_grid_variables()
def _get_layers_grid_variables(dirname):
    """Look for special layers mds files describing the layers grid."""
    files = glob(os.path.join(dirname, 'layers[0-9]*.meta'))
    for f in files:
        varname = os.path.basename(f[:-5])
        # varname will be something like 'layers1TH'
        desc, units = _layers_desc_and_units[varname[7:]]
        data = _read_mds(os.path.join(dirname, varname),
                force_dict=False).squeeze()
        Nlayers = len(data)
        # construct the three different layers coordinates
        yield (varname + '_bounds', (varname + '_bounds'),
               desc + ' boundaries', units, data)
        yield (varname + '_center', (varname + '_center'),
               desc + ' centers', units, 0.5*(data[:-1]+data[1:]))
        yield (varname + '_interface', (varname + '_interface'),
               desc + ' interfaces', units, data[1:-1])


#class MemmapArrayWrapper(NumpyIndexingAdapter):
class MemmapArrayWrapper(xray.core.utils.NDArrayMixin):
    def __init__(self, memmap_array):
        self._memmap_array = memmap_array

    @property
    def array(self):
        # We can't store the actual netcdf_variable object or its data array,
        # because otherwise scipy complains about variables or files still
        # referencing mmapped arrays when we try to close datasets without
        # having read all data in the file.
        return self._memmap_array

    @property
    def dtype(self):
        return self._memmap_array.dtype

    def __getitem__(self, key):
        data = self._memmap_array.__getitem__(key)
        return np.asarray(data)

_valid_geometry = ['Cartesian', 'SphericalPolar']

def open_mdsdataset(dirname, iters=None, deltaT=1,
                 prefix=None, ref_date=None, calendar=None,
                 ignore_pickup=True, geometry='Cartesian',
                 grid_vars_to_coords=True):
    """Open MITgcm-style mds file output as xray datset."""

    store = _MDSDataStore(dirname, iters, deltaT,
                             prefix, ref_date, calendar,
                             ignore_pickup, geometry)
    # turn all the auxilliary grid variables into coordinates
    ds = xray.Dataset.load_store(store)
    if grid_vars_to_coords:
        for k in _grid_variables:
            ds.set_coords(k, inplace=True)
        ds.set_coords('iter', inplace=True)
    return ds


class _MDSDataStore(backends.common.AbstractDataStore):
    """Represents the entire directory of MITgcm mds output
    including all grid variables. Similar in some ways to
    netCDF.Dataset."""
    def __init__(self, dirname, iters=None, deltaT=1,
                 prefix=None, ref_date=None, calendar=None,
                 ignore_pickup=True, geometry='Cartesian'):
        """iters: list of iteration numbers
        deltaT: timestep
        prefix: list of file prefixes (if None use all)
        """
        assert geometry in _valid_geometry
        self.geometry = geometry

        # the directory where the files live
        self.dirname = dirname

        # storage dicts for variables and attributes
        self._variables = xray.core.pycompat.OrderedDict()
        self._attributes = xray.core.pycompat.OrderedDict()
        self._dimensions = []

        ### figure out the mapping between diagnostics names and variable properties

        ### read grid files
        for k in _grid_variables:
            dims, desc, units = _grid_variables[k]
            data = _read_and_shape_grid_data(k, dirname)
            if data is not None:
                self._variables[k] = Variable(
                    dims, MemmapArrayWrapper(data),
                    {'description': desc, 'units': units})
                self._dimensions.append(k)

        ## check for layers
        Nlayers = None
        for varname, dims, desc, units, data in _get_layers_grid_variables(dirname):
            self._variables[varname] = Variable(
                    dims, MemmapArrayWrapper(data),
                    {'description': desc, 'units': units})
            self._dimensions.append(varname)
            # if there are multiple layers coordinates, they all have the same
            # size, so this works (although it is sloppy)
            if varname[-7:]=='_bounds':
                Nlayers = len(data)

        ## load metadata for all possible diagnostics
        diag_meta = _parse_available_diagnostics(
                os.path.join(dirname, 'available_diagnostics.log'),
                Nlayers=Nlayers)

        # now get variables from our iters
        if iters is not None:

            # create iteration array
            iterdata = np.asarray(iters)
            self._variables['iter'] = Variable(('time',), iterdata,
                                                {'description': 'model timestep number'})

            # create time array
            timedata = np.asarray(iters)*deltaT
            time_attrs = {'description': 'model time'}
            if ref_date is not None:
                time_attrs['units'] = 'seconds since %s' % ref_date
            else:
                time_attrs['units'] = 'seconds'
            if calendar is not None:
                time_attrs['calendar'] = calendar
            self._variables['time'] = Variable(
                                        ('time',), timedata, time_attrs)
            self._dimensions.append('time')

            varnames = []
            fnames = []
            _data_vars = xray.core.pycompat.OrderedDict()
            # look at first iter to get variable metadata
            for f in glob(os.path.join(dirname, '*.%010d.meta' % iters[0])):
                if ignore_pickup and re.search('pickup', f):
                    pass
                else:
                    go = True
                    if prefix is not None:
                        bname = os.path.basename(f[:-16])
                        matches = [bname==p for p in prefix]
                        if not any(matches):
                            go = False
                    if go:
                        meta = _parse_meta(f)
                        if meta.has_key('fldList'):
                            flds = meta['fldList']
                            [varnames.append(fl) for fl in flds]
                        else:
                            varnames.append(meta['basename'])
                        fnames.append(os.path.join(dirname,meta['basename']))

            # read data as dask arrays (should be an option)
            vardata = {}
            for k in varnames:
                vardata[k] = []
            for i in iters:
                for f in fnames:
                    try:
                        data = _read_mds(f, i, force_dict=True)
                        # this can screw up if the same variable appears in
                        # multiple diagnostic files
                        for k in data.keys():
                            mwrap = MemmapArrayWrapper(data[k])
                            # for some reason, da.from_array does not
                            # necessarily give a unique name
                            # need to specify array name
                            myda = da.from_array(mwrap, mwrap.shape,
                                    name='%s_%010d' % (k, i))
                            vardata[k].append(myda)
                    except IOError:
                        # couldn't find the variable, remove it from the list
                        #print 'Removing %s from list (iter %g)' % (k, i)
                        varnames.remove(k)

            # final loop to create Variable objects
            for k in varnames:
                try:
                    dims, desc, units = _state_variables[k]
                except KeyError:
                    try:
                        dims, desc, units = _ptracers[k]
                    except KeyError:
                        dims, desc, units = diag_meta[k]

                # check for shape compatability
                varshape = vardata[k][0].shape
                varndims = len(varshape)
                # maybe promote 2d data to 3d
                if (len(dims)==3) and (varndims==2):
                    if len(self._variables[dims[0]])==1:
                        vardata[k] = \
                            [v.reshape((1,) + varshape) for v in vardata[k]]
                        warnings.warn('Promiting 2D data to 3D data '
                                      'for variable %s' % k)
                        varndims += 1
                if len(dims) != varndims:
                    warnings.warn("Shape of variable data is not compatible "
                                  "with expected number of dimensions. This "
                                  "can arise if the 'levels' option is used "
                                  "in data.diagnostics. Right now we have no "
                                  "way to infer the level, so the variable is "
                                  "skipped: " + k)
                else:
                    # add time to dimension
                    dims_time = ('time',) + dims
                    # wrap variable in dask array
                    # -- why? it's already a dask array
                    #vardask = da.stack([da.from_array(d, varshape) for d in vardata[k]])
                    vardask = da.stack(vardata[k])
                    #for nkdsk in range(len(vardata[k])):
                    #    print 'Key %s, vardata[%g] sum %g, name %s' % (k, nkdsk,
                    #        vardata[k][nkdsk].sum(), vardata[k][nkdsk].name)
                    #    print 'Key %s, vardask[%g] sum %g' % (k, nkdsk,
                    #        vardask[nkdsk].sum())
                    newvar = Variable( dims_time, vardask,
                                       {'description': desc, 'units': units})
                    self._variables[k] = newvar

        self._attributes = {'history': 'Some made up attribute'}


    def get_variables(self):
        return self._variables

    def get_attrs(self):
        return self._attributes

    def get_dimensions(self):
        return self._dimensions

    def close(self):
        pass


# from MITgcm netCDF grid file
# dimensions:
# Z = 30 ;
# Zp1 = 31 ;
# Zu = 30 ;
# Zl = 30 ;
# X = 25 ;
# Y = 40 ;
# Xp1 = 26 ;
# Yp1 = 41 ;
# variables:
# double Z(Z) ;
#     Z:long_name = "vertical coordinate of cell center" ;
#     Z:units = "meters" ;
#     Z:positive = "up" ;
# double RC(Z) ;
#     RC:description = "R coordinate of cell center" ;
#     RC:units = "m" ;
# double Zp1(Zp1) ;
#     Zp1:long_name = "vertical coordinate of cell interface" ;
#     Zp1:units = "meters" ;
#     Zp1:positive = "up" ;
# double RF(Zp1) ;
#     RF:description = "R coordinate of cell interface" ;
#     RF:units = "m" ;
# double Zu(Zu) ;
#     Zu:long_name = "vertical coordinate of lower cell interface" ;
#     Zu:units = "meters" ;
#     Zu:positive = "up" ;
# double RU(Zu) ;
#     RU:description = "R coordinate of upper interface" ;
#     RU:units = "m" ;
# double Zl(Zl) ;
#     Zl:long_name = "vertical coordinate of upper cell interface" ;
#     Zl:units = "meters" ;
#     Zl:positive = "up" ;
# double RL(Zl) ;
#     RL:description = "R coordinate of lower interface" ;
#     RL:units = "m" ;
# double drC(Zp1) ;
#     drC:description = "r cell center separation" ;
# double drF(Z) ;
#     drF:description = "r cell face separation" ;
# double X(X) ;
#     X:long_name = "X-coordinate of cell center" ;
#     X:units = "meters" ;
# double Y(Y) ;
#     Y:long_name = "Y-Coordinate of cell center" ;
#     Y:units = "meters" ;
# double XC(Y, X) ;
#     XC:description = "X coordinate of cell center (T-P point)" ;
#     XC:units = "degree_east" ;
# double YC(Y, X) ;
#     YC:description = "Y coordinate of cell center (T-P point)" ;
#     YC:units = "degree_north" ;
# double Xp1(Xp1) ;
#     Xp1:long_name = "X-Coordinate of cell corner" ;
#     Xp1:units = "meters" ;
# double Yp1(Yp1) ;
#     Yp1:long_name = "Y-Coordinate of cell corner" ;
#     Yp1:units = "meters" ;
# double XG(Yp1, Xp1) ;
#     XG:description = "X coordinate of cell corner (Vorticity point)" ;
#     XG:units = "degree_east" ;
# double YG(Yp1, Xp1) ;
#     YG:description = "Y coordinate of cell corner (Vorticity point)" ;
#     YG:units = "degree_north" ;
# double dxC(Y, Xp1) ;
#     dxC:description = "x cell center separation" ;
# double dyC(Yp1, X) ;
#     dyC:description = "y cell center separation" ;
# double dxF(Y, X) ;
#     dxF:description = "x cell face separation" ;
# double dyF(Y, X) ;
#     dyF:description = "y cell face separation" ;
# double dxG(Yp1, X) ;
#     dxG:description = "x cell corner separation" ;
# double dyG(Y, Xp1) ;
#     dyG:description = "y cell corner separation" ;
# double dxV(Yp1, Xp1) ;
#     dxV:description = "x v-velocity separation" ;
# double dyU(Yp1, Xp1) ;
#     dyU:description = "y u-velocity separation" ;
# double rA(Y, X) ;
#     rA:description = "r-face area at cell center" ;
# double rAw(Y, Xp1) ;
#     rAw:description = "r-face area at U point" ;
# double rAs(Yp1, X) ;
#     rAs:description = "r-face area at V point" ;
# double rAz(Yp1, Xp1) ;
#     rAz:description = "r-face area at cell corner" ;
# double fCori(Y, X) ;
#     fCori:description = "Coriolis f at cell center" ;
# double fCoriG(Yp1, Xp1) ;
#     fCoriG:description = "Coriolis f at cell corner" ;
# double R_low(Y, X) ;
#     R_low:description = "base of fluid in r-units" ;
# double Ro_surf(Y, X) ;
#     Ro_surf:description = "surface reference (at rest) position" ;
# double Depth(Y, X) ;
#     Depth:description = "fluid thickness in r coordinates (at rest)" ;
# double HFacC(Z, Y, X) ;
#     HFacC:description = "vertical fraction of open cell at cell center" ;
# double HFacW(Z, Y, Xp1) ;
#     HFacW:description = "vertical fraction of open cell at West face" ;
# double HFacS(Z, Yp1, X) ;
#     HFacS:description = "vertical fraction of open cell at South face" ;
