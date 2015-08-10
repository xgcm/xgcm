import operator
from glob import glob
import os
import re
import warnings
import numpy as np

import dask.array as da

from xray import Variable
from xray.backends.common import AbstractDataStore
from xray.core.utils import NDArrayMixin
from xray.core.pycompat import OrderedDict
from xray.core.indexing import NumpyIndexingAdapter 

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

_grid_variables = OrderedDict(
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
             "vertical fraction of open cell at South face", "none (0-1)")
)

_grid_special_mapping = {
    'Z': ('RC', (slice(None),0,0)),
    'Zp1': ('RF', (slice(None),0,0)),
    'Zu': ('RC', (slice(1,None),0,0)),
    'Zl': ('RF', (slice(None,-1),0,0)),
    'X': ('XC', (0,slice(None))),
    'Y': ('YC', (slice(None),0)),
    'Xp1': ('XG', (0,slice(None))),
    'Yp1': ('YG', (slice(None),0)),
    'rA': ('RAC', None),
    'HFacC': ('hFacC', None),
    'HFacW': ('hFacW', None),
    'HFacS': ('hFacS', None),    
}

_state_variables = OrderedDict(
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
    WTtave=(('Zl','Y','X'), 'Vertical Transport of Potential Temperature', "degC m/s")
)

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

def _parse_available_diagnostics(fname):
    all_diags = {}
    
    # add default diagnostics for grid, tave, and state
    
    with open(fname) as f:
        # will automatically skip first four header lines
        for l in f:
            c = re.split('\|',l)
            if len(c)==7 and c[0].strip()!='Num':
                key = c[1].strip()
                levs = int(c[2].strip())
                mate = c[3].strip()
                if mate: mate = int(mate) 
                code = c[4]
                units = c[5].strip()
                desc = c[6].strip()
                dds = MITgcmDiagnosticDescription(
                    key, code, units, desc, levs, mate)
                # return dimensions, description, units
                all_diags[key] = (dds.coords(), dds.units, dds.desc)
    return all_diags


class MITgcmDiagnosticDescription(object):
    
    def __init__(self, key, code, units=None, desc=None, levs=None, mate=None):
        self.key = key
        self.levs = levs
        self.mate = mate
        self.code = code
        self.units = units
        self.desc = desc
    
    def coords(self):
        """Parse code to determine coordinates."""
        hpoint = self.code[1]
        rpoint = self.code[8]
        rlev = self.code[9]
        xcoords = {'U': 'Xp1', 'V': 'X', 'M': 'X', 'Z': 'Xp1'}
        ycoords = {'U': 'Y', 'V': 'Yp1', 'M': 'Y', 'Z': 'Yp1'}
        rcoords = {'M': 'Z', 'U': 'Zu', 'L': 'Zl'}
        if rlev=='1' and self.levs==1:
            return (ycoords[hpoint], xcoords[hpoint])
        elif rlev=='R':
            return (rcoords[rpoint], ycoords[hpoint], xcoords[hpoint])
        else:
            warnings.warn("Not sure what to do with rlev = " + rlev)
            return (rcoords[rpoint], ycoords[hpoint], xcoords[hpoint])


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


class MDSArrayWrapper(NDArrayMixin):
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


#class MemmapArrayWrapper(NumpyIndexingAdapter):
class MemmapArrayWrapper(NDArrayMixin):    
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

class MDSDataStore(AbstractDataStore):
    """Represents the entire directory of MITgcm mds output
    including all grid variables. Similar in some ways to
    netCDF.Dataset."""
    def __init__(self, dirname, iters=None, deltaT=1,
                 prefix=None,
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
        self._variables = OrderedDict()
        self._attributes = OrderedDict()
        self._dimensions = []
 
        ### figure out the mapping between diagnostics names and variable properties
        # all possible diagnostics
        diag_meta = _parse_available_diagnostics(
                os.path.join(dirname, 'available_diagnostics.log'))

        ### read grid files
        for k in _grid_variables:
            if _grid_special_mapping.has_key(k):
                fname = _grid_special_mapping[k][0]
                sl = _grid_special_mapping[k][1]
            else:
                fname = k
                sl = None
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
                data = data[sl] if sl is not None else data.squeeze()
                dims, desc, units = _grid_variables[k]
                self._variables[k] = Variable(
                    dims, MemmapArrayWrapper(data), {'description': desc, 'units': units})
                self._dimensions.append(k)
                
        # now get variables from our iters
        if iters is not None:
            
            # create time array
            timedata = np.asarray(iters)*deltaT
            self._variables['time'] = Variable(
                                        ('time',), timedata,
                                        {'description': 'model time', 'units': 'seconds'})
            self._dimensions.append('time')
            
            varnames = []
            fnames = []
            _data_vars = OrderedDict()
            # look at first iter to get variable metadata
            for f in glob(os.path.join(dirname, '*.%010d.meta' % iters[0])):
                if ignore_pickup and re.search('pickup', f):
                    pass
                else:
                    go = True
                    if prefix is not None:
                        matches = [re.search(p, f) for p in prefix]
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
            
            # read data
            vardata = {}
            for k in varnames:
                vardata[k] = []
            for i in iters:
                for f in fnames:
                    try:
                        data = _read_mds(f, i, force_dict=True)
                        for k in data.keys():
                            vardata[k].append(MemmapArrayWrapper(data[k]))
                    except IOError:
                        # couldn't find the variable, remove it from the list
                        #print 'Removing %s from list (iter %g)' % (k, i)
                        varnames.remove(k)

            # final loop to create Variable objects
            for k in varnames:
                try:
                    dims, desc, units = _state_variables[k]
                except KeyError:
                    dims, desc, units = diag_meta[k]
                # check for shape compatability
                varshape = vardata[k][0].shape
                varndims = len(varshape)
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
                    vardask = da.stack([da.from_array(d, varshape) for d in vardata[k]])
                    self._variables[k] = Variable( dims_time, vardask,
                                                   {'description': desc, 'units': units})
                                        
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