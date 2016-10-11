"""
Utility functions for reading MITgcm mds files (.meta / .data)
"""
# python 3 compatiblity
from __future__ import print_function, division

import re
import os
import numpy as np
import warnings
from functools import reduce


def parse_meta_file(fname):
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
        match = re.match('(\w+) = (\[|\{)(.*)(\]|\})', item, re.DOTALL)
        if match:
            key, _, value, _ = match.groups()
            # remove more whitespace
            value = re.sub('^\s+', '', value)
            value = re.sub('\s+$', '', value)
            # print key,':', value
            flds[key] = value
    # now check the needed things are there
    needed_keys = ['dimList', 'nDims', 'nrecords', 'dataprec']
    for k in needed_keys:
        assert k in flds
    # transform datatypes
    flds['nDims'] = int(flds['nDims'])
    flds['nrecords'] = int(flds['nrecords'])
    # endianness is set by _read_mds
    flds['dataprec'] = np.dtype(re.sub("'", '', flds['dataprec']))
    flds['dimList'] = [[int(h) for h in
                       re.split(',', g)] for g in
                       re.split(',\n', flds['dimList'])]
    if 'fldList' in flds:
        flds['fldList'] = [re.match("'*(\w+)", g).groups()[0] for g in
                           re.split("'\s+'", flds['fldList'])]
        assert flds['nrecords'] == len(flds['fldList'])
    return flds


def read_mds(fname, iternum=None, use_mmap=True,
             force_dict=True, endian='>'):
    """Read an MITgcm .meta / .data file pair"""

    if iternum is None:
        istr = ''
    else:
        assert isinstance(iternum, int)
        istr = '.%010d' % iternum
    datafile = fname + istr + '.data'
    metafile = fname + istr + '.meta'

    # get metadata
    meta = parse_meta_file(metafile)
    # why does the .meta file contain so much repeated info?
    # just get the part we need
    # and reverse order (numpy uses C order, mds is fortran)
    shape = [g[0] for g in meta['dimList']][::-1]
    assert len(shape) == meta['nDims']
    # now add an extra for number of recs
    nrecs = meta['nrecords']
    shape.insert(0, nrecs)

    # load and shape data
    dtype = meta['dataprec'].newbyteorder(endian)
    d = read_raw_data(datafile, dtype, shape, use_mmap=use_mmap)

    if nrecs == 1:
        if 'fldList' in meta:
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


def read_raw_data(datafile, dtype, shape, use_mmap=False):
    """Read a raw binary file and shape it."""

    # first check to be sure there is the right number of bytes in the file
    number_of_values = reduce(lambda x, y: x * y, shape)
    expected_number_of_bytes = number_of_values * dtype.itemsize
    actual_number_of_bytes = os.path.getsize(datafile)
    if expected_number_of_bytes != actual_number_of_bytes:
        raise IOError('File `%s` does not have the correct size '
                      '(expected %g, found %g)' %
                      (datafile,
                       expected_number_of_bytes,
                       actual_number_of_bytes))
    if use_mmap:
        # print("Reading %s using memmap" % datafile)
        d = np.memmap(datafile, dtype, 'r')
    else:
        # print("Reading %s using fromfile" % datafile)
        d = np.fromfile(datafile, dtype)
    d.shape = shape
    return d


def parse_available_diagnostics(fname, layers={}):
    """Examine the available_diagnostics.log file and translate it into
    useful variable metadata.

    PARAMETERS
    ----------
    fname : str
        the path to the diagnostics file
    layers : dict (optional)
        dictionary mapping layers names to dimension sizes

    RETURNS
    -------
    all_diags : a dictionary keyed by variable names with values
        (coords, description, units)
    """
    all_diags = {}
    diag_id_lookup = {}
    mate_lookup = {}

    # mapping between the available_diagnostics.log codes and the actual
    # coordinate names
    # http://mitgcm.org/public/r2_manual/latest/online_documents/node268.html
    xcoords = {'U': 'i_g', 'V': 'i', 'M': 'i', 'Z': 'i_g'}
    ycoords = {'U': 'j', 'V': 'j_g', 'M': 'j', 'Z': 'j_g'}
    rcoords = {'M': 'k', 'U': 'k_u', 'L': 'k_l'}

    with open(fname) as f:
        # will automatically skip first four header lines
        for l in f:
            c = re.split('\|', l)
            if len(c) == 7 and c[0].strip() != 'Num':
                # parse the line to extract the relevant variables
                key = c[1].strip()
                diag_id = int(c[0].strip())
                diag_id_lookup[diag_id] = key
                levs = int(c[2].strip())
                mate = c[3].strip()
                if mate:
                    mate = int(mate)
                    mate_lookup[key] = mate
                code = c[4]
                units = c[5].strip()
                desc = c[6].strip()

                # decode what those variables mean
                hpoint = code[1]
                rpoint = code[8]
                xycoords = [ycoords[hpoint], xcoords[hpoint]]
                rlev = code[9]

                if rlev == '1' and levs == 1:
                    zcoord = []
                elif rlev == 'R':
                    zcoord = [rcoords[rpoint]]
                elif rlev == 'X' and layers:
                    layer_name = key.ljust(8)[-4:].strip()
                    n_layers = layers[layer_name]
                    if levs == n_layers:
                        suffix = 'bounds'
                    elif levs == (n_layers-1):
                        suffix = 'center'
                    elif levs == (n_layers-2):
                        suffix = 'interface'
                    else:
                        suffix = None
                        warnings.warn("Could not match rlev = %g to a layers"
                                      "coordiante" % rlev)
                    # dimname = ('layer_' + layer_name + '_' + suffix if suffix
                    dimname = (('l' + layer_name[0] + '_' + suffix[0]) if suffix
                               else '_UNKNOWN_')
                    zcoord = [dimname]
                else:
                    warnings.warn("Not sure what to do with rlev = " + rlev)
                    zcoord = ['_UNKNOWN_']
                coords = zcoord + xycoords
                all_diags[key] = dict(dims=coords,
                                      # we need a standard name
                                      attrs={'standard_name': key,
                                             'long_name': desc,
                                             'units': units})
    # add mate information
    for key, mate_id in mate_lookup.items():
        all_diags[key]['attrs']['mate'] = diag_id_lookup[mate_id]
    return all_diags


def llc_face_shape(llc_id):
    """Given an integer identifier for the llc grid, return the face shape."""

    # known valid LLC configurations
    if llc_id in (90, 270, 1080, 2160, 4320):
        return (llc_id, llc_id)
    else:
        raise ValueError("%g is not a valid llc identifier" % llc_id)

def llc_data_shape(llc_id, nz=None):
    """Given an integer identifier for the llc grid, and possibly a number of
    vertical grid points, return the expected shape of the full data field."""

    # this is a constant for all LLC setups
    NUM_FACES = 13

    tile_shape = llc_face_shape(llc_id)
    data_shape = (NUM_FACES,) + face_shape
    if nz is not None:
        data_shape = (nz,) + data_shape

    # should we accomodate multiple records?
    # no, not in this function
    return data_shape
