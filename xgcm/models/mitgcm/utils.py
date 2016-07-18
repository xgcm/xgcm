"""
Utility functions for reading MITgcm mds files (.meta / .data)
"""
# python 3 compatiblity
from __future__ import print_function, division

import re
import os
import numpy as np

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
            #print key,':', value
            flds[key] = value
    # now check the needed things are there
    needed_keys = ['dimList','nDims','nrecords','dataprec']
    for k in needed_keys:
        assert k in flds
    # transform datatypes
    flds['nDims'] = int(flds['nDims'])
    flds['nrecords'] = int(flds['nrecords'])
    # endianness is set by _read_mds
    flds['dataprec'] = np.dtype(re.sub("'",'',flds['dataprec']))
    flds['dimList'] = [[int(h) for h in
                       re.split(',', g)] for g in
                       re.split(',\n',flds['dimList'])]
    if 'fldList' in flds:
        flds['fldList'] = [re.match("'*(\w+)",g).groups()[0] for g in
                           re.split("'\s+'",flds['fldList'])]
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
    if  expected_number_of_bytes != actual_number_of_bytes:
        raise IOError('File `%s` does not have the correct size '
                      '(expected %g, found %g)' % (datafile,
                        expected_number_of_bytes, actual_number_of_bytes))

    if use_mmap:
        d = np.memmap(datafile, dtype, 'r')
    else:
        d = np.fromfile(datafile, dtype)
    d.shape = shape
    return d
