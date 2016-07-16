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
