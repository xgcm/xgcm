import pytest
import os
import tarfile
import xarray as xr
import numpy as np

import xgcm

_TESTDATA_FILENAME = 'testdata.tar.gz'
_TESTDATA_ITERS = [39600,]
_TESTDATA_DELTAT = 86400

_xc_meta_content = """ simulation = { 'global_oce_latlon' };
 nDims = [   2 ];
 dimList = [
    90,    1,   90,
    40,    1,   40
 ];
 dataprec = [ 'float32' ];
 nrecords = [     1 ];
"""

@pytest.fixture(scope='module')
def mds_datadir(tmpdir_factory, request):
    # find the tar archive in the test directory
    # http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    filename = request.module.__file__
    datafile = os.path.join(os.path.dirname(filename), _TESTDATA_FILENAME)
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    # tmpdir_factory returns LocalPath objects
    # for stuff to work, has to be converted to string
    target_dir = str(tmpdir_factory.mktemp('mdsdata'))
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    return target_dir

def test_parse_meta(tmpdir):
    """Check the parsing of MITgcm .meta into python dictionary."""

    from xgcm.models.mitgcm.utils import parse_meta_file
    p = tmpdir.join("XC.meta")
    p.write(_xc_meta_content)
    fname = str(p)
    result = parse_meta_file(fname)
    expected = {
        'nrecords': 1,
        'basename': 'XC',
        'simulation': "'global_oce_latlon'",
        'dimList': [[90, 1, 90], [40, 1, 40]],
        'nDims': 2,
        'dataprec': np.dtype('float32')
    }
    for k, v in expected.items():
        assert result[k] == v

def test_open_mdsdataset_minimal(mds_datadir):
    """Create a minimal xarray object with only dimensions in it."""
    
    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
            os.path.join(mds_datadir, 'testdata'))

    # the expected dimensions of the dataset
    nx, ny, nz = 90, 40, 15
    ds_expected = xr.Dataset(coords={
        'i': np.arange(nx),
        'i_g': np.arange(nx),
        'i_z': np.arange(nx),
        'j': np.arange(ny),
        'j_g': np.arange(ny),
        'j_z': np.arange(ny),
        'k': np.arange(nz),
        'k_u': np.arange(nz),
        'k_l': np.arange(nz),
        'k_p1': np.arange(nz+1)
    })

    assert ds_expected.equals(ds)


def test_open_mdsdataset_full(mds_datadir):
    # most basic test: make sure we can open an mds dataset
    ds = xgcm.open_mdsdataset(os.path.join(mds_datadir, 'testdata'),
            _TESTDATA_ITERS, deltaT=_TESTDATA_DELTAT)
    #print(ds)

    # check just a single value
    assert ds['X'][0].values == 2.0

    # check little endianness
    ds = xgcm.open_mdsdataset(os.path.join(mds_datadir, 'testdata'),
            _TESTDATA_ITERS, deltaT=_TESTDATA_DELTAT, endian="<")
    assert ds['X'][0].values == 8.96831017167883e-44
    #print(ds)
