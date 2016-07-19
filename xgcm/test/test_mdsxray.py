import pytest
import os
import tarfile
import xarray as xr
import numpy as np

import xgcm

_TESTDATA_FILENAME = 'testdata.tar.gz'
_TESTDATA_ITERS = [39600,]
_TESTDATA_DELTAT = 86400

_EXPECTED_GRID_VARS = ['XC', 'YC', 'XG', 'YG', 'Zl', 'Zu', 'Z', 'Zp1', 'dxC',
                       'rAs', 'rAw', 'Depth', 'rA', 'dxG', 'dyG', 'rAz', 'dyC',
                     'PHrefC', 'drC', 'PHrefF', 'drF', 'hFacS', 'hFacC','hFacW']


_xc_meta_content = """ simulation = { 'global_oce_latlon' };
 nDims = [   2 ];
 dimList = [
    90,    1,   90,
    40,    1,   40
 ];
 dataprec = [ 'float32' ];
 nrecords = [     1 ];
"""

def _untar(datafile, target_dir):
    """Unzip a tar file into the target directory. Return path to unzipped
    directory."""
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    # subdirectory where file should have been untarred.
    # assumes the directory is the same name as the tar file itself.
    # e.g. testdata.tar.gz --> testdata/
    basedir = os.path.basename(datafile).split('.tar.gz')[0]
    fulldir = os.path.join(target_dir, basedir)
    if not os.path.exists(fulldir):
        raise IOError('Could not find tar file output dir %s' % basedir)
    # the actual data lives in a file called testdata
    return fulldir

# find the tar archive in the test directory
# http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
@pytest.fixture(scope='module')
def mds_datadir(tmpdir_factory, request):
    """The standard, 3D, spherical polar dataset."""
    target_dir = str(tmpdir_factory.mktemp('mdsdata'))
    filename = request.module.__file__
    datafile = os.path.join(os.path.dirname(filename), 'testdata.tar.gz')
    return _untar(datafile, target_dir)

@pytest.fixture(scope='module')
def barotropic_gyre_datadir(tmpdir_factory, request):
    """2D (x,y) dataset."""
    target_dir = str(tmpdir_factory.mktemp('mdsdata'))
    filename = request.module.__file__
    datafile = os.path.join(os.path.dirname(filename), 'barotropic_gyre.tar.gz')
    return _untar(datafile, target_dir)

@pytest.fixture(scope='module')
def barotropic_gyre_datadir(tmpdir_factory, request):
    """2D (x,z) dataset."""
    target_dir = str(tmpdir_factory.mktemp('mdsdata'))
    filename = request.module.__file__
    datafile = os.path.join(os.path.dirname(filename), 'internal_wave.tar.gz')
    return _untar(datafile, target_dir)

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

def test_read_raw_data(tmpdir):
    """Check our utility for reading raw data."""

    from xgcm.models.mitgcm.utils import read_raw_data
    shape = (2,4)
    for dtype in [np.dtype('f8'), np.dtype('f4'), np.dtype('i4')]:
        # create some test data
        testdata = np.zeros(shape, dtype)
        # write to a file
        datafile = tmpdir.join("tmp.data")
        datafile.write_binary(testdata.tobytes())
        fname = str(datafile)
        # now test the function
        data = read_raw_data(fname, dtype, shape)
        np.testing.assert_allclose(data, testdata)
        assert isinstance(data, np.ndarray)
        # check memmap
        mdata = read_raw_data(fname, dtype, shape, use_mmap=True)
        assert isinstance(mdata, np.memmap)

    # make sure errors are correct
    wrongshape = (2,5)
    with pytest.raises(IOError):
        _ = read_raw_data(fname, dtype, wrongshape)

def test_read_mds(mds_datadir):
    """Check that we can read mds data from .meta / .data pairs"""

    from xgcm.models.mitgcm.utils import read_mds

    prefix = 'XC'
    basename = os.path.join(mds_datadir, prefix)
    res = read_mds(basename)
    assert isinstance(res, dict)
    assert prefix in res
    # should be memmap by default
    assert isinstance(res[prefix], np.memmap)

    # try some options
    res = read_mds(basename, force_dict=False)
    assert isinstance(res, np.memmap)
    res = read_mds(basename, force_dict=False, use_mmap=False)
    assert isinstance(res, np.ndarray)

    # try reading with iteration number
    prefix = 'Ttave'
    basename = os.path.join(mds_datadir, prefix)
    iternum = 39600
    res = read_mds(basename, iternum=iternum)
    assert 'Ttave' in res

def test_open_mdsdataset_minimal(mds_datadir):
    """Create a minimal xarray object with only dimensions in it."""

    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
                mds_datadir,
                read_grid=False)

    # the expected dimensions of the dataset
    nx, ny, nz = 90, 40, 15
    ds_expected = xr.Dataset(coords={
        'i': np.arange(nx),
        'i_g': np.arange(nx),
        #'i_z': np.arange(nx),
        'j': np.arange(ny),
        'j_g': np.arange(ny),
        #'j_z': np.arange(ny),
        'k': np.arange(nz),
        'k_u': np.arange(nz),
        'k_l': np.arange(nz),
        'k_p1': np.arange(nz+1)
    })

    assert ds_expected.equals(ds)

def test_open_mdsdataset_read_grid(mds_datadir):
    """Make sure we read all the grid variables."""

    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
                mds_datadir, read_grid=True)

    for vname in _EXPECTED_GRID_VARS:
        assert vname in ds

def test_open_mdsdataset_swap_dims(mds_datadir):
    """Make sure we read all the grid variables."""

    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
                mds_datadir, read_grid=True, swap_dims=True)

    expected_dims = ['XC', 'XG', 'YC', 'YG', 'Z', 'Zl', 'Zp1', 'Zu']
    assert ds.dims.keys() == expected_dims

def test_open_mdsdataset_with_prefixes(mds_datadir):
    """Make sure we read all the grid variables."""

    prefixes = ['U', 'V', 'W', 'T', 'S', 'PH', 'PHL', 'Eta']
    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
                mds_datadir, iters=_TESTDATA_ITERS, prefix=prefixes,
                read_grid=False)

    for p in prefixes:
        assert p in ds
    print ds

    # try with dim swapping
    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
                mds_datadir, iters=_TESTDATA_ITERS, prefix=prefixes,
                read_grid=True, swap_dims=True)

    for p in prefixes:
        assert p in ds


@pytest.mark.skipif(True, reason="Not ready")
def test_open_mdsdataset_full(mds_datadir):
    # most basic test: make sure we can open an mds dataset
    ds = xgcm.open_mdsdataset(mds_datadir,
            _TESTDATA_ITERS, deltaT=_TESTDATA_DELTAT)
    #print(ds)

    # check just a single value
    assert ds['X'][0].values == 2.0

    # check little endianness
    ds = xgcm.open_mdsdataset(mds_datadir,
            _TESTDATA_ITERS, deltaT=_TESTDATA_DELTAT, endian="<")
    assert ds['X'][0].values == 8.96831017167883e-44
    #print(ds)
