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

def _untar(data_dir, basename, target_dir):
    """Unzip a tar file into the target directory. Return path to unzipped
    directory."""
    datafile = os.path.join(data_dir, basename + '.tar.gz')
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    # subdirectory where file should have been untarred.
    # assumes the directory is the same name as the tar file itself.
    # e.g. testdata.tar.gz --> testdata/
    fulldir = os.path.join(target_dir, basename)
    if not os.path.exists(fulldir):
        raise IOError('Could not find tar file output dir %s' % basedir)
    # the actual data lives in a file called testdata
    return fulldir

# parameterized fixture are complicated
# http://docs.pytest.org/en/latest/fixture.html#fixture-parametrize

# dictionary of archived experiments and some expected properties
_experiments = {
    'global_oce_latlon': {'shape': (15, 40, 90), 'test_iternum': 39600},
    'barotropic_gyre': {'shape': (1,60,60), 'test_iternum': 10},
    'internal_wave': {'shape': (20,1,30), 'test_iternum': 100,
                      'multiple_iters': [0,100,200]}
}

# find the tar archive in the test directory
# http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
@pytest.fixture(scope='module', params=_experiments.keys())
def all_mds_datadirs(tmpdir_factory, request):
    """The datasets."""
    expt_name = request.param
    expected_results = _experiments[expt_name]
    target_dir = str(tmpdir_factory.mktemp('mdsdata'))
    data_dir = os.path.dirname(request.module.__file__)
    return _untar(data_dir, expt_name, target_dir), expected_results

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

def test_read_mds(all_mds_datadirs):
    """Check that we can read mds data from .meta / .data pairs"""

    dirname, expected = all_mds_datadirs

    from xgcm.models.mitgcm.utils import read_mds

    prefix = 'XC'
    basename = os.path.join(dirname, prefix)
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
    prefix = 'T'
    basename = os.path.join(dirname, prefix)
    iternum = expected['test_iternum']
    res = read_mds(basename, iternum=iternum)
    assert prefix in res

# @pytest.mark.parametrize("datadir,expected_shape", [
#     (all_mds_datadirs, (90, 40, 15)),
# ])

def test_open_mdsdataset_minimal(all_mds_datadirs):
    """Create a minimal xarray object with only dimensions in it."""

    dirname, expected = all_mds_datadirs

    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
            dirname, read_grid=False)

    # the expected dimensions of the dataset
    nz, ny, nx = expected['shape']
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

def test_read_grid(all_mds_datadirs):
    """Make sure we read all the grid variables."""
    dirname, expected = all_mds_datadirs
    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
                dirname, read_grid=True)

    for vname in _EXPECTED_GRID_VARS:
        assert vname in ds

def test_swap_dims(all_mds_datadirs):
    """Make sure we read all the grid variables."""

    dirname, expected = all_mds_datadirs
    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
                dirname, read_grid=True, swap_dims=True)

    expected_dims = ['XC', 'XG', 'YC', 'YG', 'Z', 'Zl', 'Zp1', 'Zu']
    assert ds.dims.keys() == expected_dims

def test_prefixes(all_mds_datadirs):
    """Make sure we read all the grid variables."""

    dirname, expected = all_mds_datadirs
    prefixes = ['U', 'V', 'W', 'T', 'S', 'PH'] #, 'PHL', 'Eta']
    iters = [expected['test_iternum']]
    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
                dirname, iters=iters, prefix=prefixes,
                read_grid=False)

    for p in prefixes:
        assert p in ds

    # try with dim swapping
    ds = xgcm.models.mitgcm.mds_store.open_mdsdataset(
                dirname, iters=iters, prefix=prefixes,
                read_grid=True, swap_dims=True)

    for p in prefixes:
        assert p in ds


# @pytest.mark.skipif(True, reason="Not ready")
# def test_open_mdsdataset_full(all_mds_datadirs):
#     # most basic test: make sure we can open an mds dataset
#     ds = xgcm.open_mdsdataset(all_mds_datadirs,
#             _TESTDATA_ITERS, deltaT=_TESTDATA_DELTAT)
#     #print(ds)
#
#     # check just a single value
#     assert ds['X'][0].values == 2.0
#
#     # check little endianness
#     ds = xgcm.open_mdsdataset(all_mds_datadirs,
#             _TESTDATA_ITERS, deltaT=_TESTDATA_DELTAT, endian="<")
#     assert ds['X'][0].values == 8.96831017167883e-44
#     #print(ds)
