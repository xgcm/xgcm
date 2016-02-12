import pytest
import os
import tarfile

import xgcm

_TESTDATA_FILENAME = 'testdata.tar.gz'
_TESTDATA_ITERS = [39600,]
_TESTDATA_DELTAT = 86400


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

def test_open_mdsdataset(mds_datadir):
    # most basic test: make sure we can open an mds dataset
    ds = xgcm.open_mdsdataset(os.path.join(mds_datadir, 'testdata'),
            _TESTDATA_ITERS, deltaT=_TESTDATA_DELTAT)
