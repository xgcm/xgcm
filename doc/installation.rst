
Installation
------------

Requirements
^^^^^^^^^^^^

xgcm is compatible with python 3 (>= version 3.9). It requires xarray_
(>= version 0.20.0) numpy_ and dask_.

Installation from Conda Forge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install xgcm along with its dependencies is via conda
forge::

    conda install -c conda-forge xgcm


Installation from Pip
^^^^^^^^^^^^^^^^^^^^^

An alternative is to use pip::

    pip install xgcm

This will install the latest release from
`pypi <https://pypi.python.org/pypi>`_.

Installation from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^

xgcm is under active development. To obtain the latest development version,
you may clone the `source repository <https://github.com/xgcm/xgcm>`_
and install it::

    git clone https://github.com/xgcm/xgcm.git
    cd xgcm
    python setup.py install

or simply::

    pip install git+https://github.com/xgcm/xgcm.git

Users are encouraged to `fork <https://help.github.com/articles/fork-a-repo/>`_
xgcm and submit issues_ and `pull requests`_.

.. _dask: http://dask.pydata.org
.. _numpy: https://numpy.org
.. _xarray: http://xarray.pydata.org
.. _issues: https://github.com/xgcm/xgcm/issues
.. _`pull requests`: https://github.com/xgcm/xgcm/pulls
