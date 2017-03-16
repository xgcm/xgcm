
Installation
------------

Requirements
^^^^^^^^^^^^

xgcm is compatible with python 3 and python 2.7. It requires xarray_
(>= version 0.8.2) and dask_ (>= version 0.11.2).
These packages are most reliably installed via the
`conda <http://conda.pydata.org/docs/>`_ environment management
system, which is part of the Anaconda_ python distribution. Assuming you have
conda available on your system, the dependencies can be installed with the
command::

    conda install xarray dask

If you are using earlier versions of these packages, you should update before
installing xgcm.

Installation via pip
^^^^^^^^^^^^^^^^^^^^

If you just want to use xgcm, the easiest way is to install via pip::

    pip install xgcm

This will automatically install the latest release from
`pypi <https://pypi.python.org/pypi>`_.

Installation from github
^^^^^^^^^^^^^^^^^^^^^^^^

xgcm is under active development. To obtain the latest development version,
you may clone the `source repository <https://github.com/xgcm/xgcm>`_
and install it:

    git clone https://github.com/xgcm/xgcm.git
    cd xgcm
    python setup.py install

Users are encouraged to `fork <https://help.github.com/articles/fork-a-repo/>`_
xgcm and submit issues_ _ and `pull requests`_.

.. _dask: http://dask.pydata.org
.. _xarray: http://xarray.pydata.org
.. _Comodo: http://pycomodo.forge.imag.fr/norm.html
.. _issues: https://github.com/xgcm/xgcm/issues
.. _`pull requests`: https://github.com/xgcm/xgcm/pulls
.. _MITgcm: http://mitgcm.org/public/r2_manual/latest/online_documents/node277.html
.. _out-of-core: https://en.wikipedia.org/wiki/Out-of-core_algorithm
.. _Anaconda: https://www.continuum.io/downloads
.. _`CF conventions`: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch04s04.html
.. _gcmfaces: http://mitgcm.org/viewvc/*checkout*/MITgcm/MITgcm_contrib/gael/matlab_class/gcmfaces.pdf
