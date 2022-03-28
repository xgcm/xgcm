xgcm: General Circulation Model Postprocessing with xarray
==========================================================

|pypi| |conda forge| |conda-forge| |Build Status| |codecov| |docs| |DOI| |license| |Code style|

Binder Examples
---------------

========= ============== ============================================================================
Link      Provider       Description
========= ============== ============================================================================
|Binder|  mybinder.org   Basic self-contained example
|PBinder| Pangeo Binder  More complex examples integrated with other Pangeo tools (dask, zarr, etc.)
========= ============== ============================================================================

Description
-----------

**xgcm** is a python package for working with the datasets produced by numerical
`General Circulation Models <https://en.wikipedia.org/wiki/General_circulation_model>`_
(GCMs) and similar gridded datasets that are amenable to
`finite volume <https://en.wikipedia.org/wiki/Finite_volume_method>`_ analysis.
In these datasets, different variables are located at different positions with
respect to a volume or area element (e.g. cell center, cell face, etc.)
xgcm solves the problem of how to interpolate and difference these variables
from one position to another.

xgcm consumes and produces xarray_ data structures, which are coordinate and
metadata-rich representations of multidimensional array data. xarray is ideal
for analyzing GCM data in many ways, providing convenient indexing and grouping,
coordinate-aware data transformations, and (via dask_) parallel,
out-of-core array computation. On top of this, xgcm adds an understanding of
the finite volume `Arakawa Grids`_ commonly used in ocean and atmospheric
models and differential and integral operators suited to these grids.

xgcm was motivated by the rapid growth in the numerical resolution of
ocean, atmosphere, and climate models. While highly parallel supercomputers can
now easily generate tera- and petascale datasets, common post-processing
workflows struggle with these volumes. Furthermore, we believe that a flexible,
evolving, open-source, python-based framework for GCM analysis will enhance
the productivity of the field as a whole, accelerating the rate of discovery in
climate science. xgcm is part of the Pangeo_ initiative.

Getting Started
---------------

To learn how to install and use xgcm for your dataset, visit the `xgcm documentation`_.


.. _Pangeo: http://pangeo.io
.. _dask: http://dask.pydata.org
.. _xarray: http://xarray.pydata.org
.. _Arakawa Grids: https://en.wikipedia.org/wiki/Arakawa_grids
.. _xgcm documentation: https://xgcm.readthedocs.io/

.. |conda forge| image:: https://img.shields.io/conda/vn/conda-forge/xgcm
   :target: https://anaconda.org/conda-forge/xgcm
.. |DOI| image:: https://zenodo.org/badge/41581350.svg
   :target: https://zenodo.org/badge/latestdoi/41581350
.. |Build Status| image:: https://img.shields.io/github/workflow/status/xgcm/xgcm/CI?logo=github
   :target: https://github.com/xgcm/xgcm/actions
   :alt: GitHub Workflow CI Status
.. |codecov| image:: https://codecov.io/github/xgcm/xgcm/coverage.svg?branch=master
   :target: https://codecov.io/github/xgcm/xgcm?branch=master
   :alt: code coverage
.. |pypi| image:: https://badge.fury.io/py/xgcm.svg
   :target: https://badge.fury.io/py/xgcm
   :alt: pypi package
.. |docs| image:: http://readthedocs.org/projects/xgcm/badge/?version=latest
   :target: http://xgcm.readthedocs.org/en/stable/?badge=latest
   :alt: documentation status
.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/xgcm/xgcm
   :alt: license
.. |Code style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/python/black
   :alt: Code style
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/xgcm/xgcm/master?filepath=doc%2Fexample_mitgcm.ipynb
.. |PBinder| image:: https://binder.pangeo.io/badge_logo.svg
   :target: https://binder.pangeo.io/v2/gh/pangeo-data/pangeo-ocean-examples/master
.. |conda-forge| image:: https://img.shields.io/conda/dn/conda-forge/xgcm?label=conda-forge
   :target: https://anaconda.org/conda-forge/xgcm
