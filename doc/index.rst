.. xgcm documentation master file, created by
   sphinx-quickstart on Sat Aug 29 00:18:20 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xgcm: General Circulation Model Postprocessing with xarray
==========================================================

**xgcm** aims to become a general purpose tool for processing GCM output in
python. xgcm is built on top of xarray_, which provides most of the core data
model, indexing, and (via dask_) parallel, out-of-core array computation. On
top of this, xgcm adds an understanding of the finite volume `Arakawa Grids`_
commonly used in ocean and atmospheric models, differential and integral
operators suited to these grids, and a toolbox of common analysis functions.

**xgcm** was motivated by the rapid growth in the numerical resolution of
ocean, atmosphere, and climate models. While highly parallel supercomputers can
now easily generate tera- and petascale datasets, common post-processing
workflows struggle with these volumes. Furthermore, we believe that a flexible,
evoliving, open-source, python-based framework for GCM analysis will enhance
the productivity of the field as a whole, accelerating the rate of discovery in
climate science.

.. _dask: http://dask.pydata.org
.. _xarray: http://xarray.pydata.org
.. _Arakawa Grids: https://en.wikipedia.org/wiki/Arakawa_grids

Contents
--------

.. toctree::
   :maxdepth: 1

   installing
   examples
   supported-models
   grids
   fluxes
   budgets
   coordinate-transformation
   api
   faq
   contributing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
