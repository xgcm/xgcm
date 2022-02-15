.. _grid_ufuncs:

Grid Ufuncs
-----------

Concept of a Grid Ufunc
~~~~~~~~~~~~~~~~~~~~~~~

- Generalisation of numpy generalized ufuncs to include axis positions of input and output variables
- Primer on numpy generalized ufuncs

Specifying the ``signature``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Syntax
- Dummy indices and naming conventions
- Some simple examples
- Can't supply data as kwargs

Defining New Grid Ufuncs
~~~~~~~~~~~~~~~~~~~~~~~~

Syntax Choices
^^^^^^^^^^^^^^

- Decorator with signature
- Decorator with type hints
- Applying directly

A Simple Example
^^^^^^^^^^^^^^^^

- Can we think of one that has no boundary?
- Something with an axis?
- Mean

Boundaries and Padding
~~~~~~~~~~~~~~~~~~~~~~

- ``boundary_widths``
- ``boundary``
- 1D forward differencing?
- Link to more specific docs?
- Link to more complex examples?

Metrics
~~~~~~~

- Specifying metrics
- An example

Parallelizing with Dask
~~~~~~~~~~~~~~~~~~~~~~~

Parallelizing Along Broadcast Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Under the hood we first call ``xarray.apply_ufunc``
- Primer on ``xarray.apply_ufunc``
- The ``dask`` kwarg
- Showing off the dask graph

Parallelizing Along Core Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- We also optionally call ``dask.map_blocks``
- Primer on ``dask.map_blocks``
- The ``map_blocks`` kwarg
- Rechunking that occurs when padding?
- Showing off the dask graph
