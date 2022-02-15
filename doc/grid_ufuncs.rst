.. _grid_ufuncs:

Grid Ufuncs
-----------

Concept of a Grid Ufunc
~~~~~~~~~~~~~~~~~~~~~~~

In short, a "grid ufunc" is a generalisation of a `numpy generalized universal function`_ to include the xGCM Axes and Positions of input and output variables.
We tell a function about the axes information through a ``signature``, which for a function which accepts data located at the center grid positions and returns data located on the same axis but now at the left-hand grid positions, would look like
``"(Ax1:center)->(Ax1:left)"``.

If you are not already familiar with numpy generalised universal functions (hereon referred to as "numpy ufuncs"), then here is a quick primer.

- Primer on numpy generalized ufuncs (dropdown)

Grid ufuncs allow us to:

- Avoid mistakes by stating that functions are only valid for data on specific grid positions,
- Neatly promote numpy functions to grid-aware xarray functions,
- Conveniently apply boundary conditions and grid topologies (see :ref:`Boundaries and Padding`),
- Immediately parallelize our operations with dask (see :ref:`Parallelizing with Dask`).

.. _numpy generalized universal function: https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html

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

.. _Boundaries and Padding:

Boundaries and Padding
~~~~~~~~~~~~~~~~~~~~~~

- ``boundary_width``
- Relationship to padding
- ``boundary``
- 1D forward differencing?
- Link to more specific docs?
- Link to more complex examples?

Metrics
~~~~~~~

- Specifying metrics
- An example

.. _Parallelizing with Dask:

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
- Primer on ``dask.map_overlap``
- The ``map_overlap`` kwarg
- Restriction that you can't do this with grid ufuncs that change length (e.g. center to outer)
- Rechunking that occurs when padding?
- Showing off the dask graph
