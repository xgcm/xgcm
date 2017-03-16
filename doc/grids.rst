Grids
-----

General Concepts
~~~~~~~~~~~~~~~~

Most finite volume ocean models use `Arakawa Grids`_, in which different
variables are offset from one another and situated at different locations with
respect to the cell center and edge points. `xgcm` currently supports only
*C-grid geometry*. As illustrated below, C-grids place scalars (such as
temperature) at the cell center and vector components (such as velocity) at
the cell faces. This type of grid is widely used because of its favorable
conservation properties.

.. figure:: images/grid2d_hv.svg
  :scale: 100
  :alt: C-grid Geometry

  Layout of variables with respect to cell centers and edges in a C-grid
  ocean model. Image from the
  `pycomodo project <http://pycomodo.forge.imag.fr/norm.html>`_.

These grids present a dilemma for the `xarray`_ data model. The ``u`` and ``t``
points in the example above are located at different points along the x-axis,
meaning they can't be represented using a single coordinate. But they are
clearly related and can be transformed via well defined interpolation and
difference operators. One goal of ``xgcm`` is to provide these interpolation
and difference operators.

We use `MITgcm notation`_ to denote the basic operators that transform between
grid points. The difference operator is defined as

.. math::

   \delta_i \Phi = \Phi_{i+1/2} - \Phi_{i-1/2}

where :math:`\Phi` is any variable and ``i`` represents the grid index.
The other basic operator is interpolation,
defined as

.. math::

   \overline{\Phi} = (\Phi_{i+1/2} + \Phi_{i-1/2})/2

Both operators return a variable that is shifted by half a gridpoint
with respect to the input variable.
With these two operators, the entire suite of finite volume vector calculus
operations can be represented.

Grid Metadata
~~~~~~~~~~~~~

``xgcm`` works with ``xarray.DataSet`` and ``xarray.DataArray`` objects. In
order to understand the relationship between different coordinates within
these objects, ``xgcm`` looks for metadata in the form of variable attributes.
Wherever possible, we try to follow established metadata conventions, rather
than defining new metadata conventions. The two main established conventions
are the `CF Conventions`_, which apply broadly to Climate and Forecast datasets
that follow the netCDF data model, and the `COMODO conventions`_, which define
specific attributes relevant to C-grids.

.. note::

  ``xgcm`` never requires datasets to have specific variable names. Rather,
  the grid geometry is inferred through the attributes, allowing users to use
  the variable names they prefer.

When creating a grid, ``xgcm`` looks for the ``axis`` in all the dimensions.
Currently the only allowable values are ``X`` and ``Y``, corresponding to the
two horizontal dimensions. ``xgcm`` will search through the dataset dimensions
and look for dimensions with the ``axis`` parameter in order to determine the
the relevant coordinates. The next attribute is ``c_grid_axis_shift``, which
determines the position of the coordinate with respect to the cell center. If
this attribute is absent, the coordinate is assumed to describe a cell center.
The only acceptable values of ``c_grid_axis_shift`` are ``-0.5`` and ``0.5``.

.. warning::

  ``xgcm`` can currently only handle two different coordinates per axis: a cell
  center and a cell edge. Datasets that have more than two dimensions with the
  same values of ``axis`` will raise errors.

The cell-center and cell-edge coordinates should either have the same length, or
else the cell-edge coordinate should have one more point than the cell-center.

``Grid`` Objects
~~~~~~~~~~~~~~~~

The core object in xgcm is a :class:`xgcm.Grid`. To create a grid, first we need an
``xarray.DataSet`` with proper attributes. We can create one as follows.

.. code-block:: python

    >>> import xarray as xr
    >>> import numpy as np
    >>> ds = xr.Dataset(
               coords={'x_c': (['x_c',], np.arange(1,10), {'axis': 'X'}),
                       'x_g': (['x_g',], np.arange(0.5,9),
                               {'axis': 'X', 'c_grid_axis_shift': -0.5})})
    >>> ds
    <xarray.Dataset>
    Dimensions:  (x_c: 9, x_g: 10)
    Coordinates:
      * x_g      (x_g) float64 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5
      * x_c      (x_c) int64 1 2 3 4 5 6 7 8 9
    Data variables:
        *empty*

(Note that this dataset has no data variables yet, just coordinates.)
We now create a ``Grid`` object from this dataset:

.. code-block:: python

    >>> from xgcm import Grid
    >>> grid = Grid(ds)
    >>> grid
    <xgcm.Grid>
    X-axis:     x_c: 9 (cell center), x_g: 9 (cell face, shift -1) periodic

We see that ``xgcm`` successfully parsed the metadata and inferred the relative
location of the different coordinates along the x axis. Because we did not
specify the ``x_periodic`` keyword argument, ``xgcm`` assumed that the data
is periodic along the X axis. Now we can use this grid to interpolate or
take differences along the axis. First we create some test data:

.. code-block:: python

    >>> f = np.sin(ds.x_c * 2*np.pi/9)
    >>> f
    <xarray.DataArray 'x_c' (x_c: 9)>
    array([  6.427876e-01,   9.848078e-01,   8.660254e-01,   3.420201e-01,
            -3.420201e-01,  -8.660254e-01,  -9.848078e-01,  -6.427876e-01,
            -2.449294e-16])
    Coordinates:
      * x_c      (x_c) int64 1 2 3 4 5 6 7 8 9

We interpolate as follows:

.. code-block:: python

    >>> grid.interp(f, axis='X')
    <xarray.DataArray 'x_c' (x_g: 9)>
    array([  3.213938e-01,   8.137977e-01,   9.254166e-01,   6.040228e-01,
             1.110223e-16,  -6.040228e-01,  -9.254166e-01,  -8.137977e-01,
            -3.213938e-01])
    Coordinates:
      * x_g      (x_g) float64 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5

We see that the output is on the ``x_g`` points rather than the original ``xc``
points. The same transformation happens with a diffrence

.. code-block:: python

    >>> grid.diff(f, axis='X')
    <xarray.DataArray 'x_c' (x_g: 9)>
    array([ 0.642788,  0.34202 , -0.118782, -0.524005, -0.68404 , -0.524005,
           -0.118782,  0.34202 ,  0.642788])
    Coordinates:
      * x_g      (x_g) float64 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5

.. warning::

    ``xgcm`` does not perform input validation to verify that ``f`` is
    compatible with ``grid``.

.. _Arakawa Grids: https://en.wikipedia.org/wiki/Arakawa_grids
.. _xarray: http://xarray.pydata.org
.. _MITgcm notation: http://mitgcm.org/public/r2_manual/latest/online_documents/node31.html
.. _CF Conventions: http://cfconventions.org/
.. _COMODO Conventions: http://pycomodo.forge.imag.fr/norm.html
