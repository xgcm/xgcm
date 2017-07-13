Grids
-----

General Concepts
~~~~~~~~~~~~~~~~

Most finite volume ocean models use `Arakawa Grids`_, in which different
variables are offset from one another and situated at different locations with
respect to the cell center and edge points.
As an example, we will consider *C-grid geometry*.
As illustrated in the figure below, C-grids place scalars (such as
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
difference operators. One goal of xgcm is to provide these interpolation
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

An important consideration for both interpolation and different operators is
boundary conditions.
xgcm currently supports periodic,
`Dirichlet <https://en.wikipedia.org/wiki/Dirichlet_boundary_condition>`_, and
`Neumann <https://en.wikipedia.org/wiki/Neumann_boundary_condition>`_ boundary
conditions, although the latter two are limited to simple cases.


Axes and Positions
~~~~~~~~~~~~~~~~~~

A fundamental concept in xgcm is the notion of an "axis". An axis is a group
of coordinates that all lie along the same physical dimension but describe
different positions relative to a grid cell. There are currently five
possible positions supported by xgcm.

    ``center``
        The variable values are located at the cell center.

    ``left``
        The variable values are located at the left (i.e. lower) face of the
        cell. The ``c_grid_axis_shift``

    ``right``
        The variable values are located at the right (i.e. upper) face of the
        cell.

    ``inner``
        The variable values are located on the cell faces, excluding both
        outer boundaries.

    ``outer``
        The variable values are located on the cell faces, including both
        outer boundaries.

The first three (``center``, ``left``, and ``right``) all have the same length
along the axis dimension, while ``inner`` has one fewer point and ``outer`` has
one extra point. These positions are visualized in the figure below.

.. figure:: images/axis_positions.svg
   :alt: axis positions

   The different possible positions of a variable ``f`` along an axis.

xgcm represents an axis using the :class:`xgcm.Axis` class.

Grid Metadata
~~~~~~~~~~~~~

xgcm works with ``xarray.DataSet`` and ``xarray.DataArray`` objects. In
order to understand the relationship between different coordinates within
these objects, xgcm looks for metadata in the form of variable attributes.
Wherever possible, we try to follow established metadata conventions, rather
than defining new metadata conventions. The two main established conventions
are the `CF Conventions`_, which apply broadly to Climate and Forecast datasets
that follow the netCDF data model, and the `COMODO conventions`_, which define
specific attributes relevant to Arakawa grids. While the COMODO conventions
were designed with C-grids in mind, we find they are general enough to support
all the different Arakawa grids.

The key attribute xgcm looks for is ``axis``.
When creating a new grid, xgcm will search through the dataset dimensions
looking for dimensions with the ``axis`` attribute defined.
All coordinates with the same value of ``axis`` are presumed to belong to the
same logical axis.
To determine the positions of the different coordinates, xgcm considers both
the length of the coordinate variable and the ``c_grid_axis_shift`` attribute,
which determines the position of the coordinate with respect to the cell center.
The only acceptable values of ``c_grid_axis_shift`` are ``-0.5`` and ``0.5``.
If the ``c_grid_axis_shift`` attribute attribute is absent, the coordinate is
assumed to describe a cell center.
The cell center coordinate is identified first; the length of other coordinates
relative to the cell center coordinate is used in conjunction with
``c_grid_axis_shift`` to infer the coordinate positions, as summarized by the
table below.

+--------+--------------------------+----------+
| length | ``c_grid_axis_shift``    | position |
+========+==========================+==========+
| n      | *None*                   | center   |
+--------+--------------------------+----------+
| n      | -0.5                     | left     |
+--------+--------------------------+----------+
| n      | 0.5                      | right    |
+--------+--------------------------+----------+
| n-1    | 0.5 or -0.5              | inner    |
+--------+--------------------------+----------+
| n+1    | 0.5 or -0.5              | outer    |
+--------+--------------------------+----------+

If your dataset does not conform to CF and COMODO conventions already, you
must set these attributes manually before passing it to xgcm.

.. note::

  xgcm never requires datasets to have specific variable names. Rather,
  the grid geometry is inferred through the attributes.


``Grid`` Objects
~~~~~~~~~~~~~~~~

The core object in xgcm is a :class:`xgcm.Grid`. To create a grid, first we need
an ``xarray.DataSet`` with proper attributes. We can create one as follows.

.. code-block:: python

    >>> import xarray as xr
    >>> import numpy as np
    >>> ds = xr.Dataset(
               coords={'x_c': (['x_c',], np.arange(1,10), {'axis': 'X'}),
                       'x_g': (['x_g',], np.arange(0.5,9),
                               {'axis': 'X', 'c_grid_axis_shift': -0.5})})
    >>> ds
    <xarray.Dataset>
    Dimensions:  (x_c: 9, x_g: 9)
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
    X Axis (periodic):
      * center   x_c (9) --> left
      * left     x_g (9) --> center

We see that xgcm successfully parsed the metadata and inferred the relative
location of the different coordinates along the x axis.
Because we did not
specify the ``periodic`` keyword argument, xgcm assumed that the data
is periodic along all dimensions.
The arrows after each coordinate indicate the default shift positions for
interpolation and difference operations: operating on the center coordinate
(``x_c``) shifts to the left coordinate (``x_g``), and vice versa.
Now we can use this grid to interpolate or
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
    <xarray.DataArray (x_g: 9)>
    array([  3.213938e-01,   8.137977e-01,   9.254166e-01,   6.040228e-01,
             1.110223e-16,  -6.040228e-01,  -9.254166e-01,  -8.137977e-01,
            -3.213938e-01])
    Coordinates:
      * x_g      (x_g) float64 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5

We see that the output is on the ``x_g`` points rather than the original ``xc``
points. The same transformation happens with a difference:

.. code-block:: python

    >>> grid.diff(f, axis='X')
    <xarray.DataArray (x_g: 9)>
    array([ 0.642788,  0.34202 , -0.118782, -0.524005, -0.68404 , -0.524005,
           -0.118782,  0.34202 ,  0.642788])
    Coordinates:
      * x_g      (x_g) float64 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5

.. warning::

    xgcm does not perform input validation to verify that ``f`` is
    compatible with ``grid``.

For more advanced usage of xgcm, see the examples.

.. _Arakawa Grids: https://en.wikipedia.org/wiki/Arakawa_grids
.. _xarray: http://xarray.pydata.org
.. _MITgcm notation: http://mitgcm.org/public/r2_manual/latest/online_documents/node31.html
.. _CF Conventions: http://cfconventions.org/
.. _COMODO Conventions: http://pycomodo.forge.imag.fr/norm.html
