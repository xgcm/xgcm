.. _grid_ufuncs:

Grid Ufuncs
-----------

Concept of a Grid Ufunc
~~~~~~~~~~~~~~~~~~~~~~~

In short, a "grid ufunc" is a generalisation of a `numpy generalized universal function`_ to include the xGCM Axes and Positions of input and output variables.
We tell a function about the axes information through a ``signature``, which for a function which accepts data located at the center grid positions and returns data located on the same axis but now at the left-hand grid positions would look like:

``"(ax1:center)->(ax1:left)"``.

If you are not already familiar with numpy generalised universal functions (hereon referred to as "numpy ufuncs"), then here is a quick primer.

.. dropdown:: **Primer on numpy generalized universal functions**

    Content about numpy...

Grid ufuncs allow us to:

- Avoid mistakes by stating that functions are only valid for data on specific grid positions,
- Neatly promote numpy functions to grid-aware xarray functions,
- Conveniently apply boundary conditions and grid topologies (see :ref:`Boundaries and Padding`),
- Immediately parallelize our operations with dask (see :ref:`Parallelizing with Dask`).

.. _numpy generalized universal function: https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html

Meaning of the ``signature``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "signature" of a grid ufunc is how we tell it which input and output variables should be on which axis positions.
A simple example would be
``"(ax1:center)->(ax1:left)"``

The signature has two parts, one for the input variables, and one for the output variables.
The output variables live on the right of the arrow (``->``).

There needs to be one bracketed entry for each variable, so in this case the signature tells use that the function accepts one input data variable and returns one output data variable.
(Functions which accept a data variable in the form of a keyword-only argument are not supported.)

For each variable, the signature tells us the ``xgcm.Axis`` positions we require that variable to have, both before and after our grid ufunc is applied.
This information is encoded in the format ``axis_name:position``.
Each variable can be operated on along multiple axes, which are separated by a comma, e.g. ``(ax1:left, ax2:right)``.

The axis names used in the signature are dummy names: they do not have to be the same as the axis names used in your ``Grid`` object.
This allows you to write a grid ufunc that can accept axis with any name.
Therefore the signature ``"(ax1:center)->(ax1:left)"`` means all of

`"This function accepts one data variable, which is one-dimensional and lies on the center grid positions of its singular axis.
After performing its numerical operation the single return value from this function will have been shifted onto the left-hand grid positions of the same axis."`

- Some simple examples, in a table

Defining New Grid Ufuncs
~~~~~~~~~~~~~~~~~~~~~~~~

Lets imagine we have a numpy function which does forward differencing along one dimension, with an implicit periodic boundary condition.
::

    def diff_forward(a):
        return a - np.roll(a, -1, axis=-1)

All this function does is subtract each element of the given array from the element immediately to its right, with the ends of the array wrapped around in a periodic fashion.
If we imagine this function acting on a variable located at the cell centers, our axis position diagram suggests that the result would lie on the left-hand cell edges.
Therefore the signature of this function could be
``"(ax1:center)->(ax1:left)"``.

.. note::

    XGCM assumes the function acts along the last axis of the numpy array, which is why we have specified ``axis=-1`` here.

There are multiple options for how to use this numpy ufunc as a grid ufunc.

Applying directly
^^^^^^^^^^^^^^^^^

The quickest option is to apply our function directly, using ``apply_as_grid_ufunc``
::

    from xgcm import apply_as_grid_ufunc

    apply_as_grid_ufunc(grid, data, diff_forward, signature="(ax1:center)->(ax1:left)", axis=[["X"]])

Here we have applied the grid ufunc to the data, along the axis ``"X"`` of the grid.
The dummy axis name ``ax1`` gets substituted by ``"X"`` during the call, so this will fail if our data does not depend on the axis we attempt to apply the ufunc along.

Decorator with signature
^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively you can permanently turn a numpy function into a grid ufunc by using the ``@as_grid_ufunc`` decorator.
::

    from xgcm import as_grid_ufunc

    @as_grid_ufunc(signature="(ax1:center)->(ax1:left)")
    def diff_forward(a):
        return a - np.roll(a, -1, axis=-1)

Now when we call the ``diff_forward`` function, it will act as if we had applied it using ``apply_as_grid_ufunc``.

Decorator with type hints
^^^^^^^^^^^^^^^^^^^^^^^^^

Finally you can use type hints to specify the grid positions of the variables instead of passing a ``signature`` argument.
::

    from xgcm import Gridded

    @as_grid_ufunc()
    def diff_forward(a: Gridded[np.ndarray, ax1:center]) -> Gridded[np.ndarray, ax1:left]:
        return a - np.roll(a, -1, axis=-1)

.. note::

    ``Gridded`` here is really just an alias for ``typing.Annotated``.

The signature argument is incompatible with using ``Gridded`` to annotate the types of any of the function arguments - i.e. you cannot mix the signature approach with the type hinting approach.

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
