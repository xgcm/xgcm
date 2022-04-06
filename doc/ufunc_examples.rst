.. _ufunc_examples:

Ufunc Examples
--------------

This page contains examples of different Grid Ufuncs that you might find useful.
They are intended to be more advanced or realistic cases than the simple differencing operations we showed in the introduction to :ref:`grid_ufuncs`.

As each of these cases is a vector calculus operator, we will demonstrate their use on an example two-dimensional vector field.

Firstly we need a two-dimensional grid, and we use similar coordinate names to the 1D example we used when first introducing :ref:`grids`.

.. ipython:: python

    import numpy as np
    import xarray as xr

    grid_ds = xr.Dataset(
        coords={
            "x_c": (
                [
                    "x_c",
                ],
                np.arange(1, 10),
                {"axis": "X"},
            ),
            "x_g": (
                [
                    "x_g",
                ],
                np.arange(0.5, 9),
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "y_c": (
                [
                    "y_c",
                ],
                np.arange(1, 10),
                {"axis": "Y"},
            ),
            "y_g": (
                [
                    "y_g",
                ],
                np.arange(0.5, 9),
                {"axis": "Y", "c_grid_axis_shift": -0.5},
            ),
        }
    )
    grid_ds


.. ipython:: python

    from xgcm import Grid

    grid = Grid(
        grid_ds,
        coords={
            "X": {"center": "x_c", "left": "x_g"},
            "Y": {"center": "y_c", "left": "y_g"},
        },
    )
    grid


Now we need some data.
We will create a 2D vector field, with components ``U`` and ``V``.

.. ipython:: python

    U = np.sin(grid_ds.y_c * 2 * np.pi / 9).expand_dims(x_c=9)
    V = np.sin(grid_ds.x_c * 2 * np.pi / 9).expand_dims(y_c=9)

    ds = xr.Dataset({"V": V, "U": U})
    ds


.. ipython:: python

    @savefig example_vector_field.png width=4in
    ds.plot.quiver("x_c", "y_c", u="U", v="V")




Divergence
~~~~~~~~~~

In two dimensions, the divergence operator accepts two vector components and returns one scalar result




Gradient
~~~~~~~~

The gradient is almost like the opposite of divergence in the sense that it accepts one scalar and returns multiple vectors



Curl/Vorticity
~~~~~~~~~~~~~~



Advection
~~~~~~~~~
