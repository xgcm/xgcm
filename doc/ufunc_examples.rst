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

    ds = xr.Dataset(
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
    ds


.. ipython:: python

    from xgcm import Grid

    grid = Grid(
        ds,
        coords={
            "X": {"center": "x_c", "left": "x_g"},
            "Y": {"center": "y_c", "left": "y_g"},
        },
    )
    grid


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