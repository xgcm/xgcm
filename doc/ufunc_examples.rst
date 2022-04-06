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
                np.arange(1, 20),
                {"axis": "X"},
            ),
            "x_g": (
                [
                    "x_g",
                ],
                np.arange(0.5, 19),
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "y_c": (
                [
                    "y_c",
                ],
                np.arange(1, 20),
                {"axis": "Y"},
            ),
            "y_g": (
                [
                    "y_g",
                ],
                np.arange(0.5, 19),
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

    U = np.sin(grid_ds.x_c * 2 * np.pi / 19) * np.cos(grid_ds.y_c * 2 * np.pi / 19)
    V = np.cos(grid_ds.x_c * 2 * np.pi / 19) * np.sin(grid_ds.y_c * 2 * np.pi / 19)

    ds = xr.Dataset({"V": V, "U": U})
    ds


.. ipython:: python

    @savefig example_vector_field.png width=4in
    ds.plot.quiver("x_c", "y_c", u="U", v="V")


Divergence
~~~~~~~~~~

Let's first import the decorator.

.. ipython:: python

    from xgcm import as_grid_ufunc


In two dimensions, the divergence operator accepts two vector components and returns one scalar result.
Each vector component will be differentiated along one axis, and doing so with a first order forward difference would
shift the data's position along that axis.
Therefore our signature should look something like this ``"(X:center,Y:center),(X:center,Y:center)->(X:center,Y:center)"``.

A divergence is the sum of multiple partial derivatives, so first let's define a derivative function like this

.. ipython:: python

    def diff_1d(a):
        return 0.5 * (a[..., 2:] - a[..., :-2])

    def diff_center_to_center_second_order(arr, axis):
        return np.apply_along_axis(diff_1d, axis, arr)

Now if we treat the components of the ``(U, V)`` vector as independent scalars, our grid ufunc could be defined like this

.. ipython:: python

    @as_grid_ufunc("(X:center,Y:center),(X:center,Y:center)->(X:center,Y:center)", boundary_width={'X': (2, 0), 'Y': (2, 0)})
    def divergence(u, v):
        u_diff_x = diff_center_to_center_second_order(u, axis=-2)
        v_diff_y = diff_center_to_center_second_order(v, axis=-1)
        # Need to trim off elements so that the two arrays have same shape
        div = u_diff_x[..., :-2] + v_diff_y[..., :-2, :]
        return div

Now we can compute the divergence of our example vector field

.. ipython:: python

    div = divergence(grid, ds['U'], ds['V'], axis=[('X', 'Y'), ('X', 'Y')])

    @savefig div_vector_field.png width=4in
    div.plot(x='x_c')



Gradient
~~~~~~~~

The gradient is almost like the opposite of divergence in the sense that it accepts one scalar and returns multiple vectors

``"(X:center,Y:center)->(X:inner,Y:center),(X:center,Y:inner)"``


Curl/Vorticity
~~~~~~~~~~~~~~



Advection
~~~~~~~~~
