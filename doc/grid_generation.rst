Grid Generation
---------------
Ocean model output is usually provided with geometric information like the
position of the cell center and boundaries. Gridded observational datasets
often lack these informations and provide only the position of either
the gridcell center or cell boundary.
This makes the calculation of common vector calculus operators like the
gradient and divergence difficult, and results depend on the particular method
used.

xgcm can infer the cell boundary or cell center location depending on the
geometry of the gridded source dataset. This enables consistent and easily
reproducible calculations across model and observational datasets.

A common grid structure found in observational data sets is an array of tracer
concentrations given at the center of a grid box

.. code-block:: python

        >>> from xgcm import generate_grid_ds, Grid
        >>> import xarray as xr
        >>> import numpy as np

        >>> dx = 0.5
        >>> dy = 1.0
        >>> dz = 0.5
        >>> a = np.random.rand(180, int(360/dx), int(10/dz))

        >>> x = np.arange(-180, 180, dx)
        >>> y = np.arange(-90, 90, dy)
        >>> z = np.arange(0, 10, dz)

        >>> xx, yy = np.meshgrid(x, y)
        >>> _, _, zz = np.meshgrid(x, y, z)

        >>> ds_original = xr.Dataset(
                                     {'somedata': (['lat', 'lon', 'z'], a)},
                                     coords={'lon': (['lon', ], x+(dx/2.0)),
                                             'lat': (['lat', ], y+(dy/2.0)),
                                             'z': (['z', ], z+(dx/2.0)),
                                             'llon': (['lat', 'lon'], xx+(dx/2.0)),
                                             'llat': (['lat', 'lon'], yy+(dy/2.0)),
                                             'zz': (['lat', 'lon', 'z'], zz+(dx/2.0))}
                                    )

In order to infer the cell boundary for both the dimensions('lon', 'lat', 'z')
and the coordinates ('llon', 'llat', 'zz') xgcm needs a dictionary which
assigns dimension/coordinate names to a xgcm axis.
Furthermore in order to correctly handle a periodic boundary like longitude,
the 'wrap' parameter has to be defined (in this case as a dictionary specifying
the value of the discontinutiy at the boundary for each dimension/coordinate).
Similarly the boundary can be padded. In this case the depth dimension/
coordinate is padded with the 'auto' option, which extrapolates the values
linearly

.. code-block:: python

        >>> axis_dims =  {'X':'lon','Y':'lat','Z':'z'}
        >>> axis_coords =  {'X':'llon','Y':'llat','Z':'zz'}
        >>> # Define the position of the gridcells in 'ds_original' as center and infer the left cell boundary (default)
        >>> position=('center','left')
        >>> ds_new = generate_grid_ds(ds_original,
                                      axis_dims,
                                      axis_coords,
                                      position=position,
                                      boundary_discontinuity={'lon':360,
                                                              'lat':180,
                                                              'llon':360,
                                                              'llat':180},
                                      pad={'z':'auto','zz':'auto'})
        >>> ds_new

        <xarray.Dataset>
        Dimensions:        (lat: 180, lat_inferred: 180, lon: 720, lon_inferred: 720, z: 20, z_inferred: 20)
        Coordinates:
          * lon            (lon) float64 -179.8 -179.2 -178.8 -178.2 -177.8 -177.2 ...
          * lat            (lat) float64 -89.5 -88.5 -87.5 -86.5 -85.5 -84.5 -83.5 ...
          * z              (z) float64 0.25 0.75 1.25 1.75 2.25 2.75 3.25 3.75 4.25 ...
            llon           (lat, lon) float64 -179.8 -179.2 -178.8 -178.2 -177.8 ...
            llat           (lat, lon) float64 -89.5 -89.5 -89.5 -89.5 -89.5 -89.5 ...
            zz             (lat, lon, z) float64 0.25 0.75 1.25 1.75 2.25 2.75 3.25 ...
          * lon_inferred   (lon_inferred) float64 -180.0 -179.5 -179.0 -178.5 -178.0 ...
          * lat_inferred   (lat_inferred) float64 -90.0 -89.0 -88.0 -87.0 -86.0 ...
          * z_inferred     (z_inferred) float64 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 ...
            llon_inferred  (lat, lon_inferred) float64 -180.0 -179.5 -179.0 -178.5 ...
            llat_inferred  (lat_inferred, lon) float64 -90.0 -90.0 -90.0 -90.0 -90.0 ...
            zz_inferred    (lat, lon, z_inferred) float64 0.0 0.5 1.0 1.5 2.0 2.5 ...
        Data variables:
            somedata       (lat, lon, z) float64 0.425 0.6889 0.289 0.8077 0.1208 ...


The new dataset can now be used to create an xgcm grid object.
        >>> grid = Grid(ds_new)
        >>> grid
        <xgcm.Grid>
        Z Axis (periodic):
          * center   z (20) --> left
          * left     z_inferred (20) --> center
        X Axis (periodic):
          * center   lon (720) --> left
          * left     lon_inferred (720) --> center
        Y Axis (periodic):
          * center   lat (180) --> left
          * left     lat_inferred (180) --> center
