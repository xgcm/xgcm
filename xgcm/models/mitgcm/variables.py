"""
All of the metadata related to MITgcm variables, grids, naming conventions, etc.
"""
# python 3 compatiblity
from __future__ import print_function, division
import numpy as np

from xarray.core.pycompat import OrderedDict

# We are trying to combine the following two things:
# - MITgcm grid
#   http://mitgcm.org/sealion/online_documents/node47.html
# - Comodo conventions
#   http://pycomodo.forge.imag.fr/norm.html
# To do that, we have to set the proper metadata (attributes) for the MITgcm
# variables.

# The spatial dimensions, all 1D
# There is no "data" to go with these. They are just indices.
dimensions = OrderedDict(
    # x direction
    i = dict(dims=['i'], attrs=dict(
                standard_name="x_grid_index", axis='X',
                long_name="x-dimension of the t grid",
                swap_dim='XC')),
    i_g = dict(dims=['i_g'], attrs=dict(
                standard_name="x_grid_index_at_u_location", axis='X',
                long_name="x-dimension of the u grid", c_grid_axis_shift=-0.5,
                swap_dim='XG')),
    # i_z = dict(dims=['i_z'], swap_dim='XG', attrs=dict(
    #             standard_name="x_grid_index_at_f_location", axis='X',
    #             long_name="x-dimension of the f grid", c_grid_axis_shift=-0.5)),
    # y direction
    j = dict(dims=['j'], attrs=dict(
                standard_name="y_grid_index", axis='Y',
                long_name="y-dimension of the t grid", swap_dim='YC')),
    j_g = dict(dims=['j_g'], attrs=dict(
                standard_name="y_grid_index_at_v_location", axis='Y',
                long_name="y-dimension of the v grid", c_grid_axis_shift=-0.5,
                swap_dim='YG')),
    # j_z = dict(dims=['j_z'], swap_dim='YG', attrs=dict(
    #             standard_name="y_grid_index_at_f_location", axis='Y',
    #             long_name="y-dimension of the f grid", c_grid_axis_shift=-0.5)),
    # x direction
    k = dict(dims=['k'], attrs=dict(
                standard_name="z_grid_index", axis="Z",
                long_name="z-dimension of the t grid", swap_dim='Z')),
    k_u = dict(dims=['k_u'], attrs=dict(
                standard_name="z_grid_index_at_lower_w_location",
                axis="Z", long_name="z-dimension of the w grid",
                c_grid_axis_shift=-0.5, swap_dim='Zu')),
    k_l = dict(dims=['k_l'], attrs=dict(
                standard_name="z_grid_index_at_upper_w_location",
                axis="Z", long_name="z-dimension of the w grid",
                c_grid_axis_shift=0.5, swap_dim='Zl')),
    # this is complicated because it is offset in both directions - allowed by comodo?
    k_p1 = dict(dims=['k_p1'], attrs=dict(
                standard_name="z_grid_index_at_w_location",
                axis="Z", long_name="z-dimension of the w grid",
                c_grid_axis_shift=(-0.5,0.5), swap_dim='Zp1'))
)

horizontal_coordinates_spherical = OrderedDict(
    XC = dict(dims=["j", "i"], attrs=dict(
                standard_name="longitude", long_name="longitude",
                units="degrees_east", coordinate="YC XC")),
    YC = dict(dims=["j", "i"], attrs=dict(
                standard_name="latitude", long_name="latitude",
                units="degrees_north", coordinate="YC XC")),
    XG = dict(dims=["j_g", "i_g"], attrs=dict(
                standard_name="longitude_at_f_location", long_name="longitude",
                units="degrees_east", coordinate="YG XG")),
    YG = dict(dims=["j_g", "i_g"], attrs=dict(
                standard_name="latitude_at_f_location", long_name="latitude",
                units="degrees_north", coordinates="YG XG"))
)

horizontal_coordinates_cartesian = OrderedDict(
    XC = dict(dims=["j", "i"], attrs=dict(
                standard_name="plane_x_coordinate", long_name="x coordinate",
                units="m", coordinate="YC XC")),
    YC = dict(dims=["j", "i"], attrs=dict(
                standard_name="plane_y_coordinate", long_name="y coordinate",
                units="m", coordinate="YC XC")),
    XG = dict(dims=["j_g", "i_g"], attrs=dict(
                standard_name="plane_x_coordinate_at_f_location",
                long_name="x coordinate", units="m", coordinate="YG XG")),
    YG = dict(dims=["j_g", "i_g"], attrs=dict(
                standard_name="plane_y_coordinate_at_f_location",
                long_name="y coordinate", units="m", coordinates="YG XG"))
)

vertical_coordinates = OrderedDict(
    Z = dict(dims=["k"], attrs=dict(
                standard_name="depth",
                long_name="vertical coordinate of cell center",
                units="m", positive="down"),
            filename="RC", slice=(slice(None),0,0)),
    Zp1 = dict(dims=["k_p1"], attrs=dict(
                standard_name="depth_at_w_location",
                long_name="vertical coordinate of cell interface",
                units="m", positive="down"),
            filename="RF", slice=(slice(None),0,0)),
    Zu= dict(dims=["k_u"], attrs=dict(
                standard_name="depth_at_lower_w_location",
                long_name="vertical coordinate of lower cell interface",
                units="m", positive="down"),
            filename="RF", slice=(slice(1,None),0,0)),
    Zl= dict(dims=["k_l"], attrs=dict(
                standard_name="depth_at_upper_w_location",
                long_name="vertical coordinate of upper cell interface",
                units="m", positive="down"),
            filename="RF", slice=(slice(None,-1),0,0))
)

# I got these variable names from the default MITgcm netcdf output
horizontal_grid_variables = OrderedDict(
    # tracer cell
    rA  = dict(dims=["j", "i"], attrs=dict(standard_name="cell_area",
                long_name="cell area", units="m2", coordinate="YC XC"),
               filename='RAC'),
    dxG = dict(dims=["j_g", "i"], attrs=dict(
                standard_name="cell_x_size_at_v_location",
                long_name="cell x size", units="m", coordinate="YG XC"),
               filename='DXG'),
    dyG = dict(dims=["j", "i_g"], attrs=dict(
                standard_name="cell_y_size_at_u_location",
                long_name="cell y size", units="m", coordinate="YC XG"),
               filename='DYG'),
    Depth=dict(dims=["j", "i"], attrs=dict( standard_name="ocean_depth",
                long_name="ocean depth", units="m", coordinate="XC YC")),
    # vorticity cell
    rAz  = dict(dims=["j_g", "i_g"], attrs=dict(
                standard_name="cell_area_at_f_location",
                long_name="cell area", units="m", coordinate="YG XG"),
               filename='RAZ'),
    dxC = dict(dims=["j", "i_g"], attrs=dict(
                standard_name="cell_x_size_at_u_location",
                long_name="cell x size", units="m", coordinate="YC XG"),
               filename='DXC'),
    dyC = dict(dims=["j_g", "i"], attrs=dict(
                standard_name="cell_y_size_at_v_location",
                long_name="cell y size", units="m", coordinate="YG XC"),
               filename='DYC'),
    # u cell
    rAw = dict(dims=["j", "i_g"], attrs=dict(
                standard_name="cell_area_at_u_location",
                long_name="cell area", units="m2", coordinate="YG XC"),
               filename='RAW'),
    # v cell
    rAs = dict(dims=["j_g", "i"], attrs=dict(
                standard_name="cell_area_at_v_location",
                long_name="cell area", units="m2", coordinates="YG XC"),
               filename='RAZ'),
)

vertical_grid_variables = OrderedDict(
    drC = dict(dims=['k_p1'], attrs=dict(
                standard_name="cell_z_size_at_w_location",
                long_name="cell z size", units="m"), filename='DRC'),
    drF = dict(dims=['k'], attrs=dict(
                standard_name="cell_z_size",
                long_name="cell z size", units="m"), filename='DRF'),
    PHrefC = dict(dims=['k'], attrs=dict(
                standard_name="cell_reference_pressure",
                long_name='Reference Hydrostatic Pressure', units='m2 s-2')),
    PHrefF = dict(dims=['k_p1'], attrs=dict(
                standard_name="cell_reference_pressure",
                long_name='Reference Hydrostatic Pressure', units='m2 s-2'))
)

volume_grid_variables = OrderedDict(
    hFacC = dict(dims=['k','j','i'], attrs=dict(
                    standard_name="cell_vertical_fraction",
                    long_name="vertical fraction of open cell")),
    hFacW = dict(dims=['k','j','i_g'], attrs=dict(
                    standard_name="cell_vertical_fraction_at_u_location",
                    long_name="vertical fraction of open cell")),
    hFacS = dict(dims=['k','j_g','i'], attrs=dict(
                    standard_name="cell_vertical_fraction_at_v_location",
                    long_name="vertical fraction of open cell"))
)

# this a template: NAME gets replaced with the layer name (e.g. 1RHO)
#                  dim gets prepended with the layer number (e.g. l1_b)
layers_grid_variables = OrderedDict(
    layer_NAME_bounds = dict(dims=['l_b'], attrs=dict(
                standard_name="ocean_layer_coordinate_NAME_bounds",
                long_name="boundaries points of layer NAME"),
            filename="layersNAME", slice=(slice(None),0,0)),
    layer_NAME_center = dict(dims=['l_c'], attrs=dict(
                standard_name="ocean_layer_coordinate_NAME_center",
                long_name="center points of layer NAME"),
            filename="layersNAME", slice=(slice(None),0,0),
            # if we don't convert to array, dask can't tokenize
            # https://github.com/pydata/xarray/issues/1014
            transform=(lambda x: np.asarray(0.5*(x[1:] + x[:-1])))),
    layer_NAME_interface = dict(dims=['l_i'], attrs=dict(
                standard_name="ocean_layer_coordinate_NAME_interface",
                long_name="interface points of layer NAME"),
            filename="layersNAME", slice=(slice(1,-1),0,0))
)

# _grid_special_mapping = {
# # name: (file_name, slice_to_extract, expecting_3D_field)
#     'Z': ('RC', (slice(None),0,0), 3),
#     'Zp1': ('RF', (slice(None),0,0), 3),
#     'Zu': ('RF', (slice(1,None),0,0), 3),
#     'Zl': ('RF', (slice(None,-1),0,0), 3),
#     # this will create problems with some curvillinear grids
#     # whate if X and Y need to be 2D?
#     'X': ('XC', (0,slice(None)), 2),
#     'Y': ('YC', (slice(None),0), 2),
#     'Xp1': ('XG', (0,slice(None)), 2),
#     'Yp1': ('YG', (slice(None),0), 2),
#     'rA': ('RAC', (slice(None), slice(None)), 2),
#     'HFacC': ('hFacC', 3*(slice(None),), 3),
#     'HFacW': ('hFacW', 3*(slice(None),), 3),
#     'HFacS': ('hFacS', 3*(slice(None),), 3),
# }


# also try to match CF standard names
# http://cfconventions.org/Data/cf-standard-names/28/build/cf-standard-name-table.html

state_variables = OrderedDict(
    # default state variables
    U = dict(dims=['k','j','i_g'], attrs=dict(
                standard_name='sea_water_x_velocity', mate='V',
                long_name='Zonal Component of Velocity', units='m s-1')),
    V = dict(dims=['k','j_g','i'], attrs=dict(
                standard_name='sea_water_y_velocity', mate='U',
                long_name='Meridional Component of Velocity', units='m s-1')),
    W = dict(dims=['k_l','j','i'], attrs=dict(
                standard_name='sea_water_z_velocity',
                long_name='Vertical Component of Velocity', units='m s-1')),
    T = dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_potential_temperature",
                long_name='Potential Temperature', units='degree_Celcius')),
    S = dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_salinity",
                long_name='Salinity', units='psu')),
    PH= dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_dynamic_pressue",
                long_name='Hydrostatic Pressure Pot.(p/rho) Anomaly',
                units='m2 s-2')),
    PHL=dict(dims=['j','i'], attrs=dict(
                standard_name="sea_water_dynamic_pressure_at_sea_floor",
                long_name='Bottom Pressure Pot.(p/rho) Anomaly',
                units='m2 s-2')),
    Eta=dict(dims=['j','i'], attrs=dict(
                standard_name="sea_surface_height_above_geoid",
                long_name='Surface Height Anomaly', units='m')),
    # default tave variables
    uVeltave = dict(dims=['k','j','i_g'], attrs=dict(
                standard_name='sea_water_x_velocity', mate='vVeltave',
                long_name='Zonal Component of Velocity', units='m s-1')),
    vVeltave = dict(dims=['k','j_g','i'], attrs=dict(
                standard_name='sea_water_y_velocity', mate='uVeltave',
                long_name='Meridional Component of Velocity', units='m s-1')),
    wVeltave = dict(dims=['k_l','j','i'], attrs=dict(
                standard_name='sea_water_z_velocity',
                long_name='Vertical Component of Velocity', units='m s-1')),
    Ttave = dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_potential_temperature",
                long_name='Potential Temperature', units='degree_Celcius')),
    Stave = dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_salinity",
                long_name='Salinity', units='psu')),
    PhHytave= dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_dynamic_pressue",
                long_name='Hydrostatic Pressure Pot.(p/rho) Anomaly',
                units='m2 s-2')),
    PHLtave=dict(dims=['j','i'], attrs=dict(
                standard_name="sea_water_dynamic_pressure_at_sea_floor",
                long_name='Bottom Pressure Pot.(p/rho) Anomaly',
                units='m2 s-2')),
    ETAtave=dict(dims=['j','i'], attrs=dict(
                standard_name="sea_surface_height_above_geoid",
                long_name='Surface Height Anomaly', units='m')),
    # TODO: finish encoding this crap!
    Convtave=dict(dims=['k_l','j','i'], attrs=dict(
                standard_name="convective_adjustment_index",
                long_name="Convective Adjustment Index")),
    Eta2tave=dict(dims=['j','i'], attrs=dict(
                standard_name="square_of_sea_surface_height_above_geoid",
                long_name='Square of Surface Height Anomaly', units='m2')),
    PHL2tave=dict(dims=['j','i'], attrs=dict(
                standard_name="square_of_sea_water_dynamic_pressure_at_sea_floor",
                long_name='Square of Bottom Pressure Pot.(p/rho) Anomaly',
                units='m4 s-4')),
    sFluxtave=dict(dims=['j','i'], attrs=dict(
                standard_name="virtual_salt_flux_into_sea_water",
                long_name='total salt flux (match salt-content variations), '
                          '>0 increases salt', units='g m-2 s-1')),
    Tdiftave=dict(dims=['k_l','j','i'], attrs=dict(
                standard_name="upward_potential_temperature_flux_"
                              "due_to_diffusion",
                long_name="Vertical Diffusive Flux of Pot.Temperature",
                units="K m3 s-1")),
    tFluxtave=dict(dims=['j','i'], attrs=dict(
                standard_name="surface_downward_heat_flux_in_sea_water",
                long_name="Total heat flux (match heat-content variations), "
                          ">0 increases theta", units="W m-2")),
    TTtave = dict(dims=['k','j','i'], attrs=dict(
                standard_name="square_of_sea_water_potential_temperature",
                long_name='Square of Potential Temperature',
                units='degree_Celcius')),
    uFluxtave=dict(dims=['j','i_g'], attrs=dict(
                standard_name="surface_downward_x_stress",
                long_name='surface zonal momentum flux, positive -> increase u',
                units='N m-2', mate='vFluxtave')),
    UStave=dict(dims=['k','j','i_g'], attrs=dict(
                standard_name="product_of_sea_water_x_velocity_and_salinity",
                long_name="Zonal Transport of Salinity",
                units="psu m s-1", mate='VStave')),
    UTtave=dict(dims=['k','j','i_g'], attrs=dict(
                standard_name="product_of_sea_water_x_velocity_and_"
                              "potential_temperature",
                long_name="Zonal Transport of Potential Temperature",
                units="K m s-1", mate='VTtave')),
    UUtave = dict(dims=['k','j','i_g'], attrs=dict(
                standard_name='square_of_sea_water_x_velocity',
                long_name='Square of Zonal Component of Velocity',
                units='m2 s-2')),
    UVtave=dict(dims=['k','j_g','i_g'], attrs=dict(
                standard_name="product_of_sea_water_x_velocity_and_"
                              "sea_water_y_velocity",
                long_name="Product of meridional and zonal velocity",
                units="m2 s-2")),
    vFluxtave=dict(dims=['j_g','i'], attrs=dict(
                standard_name="surface_downward_y_stress",
                long_name='surface meridional momentum flux, '
                          'positive -> increase u',
                units='N m-2', mate='uFluxtave')),
    VStave=dict(dims=['k','j_g','i'], attrs=dict(
                standard_name="product_of_sea_water_y_velocity_and_salinity",
                long_name="Meridional Transport of Salinity",
                units="psu m s-1", mate='UStave')),
    VTtave=dict(dims=['k','j_g','i'], attrs=dict(
                standard_name="product_of_sea_water_y_velocity_and_"
                              "potential_temperature",
                long_name="Meridional Transport of Potential Temperature",
                units="K m s-1", mate='UTtave')),
    VVtave = dict(dims=['k','j_g','i'], attrs=dict(
                standard_name='square_of_sea_water_y_velocity',
                long_name='Square of Meridional Component of Velocity',
                units='m2 s-2')),
    WStave=dict(dims=['k_l','j','i'], attrs=dict(
                standard_name="product_of_sea_water_z_velocity_and_salinity",
                long_name="Vertical Transport of Salinity",
                units="psu m s-1")),
    WTtave=dict(dims=['k_l','j','i'], attrs=dict(
                standard_name="product_of_sea_water_z_velocity_and_"
                              "potential_temperature",
                long_name="Vertical Transport of Potential Temperature",
                units="K m s-1")),
)

# these variable names have hyphens and need a different syntax
# state_variables['GM_Kwx-T'] = dict(
#     dims=['k_l','j','i'], attrs=dict(
#         standard_name="K_31_element_of_GMRedi_tensor",
#         long_name="K_31 element (W.point, X.dir) of GM-Redi tensor",
#         units="m2 s-1"
#     )
# )
#
# state_variables['GM_Kwy-T'] = dict(
#     dims=['k_l','j','i'], attrs=dict(
#         standard_name="K_32_element_of_GMRedi_tensor",
#         long_name="K_32 element (W.point, Y.dir) of GM-Redi tensor",
#         units="m2 s-1"
#     )
# )
#
# state_variables['GM_Kwy-T'] = dict(
#     dims=['k_l','j','i'], attrs=dict(
#         standard_name="K_33_element_of_GMRedi_tensor",
#         long_name="K_33 element (W.point, Z.dir) of GM-Redi tensor",
#         units="m2 s-1"
#     )
# )


# Nptracers=99
# _ptracers = { 'PTRACER%02d' % n :
#                (dims=['k','j','i'], 'PTRACER%02d Concentration' % n, "tracer units/m^3")
#                for n in range(Nptracers)}
