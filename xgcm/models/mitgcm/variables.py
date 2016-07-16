"""
All of the metadata related to MITgcm variables, grids, naming conventions, etc.
"""
# python 3 compatiblity
from __future__ import print_function, division

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
                long_name="x-dimension of the t grid")),
    i_g = dict(dims=['i_g'], attrs=dict(
                standard_name="x_grid_index_at_u_location", axis='X',
                long_name="x-dimension of the u grid", c_grid_axis_shift=-0.5)),
    i_z = dict(dims=['i_z'], attrs=dict(
                standard_name="x_grid_index_at_f_location", axis='X',
                long_name="x-dimension of the f grid", c_grid_axis_shift=-0.5)),
    # y direction
    j = dict(dims=['j'], attrs=dict(
                standard_name="y_grid_index", axis='Y',
                long_name="y-dimension of the t grid")),
    j_g = dict(dims=['j_g'], attrs=dict(
                standard_name="y_grid_index_at_v_location", axis='Y',
                long_name="y-dimension of the v grid", c_grid_axis_shift=-0.5)),
    j_z = dict(dims=['j_z'], attrs=dict(
                standard_name="y_grid_index_at_f_location", axis='Y',
                long_name="y-dimension of the f grid", c_grid_axis_shift=-0.5)),
    # x direction
    k = dict(dims=['k'], attrs=dict(
                standard_name="z_grid_index", axis="Z",
                long_name="z-dimension of the t grid")),
    k_u = dict(dims=['k_u'], attrs=dict(
                standard_name="z_grid_index_at_lower_w_location",
                axis="Z", long_name="z-dimension of the w grid",
                c_grid_axis_shift=-0.5)),
    k_l = dict(dims=['k_l'], attrs=dict(
                standard_name="z_grid_index_at_upper_w_location",
                axis="Z", long_name="z-dimension of the w grid",
                c_grid_axis_shift=0.5)),
    # this is complicated because it is offset in both directions - allowed by comodo?
    k_p1 = dict(dims=['k_p1'], attrs=dict(
                standard_name="z_grid_index_at_w_location",
                axis="Z", long_name="z-dimension of the w grid",
                c_grid_axis_shift=(-0.5,0.5)))
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
                long_name="vertical coordinate of upp cell interface",
                units="m", positive="down"),
            filename="RF", slice=(slice(None,-1),0,0))
)

horizontal_grid_variables = OrderedDict(
    # tracer cell
    rA  = dict(dims=["j", "i"], attrs=dict(standard_name="cell_area",
                long_name="cell area", units="m2", coordinate="YC XC")),
    dxG = dict(dims=["j_g", "i"], attrs=dict(
                standard_name="cell_x_size_at_v_location",
                long_name="cell x size", units="m", coordinate="YG XC")),
    dyG = dict(dims=["j", "i_g"], attrs=dict(
                standard_name="cell_y_size_at_u_location",
                long_name="cell y size", units="m", coordinate="YC XG")),
    Depth=dict(dims=["j", "i"], attrs=dict( standard_name="ocean_depth",
                long_name="ocean depth", units="m", coordinate="XC YC")),
    # vorticity cell
    rAz  = dict(dims=["j_g", "i_g"], attrs=dict(
                standard_name="cell_area_at_f_location",
                long_name="cell area", units="m", coordinate="YG XG")),
    dxC = dict(dims=["j", "i_g"], attrs=dict(
                standard_name="cell_x_size_at_u_location",
                long_name="cell x size", units="m", coordinate="YC XG")),
    dyC = dict(dims=["j_g", "i"], attrs=dict(
                standard_name="cell_y_size_at_v_location",
                long_name="cell y size", units="m", coordinate="YG XC")),
    # u cell
    rAw = dict(dims=["j", "i_g"], attrs=dict(
                standard_name="cell_area_at_u_location",
                long_name="cell area", units="m2", coordinate="YG XC")),
    # v cell
    rAs = dict(dims=["j_g", "i"], attrs=dict(
                standard_name="cell_area_at_v_location",
                long_name="cell area", units="m2", coordinates="YG XC")),
)

vertical_grid_variables = OrderedDict(
    drC = dict(dims=['k_p1'], attrs=dict(
                standard_name="cell_z_size_at_w_location",
                long_name="cell z size", units="m")),
    drF = dict(dims=['k'], attrs=dict(
                standard_name="cell_z_size",
                long_name="cell z size", units="m")),
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
                standard_name='sea_water_x_velocity',
                long_name='Zonal Component of Velocity', units='m s-1')),
    V = dict(dims=['k','j_g','i'], attrs=dict(
                standard_name='sea_water_y_velocity',
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
    # TODO: finish encoding this crap!
    # uVeltave=dict(dims=['k','j','i_g'], 'Zonal Component of Velocity', 'm/s'),
    # vVeltave=dict(dims=['k','j_g','i'], 'Meridional Component of Velocity', 'm/s'),
    # wVeltave=dict(dims=['k_l','j','i'], 'Vertical Component of Velocity', 'm/s'),
    # Ttave=dict(dims=['k','j','i'], 'Potential Temperature', 'degC'),
    # Stave=dict(dims=['k','j','i'], 'Salinity', 'psu'),
    # PhHytave=dict(dims=['k','j','i'], 'Hydrostatic Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    # PHLtave=dict(dims=['j','i'], 'Bottom Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    # ETAtave=dict(dims=['j','i'], 'Surface Height Anomaly', 'm'),
    # Convtave=dict(dims=['k_l','j','i'], "Convective Adjustment Index", "none [0-1]"),
    # Eta2tave=dict(dims=['j','i'], "Square of Surface Height Anomaly", "m^2"),
    # PHL2tave=dict(dims=['j','i'], 'Square of Hyd. Pressure Pot.(p/rho) Anomaly', 'm^4/s^4'),
    # sFluxtave=dict(dims=['j','i'], 'total salt flux (match salt-content variations), >0 increases salt', 'g/m^2/s'),
    # Tdiftave=dict(dims=['k_l','j','i'], "Vertical Diffusive Flux of Pot.Temperature", "degC.m^3/s"),
    # tFluxtave=dict(dims=['j','i'], "Total heat flux (match heat-content variations), >0 increases theta", "W/m^2"),
    # TTtave=dict(dims=['k','j','i'], 'Squared Potential Temperature', 'degC^2'),
    # uFluxtave=dict(dims=['j','i_g'], 'surface zonal momentum flux, positive -> increase u', 'N/m^2'),
    # UStave=dict(dims=['k','j','i_g'], "Zonal Transport of Salinity", "psu m/s"),
    # UTtave=dict(dims=['k','j','i_g'], "Zonal Transport of Potenial Temperature", "degC m/s"),
    # UUtave=dict(dims=['k','j','i_g'], "Zonal Transport of Zonal Momentum", "m^2/s^2"),
    # UVtave=dict(dims=['k','j_g','i_g'], 'Product of meridional and zonal velocity', 'm^2/s^2'),
    # vFluxtave=dict(dims=['j_g','i'], 'surface meridional momentum flux, positive -> increase v', 'N/m^2'),
    # VStave=dict(dims=['k','j_g','i'], "Meridional Transport of Salinity", "psu m/s"),
    # VTtave=dict(dims=['k','j_g','i'], "Meridional Transport of Potential Temperature", "degC m/s"),
    # VVtave=dict(dims=['k','j_g','i'], 'Zonal Transport of Zonal Momentum', 'm^2/s^2'),
    # WStave=dict(dims=['k_l','j','i'], 'Vertical Transport of Salinity', "psu m/s"),
    # WTtave=dict(dims=['k_l','j','i'], 'Vertical Transport of Potential Temperature', "degC m/s"),
)

# should find a better way to inlude the package variables
# _state_variables['GM_Kwx-T'] = (
#         dims=['k_l','j','i'], 'K_31 element (W.point, X.dir) of GM-Redi tensor','m^2/s')
# _state_variables['GM_Kwy-T'] = (
#         dims=['k_l','j','i'], 'K_33 element (W.point, X.dir) of GM-Redi tensor','m^2/s')
# _state_variables['GM_Kwz-T'] = (
#         dims=['k_l','j','i'], 'K_33 element (W.point, X.dir) of GM-Redi tensor','m^2/s')
#
#
# Nptracers=99
# _ptracers = { 'PTRACER%02d' % n :
#                (dims=['k','j','i'], 'PTRACER%02d Concentration' % n, "tracer units/m^3")
#                for n in range(Nptracers)}
