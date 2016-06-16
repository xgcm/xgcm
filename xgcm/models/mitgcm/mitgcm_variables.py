"""
A place to store all of the metadata related to MITgcm variables, naming
conventions, etc.
"""

from xarray.core.pycompat import OrderedDict

# comodo conventions
# http://pycomodo.forge.imag.fr/norm.html

# the spatial dimensions, all 1D
dimensions = OrderedDict(
# dimension_name = (axis, standard_name, long_name, c_grid_dynamic_range, c_grid_axis_shift)
    i = ("X", "x_grid_index", "x-dimension of the grid", None, None),
    i_g = ("X", "x_grid_index_at_u_location", "x-dimension of the grid", None, -0.5),
    i_z = ("X", "x_grid_index_at_f_location", "x-dimension of the grid", None, -0.5),
    j = ("Y", "y_grid_index", "y-dimension of the grid", None, None),
    j_g = ("Y", "y_grid_index_at_v_location", "y-dimension of the grid", None, -0.5),
    j_z = ("Y", "y_grid_index_at_f_location", "y-dimension of the grid", None, -0.5),
    k = ("Z", "z_grid_index", "z-dimension of the grid", None, None),
    k_u = ("Z", "z_grid_index_at_w_location", "z-dimension of the grid", None, -0.5),
    k_l = ("Z", "z_grid_index_at_w_location", "z-dimension of the grid", None, 0.5),
    # this is complicated because it is offset in both directions - allowed by comodo?
    k_p1 = ("Z", "z_grid_index_at_w_location", "z-dimension of the grid", None, (-0.5,0.5)),
)

# MITgcm reference
# http://mitgcm.org/sealion/online_documents/node47.html

# coordinates
horizontal_coordinates_spherical = OrderedDict(
# coordinate_name = (dims, standard_name, long_name, units, coordinates)
    XC = (["j", "i"], "longitude", "longitude", "degrees_east", "YC XC"),
    YC = (["j", "i"], "latitude", "latitude", "degrees_north", "YC XC"),
    XG = (["j_g", "i_g"], "longitude_at_f_location", "longitude", "degrees_east", "YG XG"),
    YG = (["j_g", "i_g"], "latitude_at_f_location", "latitude", "degrees_north", "YG XG"),
)

horizontal_coordinates_cartesian = OrderedDict(
    XC = (["j", "i"], "plane_x_coordinate", "x coordinate", "m", "YC XC"),
    YC = (["j", "i"], "plane_y_coordinate", "y coordinate", "m", "YC XC"),
    XG = (["j_g", "i_g"], "plane_x_coordinate_at_f_location", "x coordinate", "m", "YG XG"),
    YG = (["j_g", "i_g"], "plane_y_coordinate_at_f_location", "y coordinate", "m", "YG XG")
)

vertical_coordinates = OrderedDict(
# coordinate_name = (dims, standard_name, long_name, units, positive)
    Z = (["k"], "depth", "depth", "m", "down"),
    Zu= (["k_u"], "depth_at_w_location", "depth", "m", "down"),
    Zl= (["k_l"], "depth_at_w_location", "depth", "m", "down"),
)

horizontal_grid_variables = OrderedDict(
# coordinate_name = (dims, standard_name, long_name, units, coordinates)
    # tracer cell
    rA  = (["j", "i"], "cell_area", "cell area", "m^2", "YC XC"),
    dxG = (["j_g", "i"], "cell_x_size_at_v_location", "cell x size", "m", "YG XC"),
    dyG = (["j", "i_g"], "cell_y_size_at_u_location", "cell y size", "m", "YC XG"),
    Depth=(["j", "i"], "ocean_depth", "ocean depth", "m", "XC YC"),
    # vorticity cell
    rAz  = (["j_g", "i_g"], "cell_area_at_f_location", "cell area", "m^2", "YG XG"),
    dxC = (["j", "i_g"], "cell_x_size_at_u_location", "cell x size", "m", "YC XG"),
    dyG = (["j_g", "i"], "cell_y_size_at_v_location", "cell y size", "m", "YG XC"),
    # u cell
    rAw = (["j", "i_g"], "cell_area_at_u_location", "cell area", "m^2", "YG XC"),
    # v cell
    rAs = (["j_g", "i"], "cell_area_at_v_location", "cell area", "m^2", "YG XC"),
)

vertical_grid_variables = OrderedDict(
# coordinate_name = (dims, standard_name, long_name, units)
    drC = (['k_p1'], "cell_z_size_at_w_location", "cell z size" "m"),
    drF = (['k'], "cell_z_size", "cell z size" "m"),
)

volume_grid_variables = OrderedDict(


)
_grid_variables = xray.core.pycompat.OrderedDict(
    # horizontal grid
    X=   (('X',), "X-coordinate of cell center", "meters"),
    Y=   (('Y',), "Y-coordinate of cell center", "meters"),
    Xp1= (('Xp1',), "X-coordinate of cell corner", "meters"),
    Yp1= (('Yp1',), "Y-coordinate of cell corner", "meters"),
    # 2d versions
    XC=  (('Y','X'), "X coordinate of cell center (T-P point)", "degree_east"),
    YC=  (('Y','X'), "Y coordinate of cell center (T-P point)", "degree_north"),
    XG=  (('Yp1','Xp1'), "X coordinate of cell corner (Vorticity point)", "degree_east"),
    YG=  (('Yp1','Xp1'), "Y coordinate of cell corner (Vorticity point)", "degree_north"),
    # vertical grid
    Z=   (('Z',), "vertical coordinate of cell center", "meters"),
    Zp1= (('Zp1',), "vertical coordinate of cell interface", "meters"),
    Zu=  (('Zu',), "vertical coordinate of lower cell interface", "meters"),
    Zl=  (('Zl',), "vertical coordinate of upper cell interface", "meters"),
    # (for some reason, the netCDF files use both R and Z notation )
#    'RC':  (('Z',), "R coordinate of cell center", "m"),
#    'RF':  (('Zp1',), "R coordinate of cell interface", "m"),
#    'RU':  (('Zu',), "R coordinate of lower cell interface", "m"),
#    'RL':  (('Zl',), "R coordinate of upper cell interface", "m"),
    # horiz. differentials
    dxC= (('Y','Xp1'), "x cell center separation", "meters"),
    dyC= (('Yp1','X'), "y cell center separation", "meters"),
    dxG= (('Yp1','X'), "x cell corner separation", "meters"),
    dyG= (('Y','Xp1'), "y cell corner separation", "meters"),
    # vert. differentials
    drC= (('Zp1',), "r cell center separation", "m"),
    drF= (('Z',), "r cell face separation", "m"),
    # areas
    rA=  (('Y','X'), "r-face area at cell center", "m^2"),
    rAw= (('Y','Xp1'), "r-face area at U point", "m^2"),
    rAs= (('Yp1','X'), "r-face area at V point", "m^2"),
    rAz= (('Yp1','Xp1'), "r-face area at cell corner", "m^2"),
    # depth
    Depth=(('Y','X'), "fluid thickness in r coordinates (at rest)", "meters"),
    # thickness factors
    HFacC=(('Z','Y','X'),
             "vertical fraction of open cell at cell center", "none (0-1)"),
    HFacW=(('Z','Y','Xp1'),
             "vertical fraction of open cell at West face", "none (0-1)"),
    HFacS=(('Z','Yp1','X'),
             "vertical fraction of open cell at South face", "none (0-1)"),
    PHrefC=(('Z',), 'Reference Hydrostatic Pressure', 'm^2/s^2'),
    PHrefF=(('Zp1',), 'Reference Hydrostatic Pressure', 'm^2/s^2')
)

_grid_special_mapping = {
# name: (file_name, slice_to_extract, expecting_3D_field)
    'Z': ('RC', (slice(None),0,0), 3),
    'Zp1': ('RF', (slice(None),0,0), 3),
    'Zu': ('RF', (slice(1,None),0,0), 3),
    'Zl': ('RF', (slice(None,-1),0,0), 3),
    # this will create problems with some curvillinear grids
    # whate if X and Y need to be 2D?
    'X': ('XC', (0,slice(None)), 2),
    'Y': ('YC', (slice(None),0), 2),
    'Xp1': ('XG', (0,slice(None)), 2),
    'Yp1': ('YG', (slice(None),0), 2),
    'rA': ('RAC', (slice(None), slice(None)), 2),
    'HFacC': ('hFacC', 3*(slice(None),), 3),
    'HFacW': ('hFacW', 3*(slice(None),), 3),
    'HFacS': ('hFacS', 3*(slice(None),), 3),
}

_state_variables = xray.core.pycompat.OrderedDict(
    # state
    U=  (('Z','Y','Xp1'), 'Zonal Component of Velocity', 'm/s'),
    V=  (('Z','Yp1','X'), 'Meridional Component of Velocity', 'm/s'),
    W=  (('Zl','Y','X'), 'Vertical Component of Velocity', 'm/s'),
    T=  (('Z','Y','X'), 'Potential Temperature', 'degC'),
    S=  (('Z','Y','X'), 'Salinity', 'psu'),
    PH= (('Z','Y','X'), 'Hydrostatic Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    PHL=(('Y','X'), 'Bottom Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    Eta=(('Y','X'), 'Surface Height Anomaly', 'm'),
    # tave
    uVeltave=(('Z','Y','Xp1'), 'Zonal Component of Velocity', 'm/s'),
    vVeltave=(('Z','Yp1','X'), 'Meridional Component of Velocity', 'm/s'),
    wVeltave=(('Zl','Y','X'), 'Vertical Component of Velocity', 'm/s'),
    Ttave=(('Z','Y','X'), 'Potential Temperature', 'degC'),
    Stave=(('Z','Y','X'), 'Salinity', 'psu'),
    PhHytave=(('Z','Y','X'), 'Hydrostatic Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    PHLtave=(('Y','X'), 'Bottom Pressure Pot.(p/rho) Anomaly', 'm^2/s^2'),
    ETAtave=(('Y','X'), 'Surface Height Anomaly', 'm'),
    Convtave=(('Zl','Y','X'), "Convective Adjustment Index", "none [0-1]"),
    Eta2tave=(('Y','X'), "Square of Surface Height Anomaly", "m^2"),
    PHL2tave=(('Y','X'), 'Square of Hyd. Pressure Pot.(p/rho) Anomaly', 'm^4/s^4'),
    sFluxtave=(('Y','X'), 'total salt flux (match salt-content variations), >0 increases salt', 'g/m^2/s'),
    Tdiftave=(('Zl','Y','X'), "Vertical Diffusive Flux of Pot.Temperature", "degC.m^3/s"),
    tFluxtave=(('Y','X'), "Total heat flux (match heat-content variations), >0 increases theta", "W/m^2"),
    TTtave=(('Z','Y','X'), 'Squared Potential Temperature', 'degC^2'),
    uFluxtave=(('Y','Xp1'), 'surface zonal momentum flux, positive -> increase u', 'N/m^2'),
    UStave=(('Z','Y','Xp1'), "Zonal Transport of Salinity", "psu m/s"),
    UTtave=(('Z','Y','Xp1'), "Zonal Transport of Potenial Temperature", "degC m/s"),
    UUtave=(('Z','Y','Xp1'), "Zonal Transport of Zonal Momentum", "m^2/s^2"),
    UVtave=(('Z','Yp1','Xp1'), 'Product of meridional and zonal velocity', 'm^2/s^2'),
    vFluxtave=(('Yp1','X'), 'surface meridional momentum flux, positive -> increase v', 'N/m^2'),
    VStave=(('Z','Yp1','X'), "Meridional Transport of Salinity", "psu m/s"),
    VTtave=(('Z','Yp1','X'), "Meridional Transport of Potential Temperature", "degC m/s"),
    VVtave=(('Z','Yp1','X'), 'Zonal Transport of Zonal Momentum', 'm^2/s^2'),
    WStave=(('Zl','Y','X'), 'Vertical Transport of Salinity', "psu m/s"),
    WTtave=(('Zl','Y','X'), 'Vertical Transport of Potential Temperature', "degC m/s"),
)
# should find a better way to inlude the package variables
_state_variables['GM_Kwx-T'] = (
        ('Zl','Y','X'), 'K_31 element (W.point, X.dir) of GM-Redi tensor','m^2/s')
_state_variables['GM_Kwy-T'] = (
        ('Zl','Y','X'), 'K_33 element (W.point, X.dir) of GM-Redi tensor','m^2/s')
_state_variables['GM_Kwz-T'] = (
        ('Zl','Y','X'), 'K_33 element (W.point, X.dir) of GM-Redi tensor','m^2/s')


Nptracers=99
_ptracers = { 'PTRACER%02d' % n :
               (('Z','Y','X'), 'PTRACER%02d Concentration' % n, "tracer units/m^3")
               for n in range(Nptracers)}
