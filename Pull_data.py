# %%
from __future__ import division  # makes division not round with integers
from numba import jit
import pygrib
from netCDF4 import Dataset
import numpy as np
import xarray as xr
# import pandas as pd
# import matplotlib.pyplot as plt

# # Assuming 'year' and 'common_object.dt' are defined somewhere in your code
# year = 2022  # Example year
# time_interval_hours = 6  # Example time interval in hours

# # Create a DatetimeIndex with specified frequency
# start_date = pd.Timestamp(year, 5, 1)
# end_date = pd.Timestamp(year, 11, 1)
# date_range = pd.date_range(
#     start=start_date, end=end_date, freq=f'{time_interval_hours}H')


'''
Pull_data.py contains functions that are used by AEW_Tracks.py. All of the
functions in Pull_Data.py aid in the pulling of variables and formatting
them to be in a common format so AEW_Tracks.py does not need to know what
model the data came from.
'''


def get_common_track_data(common_object):
    """
    Obtain common data necessary for tracking atmospheric features over
    Africa and the Atlantic.

    This function takes a common_object containing information about the
    atmospheric model and assigns the latitude, longitude, latitude/longitude
    indices over Africa/the Atlantic, and a time step value (dt) based on
    the data source specified by the model attribute of
    the common_object.

    Parameters:
        common_object (object): An object containing common properties such
        as model type.

    Returns:
        None
    """

    # box of interest over Africa/the Atlantic (values are from Albany)
    north_lat = 30.
    south_lat = 5.
    west_lon = -45.
    east_lon = 25.

    # lat/lon values to crop data to speed up vorticity calculations
    north_lat_crop = 50.
    south_lat_crop = -20.
    west_lon_crop = -80.  # 90
    east_lon_crop = 40.

    # Pressure levels at which to retrieve data (hPa)
    level = [850, 700, 600]

    if common_object.model == 'ERA5':
        dt = 6  # time between files
        file_location = '/mnt/ERA5/202007/ERA5_PL-20200701_0000.grib'

        fileidx = pygrib.open(file_location)

        # Select the specific variable for U and V components of wind
        grb = fileidx.select(name='U component of wind',
                             typeOfLevel='isobaricInhPa', level=level[0])[0]

        # Extract latitudes and longitudes
        lat_2d_n_s, lon_2d_360 = grb.latlons()
        lat = np.flip(lat_2d_n_s, axis=0)
        lon = np.where(lon_2d_360 > 180, lon_2d_360 - 360, lon_2d_360)

        # switch lat and lon arrays to float32 instead of float64
        lat = np.float32(lat)
        lon = np.float32(lon)
        # make lat and lon arrays C continguous
        lat = np.asarray(lat, order='C')
        lon = np.asarray(lon, order='C')

        # get north, south, east, west indices for tracking
        lat_index_north = (np.abs(lat[:, 0] - north_lat)).argmin()
        lat_index_south = (np.abs(lat[:, 0] - south_lat)).argmin()
        lon_index_west = (np.abs(lon[0, :] - west_lon)).argmin()
        lon_index_east = (np.abs(lon[0, :] - east_lon)).argmin()

        # the total number of degrees in the longitude dimension
        lon_degrees = np.abs(lon[0, 0] - lon[0, -1])

    # set dt in the common_object
    common_object.dt = dt
    # set lat and lon in common_object
    common_object.lat = lat  # switch from float64 to float32
    common_object.lon = lon  # switch from float64 to float32
    common_object.level = lon = np.float32(level)

    # set the lat and lon indices in common_object
    common_object.lat_index_north = lat_index_north
    common_object.lat_index_south = lat_index_south
    common_object.lon_index_east = lon_index_east
    common_object.lon_index_west = lon_index_west
    # set the total number of degrees longitude in common_object
    common_object.total_lon_degrees = lon_degrees
    # print(common_object.total_lon_degrees)
    return


def get_ERA5_variables(common_object, date_time,
                       level=np.array([850, 700, 600])):
    """
    Get the ERA5 variables required for tracking.

    This function takes the common_object that holds lat/lon information,
    the date and time for the desired file. It returns u, v, relative
    vorticity, and curvature vorticity on specific pressure levels.

    Parameters:
        common_object: Object containing common attributes such as lat/lon
            information.
        date_time (datetime.datetime): Date and time for the desired file.
        level (np.array): Pressure levels at which to retrieve data.
            Defaults to np.array([850, 700, 600]).

    Returns:
        tuple: A tuple containing u, v, relative vorticity, and curvature
        vorticity on specific pressure levels.
    """

    # location of ERA5 files
    file_location = '/mnt/ERA5/'
    # open file
    tempofile = f'{file_location}{date_time.strftime("%Y%m")}/'\
        f'ERA5_PL-{date_time.strftime("%Y%m%d_%H")}00.grib'
    print(tempofile)
    fileidx = pygrib.open(tempofile)

    # Select the specific variable for U and V components of wind
    u_grbs = [fileidx.select(name='U component of wind',
                             typeOfLevel='isobaricInhPa',
                             level=level)[0] for level in level]
    v_grbs = [fileidx.select(name='V component of wind',
                             typeOfLevel='isobaricInhPa',
                             level=level)[0] for level in level]

    # Extract latitudes and longitudes
    latd, lond = u_grbs[0].values.shape

    # Initialize arrays for U and V component of wind at each pressure level
    u_levels = np.zeros([len(level), latd, lond])
    v_levels = np.zeros_like(u_levels)

    # Retrieve U and V component of wind at each pressure level
    for i, (u_grb, v_grb) in enumerate(zip(u_grbs, v_grbs)):
        u_levels[i, :, :] = np.flip(u_grb.values, axis=0)
        v_levels[i, :, :] = np.flip(v_grb.values, axis=0)

    # calculate the relative vorticity
    rel_vort_levels = calc_rel_vort(
        u_levels, v_levels, common_object.lat, common_object.lon)
    # calculate the curvature voriticty
    curve_vort_levels = calc_curve_vort_numba(
        common_object.lat, u_levels, v_levels, rel_vort_levels)

    # switch the arrays to be float32 instead of float64
    u_levels = np.float32(u_levels)
    v_levels = np.float32(v_levels)
    rel_vort_levels = np.float32(rel_vort_levels)
    curve_vort_levels = np.float32(curve_vort_levels)

    # make the arrays C contiguous
    # (will need this later for the wrapped C smoothing function)
    u_levels = np.asarray(u_levels, order='C')
    v_levels = np.asarray(v_levels, order='C')
    rel_vort_levels = np.asarray(rel_vort_levels, order='C')
    curve_vort_levels = np.asarray(curve_vort_levels, order='C')

    return u_levels, v_levels, rel_vort_levels, curve_vort_levels


def calc_rel_vort(u, v, lat, lon):
    """
    Calculate relative vorticity.

    Relative vorticity (rel vort) is a measure of the rotation of a fluid
    element with respect to its surroundings. It is defined as the difference
    between the zonal (east-west) derivative of the meridional (north-south)
    wind velocity and the meridional derivative of the zonal wind velocity.

    Parameters:
        u (numpy.ndarray): 	Zonal wind velocity component.
                            Dimensions should be (lev, lat, lon).
        v (numpy.ndarray): 	Meridional wind velocity component.
                            Dimensions should be (lev, lat, lon).
        lat (numpy.ndarray): Latitude values. Dimensions should be (lat, lon).
        lon (numpy.ndarray): Longitude values. Dimensions should be (lat, lon).

    Returns:
        numpy.ndarray: Relative vorticity calculated as dv/dx - du/dy.
        Dimensions will be the same as the input arrays (lev, lat, lon).
    """

    # take derivatives of u and v
    dv_dx = x_derivative(v.copy(), lat, lon)
    du_dy = y_derivative(u.copy(), lat)

    # subtract derivatives to calculate relative vorticity
    rel_vort = dv_dx - du_dy
    return rel_vort


def x_derivative(variable, lat, lon):
    """
    Calculate the derivative of a variable with respect to longitude (x).

    This function computes the derivative of the input variable with
    respect to longitude, representing the rate of change of the variable
    along the east-west direction.
    It is calculated using centralfinite differences.

    Parameters:
        variable (numpy.ndarray): 	Three-dimensional variable
                                    ordered lev, lat, lon.
            It represents the field for which the derivative is computed.
        lat (numpy.ndarray): Latitude values. Dimensions should be (lat, lon).
        lon (numpy.ndarray): Longitude values. Dimensions should be (lat, lon).

    Returns:
        numpy.ndarray: 	The derivative of the variable with respect
                        to longitude (x).
                        It has the same dimensions as the input
                        variable (lev, lat, lon).
    """
    # subtract neighboring longitude points to get delta lon
    # then switch to radians
    dlon = np.radians(lon[0, 2] - lon[0, 1])

    # Calculate the radius of the Earth at the latitude
    radius = 6367500.0 * np.cos(np.radians(lat[:, 0]))

    # Allocate space for the derivative array
    d_dx = np.zeros_like(variable)

    # Loop through latitudes
    for nlat in range(len(lat[:, 0])):
        # Calculate dx for this latitude
        dx = radius[nlat] * dlon
        # Compute the gradient along the longitude axis
        grad = np.gradient(variable[:, nlat, :], dx, axis=1)
        # Store the derivative with respect to longitude
        d_dx[:, nlat, :] = grad

    return d_dx


def y_derivative(variable, lat):
    """
    Calculate the derivative of a variable with respect to latitude (y).

    This function computes the derivative of the input variable with respect
    to latitude, representing the rate of change of the variable along the
    north-south direction. It is calculated using central finite differences.

    Parameters:
        variable (numpy.ndarray):    Three-dimensional variable ordered
                                    lev, lat, lon. It represents the field
                                    for which the derivative is computed.
        lat (numpy.ndarray): Latitude values. Dimensions should be (lat, lon).

    Returns:
        numpy.ndarray: The derivative of the variable with respect to
                        latitude (y). It has the same dimensions as the
                        input variable (lev, lat, lon).
    """
    # subtract neighboring latitude points to get delta lat
    # then switch to radians
    dlat = np.radians(lat[2, 0] - lat[1, 0])

    # calculate dy by multiplying dlat by the radius of the Earth, 6367500 m
    dy = 6367500.0 * dlat

    # calculate the d/dy derivative using the gradient function
    d_dy = np.gradient(variable, dy, axis=1)

    # Return the derivative with respect to latitude
    return d_dy


def calc_curve_vort(common_object, u, v, rel_vort):
    """
    Calculate the curvature vorticity.

    Curvature vorticity is a measure of the vorticity in a fluid flow field
    that arises due to the curvature of the flow lines. It is computed as the
    difference between the relative vorticity and the shear vorticity.

    Parameters:
        common_object (object): An object containing common properties such as
            latitude and longitude. Assumed to have properties
            common_object.lat and common_object.lon.
        u (numpy.ndarray): Zonal wind velocity component.
            Dimensions should be (lev, lat, lon).
        v (numpy.ndarray): Meridional wind velocity component.
            Dimensions should be (lev, lat, lon).
        rel_vort (numpy.ndarray): Relative vorticity.
            Dimensions should be the same as u and v.

    Returns:
        numpy.ndarray: Curvature vorticity calculated as the difference
            between relative vorticity and shear vorticity.
            It has the same dimensions as the input wind components
            (lev, lat, lon).
    """
    # calculate dx and dy
    # subtract neighboring latitude points to get delta lat (dlat),
    # then switch to radians
    dlat = np.radians(np.absolute(
        common_object.lat[2, 0] - common_object.lat[1, 0]))
    # calculate dy by multiplying dlat by the radius of the Earth, 6367500 m
    dy = np.full(
        (common_object.lat.shape[0],
         common_object.lat.shape[1]),
        6367500.0 * dlat)
    # calculate dx by taking the cosine of the lat (in radians) and
    # multiply by dlat times the radius of the Earth, 6367500 m
    dx = np.cos(np.radians(common_object.lat)) * (dlat * 6367500.0)

    # calculate the magnitude of the wind vector sqrt(u^2+v^2), and
    # then make u and v unit vectors
    vec_mag = np.sqrt(np.square(u) + np.square(v))
    u_unit_vec = u / vec_mag
    v_unit_vec = v / vec_mag

    # calculate the shear vorticity
    shear_vort = np.empty_like(u)
    for lon_index in range(u.shape[2]):  # loop through longitudes
        for lat_index in range(u.shape[1]):  # loop through latitudes
            # get the previous and next indices for the current lon_index
            # and lat_index
            lon_index_previous = max(lon_index - 1, 0)
            lon_index_next = min(lon_index + 1, u.shape[2] - 1)
            lat_index_previous = max(lat_index - 1, 0)
            lat_index_next = min(lat_index + 1, u.shape[1] - 1)

            di = 0
            dj = 0
            v1 = ((u_unit_vec[:, lat_index, lon_index]
                   * u[:, lat_index, lon_index_previous])
                  + (v_unit_vec[:, lat_index, lon_index]
                     * v[:, lat_index, lon_index_previous])) \
                * v_unit_vec[:, lat_index, lon_index]
            if lon_index_previous != lon_index:
                di += 1

            v2 = ((u_unit_vec[:, lat_index, lon_index]
                   * u[:, lat_index, lon_index_next])
                  + (v_unit_vec[:, lat_index, lon_index]
                     * v[:, lat_index, lon_index_next]))\
                * v_unit_vec[:, lat_index, lon_index]
            if lon_index_next != lon_index:
                di += 1

            u1 = ((u_unit_vec[:, lat_index, lon_index]
                   * u[:, lat_index_previous, lon_index])
                  + (v_unit_vec[:, lat_index, lon_index]
                     * v[:, lat_index_previous, lon_index]))\
                * u_unit_vec[:, lat_index, lon_index]
            if lat_index_previous != lat_index:
                dj += 1

            u2 = ((u_unit_vec[:, lat_index, lon_index]
                   * u[:, lat_index_next, lon_index])
                  + (v_unit_vec[:, lat_index, lon_index]
                     * v[:, lat_index_next, lon_index]))\
                * u_unit_vec[:, lat_index, lon_index]
            if lat_index_next != lat_index:
                dj += 1

            if di > 0 and dj > 0:
                shear_vort[:, lat_index, lon_index] = (
                    (v2 - v1) / (float(di) * dx[lat_index, lon_index]))\
                    - ((u2 - u1) / (float(dj) * dy[lat_index, lon_index]))

    # calculate curvature vorticity
    curve_vort = rel_vort - shear_vort

    return curve_vort


def calc_curve_vort_numba(lat, u, v, rel_vort):
    """
    Calculate the curvature vorticity.

    Curvature vorticity is a measure of the vorticity in a fluid flow field
    that arises due to the curvature of the flow lines. It is computed as the
    difference between the relative vorticity and the shear vorticity.

    Parameters:
        common_object (object): An object containing common properties such as
            latitude and longitude. Assumed to have properties
            common_object.lat and common_object.lon.
        u (numpy.ndarray): Zonal wind velocity component.
            Dimensions should be (lev, lat, lon).
        v (numpy.ndarray): Meridional wind velocity component.
            Dimensions should be (lev, lat, lon).
        rel_vort (numpy.ndarray): Relative vorticity.
            Dimensions should be the same as u and v.

    Returns:
        numpy.ndarray: Curvature vorticity calculated as the difference
            between relative vorticity and shear vorticity.
            It has the same dimensions as the input wind components
            (lev, lat, lon).
    """

    # Pre-calculate constants outside the jit-decorated function
    dlat = np.radians(np.absolute(
        lat[2, 0] - lat[1, 0]))
    earth_radius = 6367500.0

    @jit
    def calc_shear_vort(u, v, rel_vort):
        dy = np.full((lat.shape[0],
                      lat.shape[1]), earth_radius * dlat)
        dx = np.cos(np.radians(lat)) * (dy)

        vec_mag = np.sqrt(np.square(u) + np.square(v))
        u_unit_vec = u / vec_mag
        v_unit_vec = v / vec_mag

        shear_vort = np.empty_like(u)
        for lon_index in range(u.shape[2]):
            for lat_index in range(u.shape[1]):
                lon_index_previous = max(lon_index - 1, 0)
                lon_index_next = min(lon_index + 1, u.shape[2] - 1)
                lat_index_previous = max(lat_index - 1, 0)
                lat_index_next = min(lat_index + 1, u.shape[1] - 1)

                di = 0
                dj = 0
                v1 = ((u_unit_vec[:, lat_index, lon_index]
                       * u[:, lat_index, lon_index_previous])
                      + (v_unit_vec[:, lat_index, lon_index]
                         * v[:, lat_index, lon_index_previous])) \
                    * v_unit_vec[:, lat_index, lon_index]
                if lon_index_previous != lon_index:
                    di += 1

                v2 = ((u_unit_vec[:, lat_index, lon_index]
                       * u[:, lat_index, lon_index_next])
                      + (v_unit_vec[:, lat_index, lon_index]
                         * v[:, lat_index, lon_index_next]))\
                    * v_unit_vec[:, lat_index, lon_index]
                if lon_index_next != lon_index:
                    di += 1

                u1 = ((u_unit_vec[:, lat_index, lon_index]
                       * u[:, lat_index_previous, lon_index])
                      + (v_unit_vec[:, lat_index, lon_index]
                         * v[:, lat_index_previous, lon_index]))\
                    * u_unit_vec[:, lat_index, lon_index]
                if lat_index_previous != lat_index:
                    dj += 1

                u2 = ((u_unit_vec[:, lat_index, lon_index]
                       * u[:, lat_index_next, lon_index])
                      + (v_unit_vec[:, lat_index, lon_index]
                         * v[:, lat_index_next, lon_index])) \
                    * u_unit_vec[:, lat_index, lon_index]
                if lat_index_next != lat_index:
                    dj += 1

                if di > 0 and dj > 0:
                    shear_vort[:, lat_index, lon_index] = (
                        (v2 - v1) / (float(di) * dx[lat_index, lon_index])) \
                        - ((u2 - u1) / (float(dj) * dy[lat_index, lon_index]))

        return shear_vort

    # Calculate shear vorticity
    shear_vort = calc_shear_vort(u.copy(), v.copy(), rel_vort.copy())

    # Calculate curvature vorticity
    curve_vort = rel_vort - shear_vort

    return curve_vort


def get_variables(common_object, scenario_type, date_time):
    """
    Get wind components, relative vorticity, and curvature vorticity
    from various atmospheric models.

    This function is called from AEW_Tracks.py and initiates the process of
    acquiring zonal wind velocity (u), meridional wind velocity (v),
    relative vorticity, and curvature vorticity at various pressure levels.
    It dispatches model-specific functions based on the model type provided
    in common_object and returns the acquired variables.

    Parameters:
        common_object (object): An object containing common properties such
            as model type.
        scenario_type (str): The type of scenario.
        date_time (datetime.datetime): Date and time for which the
            variables are requested.

    Returns:
        tuple: A tuple containing the wind components (u_levels, v_levels),
            relative vorticity (rel_vort_levels), and curvature vorticity
            (curve_vort_levels). Each element in the tuple is a numpy.ndarray
            with dimensions corresponding to pressure levels and
            spatial coordinates.
    """
    if common_object.model == 'ERA5':
        u_levels, v_levels, rel_vort_levels, curve_vort_levels = \
            get_ERA5_variables(common_object, date_time,
                               level=common_object.level)
    return u_levels, v_levels, rel_vort_levels, curve_vort_levels

# %%


# def get_data_3D(path_file, name, typeOfLevel, region, level):

#     if level == None:
#         str_levels = ['1000', '950', '900', '850', '800', '750',
#                       '700', '650', '600', '550', '500', '450',
#                       '400', '350', '300', '250', '200', '150',
#                       '100']
#         levels = [float(l) for l in str_levels]
#     else:
#         str_levels = [str(level)]
#         levels = [float(l) for l in str_levels]

#     data_r = []
#     for level in levels:
#         fileidx = pygrib.open(path_file)
#         grb = fileidx.select(
#             name=name, typeOfLevel=typeOfLevel, level=level)[0]
#         data = grb.values[::-1, :]

#         lat = grb['distinctLatitudes'][::-1]
#         lon = grb['distinctLongitudes']-360.

#         lat1, lat2 = find_nearest(lat, region['latmin'], region['latmax'])
# [0:2]
#         lon1, lon2 = find_nearest(lon, region['lonmin'], region['lonmax'])[
#             0:2]  # con estos si toca ser mas excata por la resta
#         data = data[lat1:lat2+1, lon1:lon2+1]
#         data_r.append(data)

#     lat = lat[lat1:lat2+1]
#     lon = lon[lon1:lon2+1]
#     data_f = np.zeros([len(levels), len(lat), len(lon)])

#     for i in range(len(levels)):
#         data_f[i, :, :] = data_r[i]

#     return levels, lon, lat, data_f


# %%


# date_time = date_range[0]
# # location of ERA5 files
# file_location = '/mnt/ERA5/'
# # open file
# tempofile = f'{file_location}{date_time.strftime("%Y%m")}/'\
#     f'ERA5_PL-{date_time.strftime("%Y%m%d_%H")}00.grib'
# print(tempofile)


# fileidx = pygrib.open(tempofile)

# # pressure list (hPa)
# levels = [850, 700, 600]

# # Select the specific variable for U and V components of wind
# u_grbs = [fileidx.select(name='U component of wind',
#                          typeOfLevel='isobaricInhPa',
#                          level=level)[0] for level in levels]
# v_grbs = [fileidx.select(name='V component of wind',
#                          typeOfLevel='isobaricInhPa',
#                          level=level)[0] for level in levels]

# # Extract latitudes and longitudes
# lat_2d_n_s, lon_2d_360 = u_grbs[0].latlons()
# latd, lond = u_grbs[0].values.shape
# lat = np.flip(lat_2d_n_s, axis=0)
# lon = np.where(lon_2d_360 > 180, lon_2d_360 - 360, lon_2d_360)
# levels = np.linspace(-42, 42)
# # Plot contour for U component of wind at the first pressure level
# plt.contourf(lon[0, :], lat[:, 0], u_grbs[0].values, levels, cmap='RdBu_r')

# # Initialize arrays for U and V component of wind at each pressure level
# u_levels_360 = np.zeros([len(levels), latd, lond])
# v_levels_360 = np.zeros_like(u_levels_360)

# # Retrieve U and V component of wind at each pressure level
# for i, (u_grb, v_grb) in enumerate(zip(u_grbs, v_grbs)):
#     u_levels_360[i, :, :] = np.flip(u_grb.values, axis=0)
#     v_levels_360[i, :, :] = np.flip(v_grb.values, axis=0)

#     # u_levels_360[i, :, :] = u_grb.values
#     # v_levels_360[i, :, :] = v_grb.values


# # Roll the longitude axis for U and V variables
# # u_levels = np.roll(u_levels_360, int(u_levels_360.shape[2] / 2), axis=2)
# # v_levels = np.roll(v_levels_360, int(v_levels_360.shape[2] / 2), axis=2)

# plt.figure()
# plt.contourf(lon, lat, u_levels_360[0], levels, cmap='RdBu_r')
# # # %%


# # # %%

# # ds_era = xr.load_dataset(
# #     filename_or_obj=tempofile,
# #     engine="cfgrib")

# # ds_era.u.sel(isobaricInhPa=850).plot()

# # var_values = ds_era.u.sel(isobaricInhPa=850).values
# # yy = ds_era.latitude.values
# # xx = ds_era.longitude.values


# # %%

# rel_vort = calc_rel_vort(u=u_levels_360, v=v_levels_360, lat=lat, lon=lon)

# # %%
# curve_vort_2 = calc_curve_vort_2(lat, u=u_levels_360, v=v_levels_360,
#                                  rel_vort=rel_vort)


# # %%
# curve_vort_3 = calc_curve_vort_3(lat, u=u_levels_360, v=v_levels_360,
#                                  rel_vort=rel_vort)
# # %%

# start_time = time.time()
# curve_vort_2 = calc_curve_vort_2(
#     lat, u=u_levels_360, v=v_levels_360, rel_vort=rel_vort)
# end_time = time.time()

# execution_time = end_time - start_time
# print("Execution time for calc_curve_vort_2:", execution_time)

# start_time = time.time()
# curve_vort_3 = calc_curve_vort_3(
#     lat, u=u_levels_360, v=v_levels_360, rel_vort=rel_vort)
# end_time = time.time()

# execution_time = end_time - start_time
# print("Execution time for calc_curve_vort_3:", execution_time)
