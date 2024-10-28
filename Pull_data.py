# %%
from __future__ import division  # makes division not round with integers
import dask.array as da
from dask import delayed, compute
from numba import jit
import pygrib
from netCDF4 import Dataset
import numpy as np
import xarray as xr

'''
Pull_data.py contains functions that are used by AEW_Tracks.py. All of the
functions in Pull_Data.py aid in the pulling of variables and formatting
them to be in a common format so AEW_Tracks.py does not need to know what
model the data came from.
'''


def get_common_track_data(common_object, level=700):
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

    # lat/lon values to crop data to speed up vorticity calculations
    north_lat = 30.
    south_lat = -6.
    west_lon = -90.  # 90
    east_lon = 30.

    if common_object.model == 'ERA5':
        dt = 6  # time between files
        file_location = '/mnt/ERA5/202007/ERA5_PL-20200701_0000.grib'

        fileidx = pygrib.open(file_location)

        # Select the specific variable for U and V components of wind
        grb = fileidx.select(name='U component of wind',
                             typeOfLevel='isobaricInhPa', level=level)[0]

        # Extract latitudes and longitudes
        lat_2d_n_s, lon_2d_360 = grb.latlons()
        lat = np.flip(lat_2d_n_s, axis=0)
        lon = np.where(lon_2d_360 > 180, lon_2d_360 - 360, lon_2d_360)

        # get north, south, east, west indices for tracking
        lat_index_north = (np.abs(lat[:, 0] - north_lat)).argmin()
        lat_index_south = (np.abs(lat[:, 0] - south_lat)).argmin()
        lon_index_west = (np.abs(lon[0, :] - west_lon)).argmin()
        lon_index_east = (np.abs(lon[0, :] - east_lon)).argmin()

        # crop the lat and lon arrays. We don't need the entire global dataset
        lat = lat[lat_index_south:lat_index_north+1,
                  lon_index_west:lon_index_east+1]
        lon = lon[lat_index_south:lat_index_north+1,
                  lon_index_west:lon_index_east+1]

        # switch lat and lon arrays to float32 instead of float64
        lat = np.float32(lat)
        lon = np.float32(lon)
        # make lat and lon arrays C continguous
        lat = np.asarray(lat, order='C')
        lon = np.asarray(lon, order='C')

        # the total number of degrees in the longitude dimension
        lon_degrees = np.abs(lon[0, 0] - lon[0, -1])

        # Compute the minimum and maximum latitude and longitude values
        lat_min = np.min(lat)
        lat_max = np.max(lat)
        lon_min = np.min(lon)
        lon_max = np.max(lon)

        # Calculate the differences between consecutive
        # latitude and longitude points
        lat_diff = np.mean(np.abs(np.diff(lat, axis=0)))
        lon_diff = np.mean(np.abs(np.diff(lon, axis=1)))

        # Average the lat/lon resolutions to get a single resolution value
        res = (lat_diff + lon_diff) / 2

    # set dt in the common_object
    common_object.dt = dt
    # set lat and lon in common_object
    common_object.lat = lat  # switch from float64 to float32
    common_object.lon = lon  # switch from float64 to float32
    common_object.res = res
    common_object.level = np.float32(level)

    # set the lat and lon indices in common_object
    common_object.lat_index_north = lat_index_north
    common_object.lat_index_south = lat_index_south
    common_object.lon_index_east = lon_index_east
    common_object.lon_index_west = lon_index_west
    # set the total number of degrees longitude in common_object
    common_object.total_lon_degrees = lon_degrees

    # set the minimum and maximum lat/lon in common_object
    common_object.lat_min = lat_min
    common_object.lat_max = lat_max
    common_object.lon_min = lon_min
    common_object.lon_max = lon_max
    return


# Function to extract U and V wind components for a given level
# @delayed
def get_uv_for_level(fileidx, level):
    """Retrieve U and V components of wind at a specific level."""
    u_grb = fileidx.select(name='U component of wind',
                           typeOfLevel='isobaricInhPa',
                           level=level)[0]
    v_grb = fileidx.select(name='V component of wind',
                           typeOfLevel='isobaricInhPa',
                           level=level)[0]
    # Extract the data (values) from the GRIB messages
    u_values = np.flip(u_grb.values, axis=0)
    v_values = np.flip(v_grb.values, axis=0)
    return u_values, v_values


def get_variables_par(common_object, date_time,
                      level=700,
                      smooth_field=False):
    """
    Get the ERA5 variables required for tracking.

    This function takes the common_object that holds lat/lon information,
    the date and time for the desired file. It returns u, v, relative
    vorticity, and curvature vorticity on specific pressure level.

    Parameters:
        common_object: Object containing common attributes such as lat/lon
            information.
        date_time (datetime.datetime): Date and time for the desired file.
        level (np.array): Pressure level at which to retrieve data).

    Returns:
        tuple: A tuple containing u, v, relative vorticity, and curvature
        vorticity on specific pressure level.
    """
    from os import path
    # from joblib import Parallel, delayed
    import dask.array as da
    from dask import delayed, compute
    from utilities import get_files_in_folder
    import pandas as pd

    # location of ERA5 files
    file_location = '/mnt/ERA5/'
    # open file
    directory = f'{file_location}{date_time.strftime("%Y%m")}/'
    file_name = f'ERA5_PL-{date_time.strftime("%Y%m%d_%H")}00.grib'

    tempofile = directory + file_name
    # Check if the file exists
    if not path.exists(tempofile):
        try:
            # Get all available files and parse their timestamps
            files = get_files_in_folder(directory, 'ERA5_PL*.grib')
            if len(files) == 0:
                print(f"Error: No files found in directory - {directory}")
                return None, None, None, None

            dates_files = pd.to_datetime([f[-18:-5] for f in files],
                                         format='%Y%m%d_%H%M')
            time_diffs = abs(dates_files - date_time)

            # Find the closest file within a 1-hour threshold
            closest_idx = np.argmin(time_diffs)
            print(closest_idx)

            if time_diffs[closest_idx] <= pd.Timedelta(hours=1):
                tempofile = files[closest_idx]
                print("\tWarning: Using closest available file at "
                      f"{dates_files[closest_idx]}")
            else:
                print("Error: No suitable file within 1-hour range of "
                      f"{date_time}")
                return None, None, None, None

        except Exception as e:
            print(f"Error locating file: {e}")
            return None, None, None, None

    fileidx = pygrib.open(tempofile)

    # Use Dask delayed for parallelization of U and V wind retrieval for each level
    uv_results = get_uv_for_level(fileidx, level)

    # Compute the results in parallel
    # uv_results_computed = compute(*uv_results)

    # Extract latitudes and longitudes from the first level
    latd, lond = uv_results[0].shape

    # Initialize arrays for U and V wind components at each pressure level
    u_levels = np.zeros([latd, lond], dtype=np.float32)
    v_levels = np.zeros_like(u_levels)

    u_levels = uv_results[0]
    v_levels = uv_results[1]

    # crop the u_levels and v_levels arrays.
    # We don't need the entire global dataset
    u_levels = u_levels[common_object.lat_index_south:
                        common_object.lat_index_north+1, :]
    u_levels = u_levels[:, common_object.lon_index_west:
                        common_object.lon_index_east+1]
    v_levels = v_levels[common_object.lat_index_south:
                        common_object.lat_index_north+1, :]
    v_levels = v_levels[:, common_object.lon_index_west:
                        common_object.lon_index_east+1]

    # calculate the relative vorticity
    rel_vort_levels = calc_rel_vort(
        u_levels, v_levels, common_object.lat, common_object.lon)
    # calculate the curvature voriticty
    curve_vort_levels = calc_curve_vort_numba(
        common_object.lat, u_levels, v_levels, rel_vort_levels)

    if smooth_field:
        curve_vort_levels = radially_averaged_2d(common_object.lon,
                                                 common_object.lat,
                                                 curve_vort_levels,
                                                 res=common_object.res,
                                                 radius=600)
        rel_vort_levels = radially_averaged_2d(common_object.lon,
                                               common_object.lat,
                                               rel_vort_levels,
                                               res=common_object.res,
                                               radius=600)

    if curve_vort_levels is not None and rel_vort_levels is not None:

        rel_vort_levels = np.float32(rel_vort_levels)
        curve_vort_levels = np.float32(curve_vort_levels)

    return u_levels, v_levels, rel_vort_levels, curve_vort_levels


def get_ERA5_variables(common_object, date_time,
                       level=np.array([850, 700, 600]),
                       smooth_field=False):
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
    from os import path
    # location of ERA5 files
    file_location = '/mnt/ERA5/'
    # open file
    tempofile = f'{file_location}{date_time.strftime("%Y%m")}/'\
        f'ERA5_PL-{date_time.strftime("%Y%m%d_%H")}00.grib'

    # Check if the file exists
    if not path.exists(tempofile):
        raise OSError(f"File not found: {tempofile}")

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

    # crop the u_levels and v_levels arrays.
    # We don't need the entire global dataset
    u_levels = u_levels[:, common_object.lat_index_south:
                        common_object.lat_index_north+1, :]
    u_levels = u_levels[:, :, common_object.lon_index_west:
                        common_object.lon_index_east+1]
    v_levels = v_levels[:, common_object.lat_index_south:
                        common_object.lat_index_north+1, :]
    v_levels = v_levels[:, :, common_object.lon_index_west:
                        common_object.lon_index_east+1]

    # switch the arrays to be float32 instead of float64
    u_levels = np.float32(u_levels)
    v_levels = np.float32(v_levels)
    # make the arrays C contiguous
    # (will need this later for the wrapped C smoothing function)
    # u_levels = np.asarray(u_levels, order='C')
    # v_levels = np.asarray(v_levels, order='C')

    # calculate the relative vorticity
    rel_vort_levels = calc_rel_vort(
        u_levels, v_levels, common_object.lat, common_object.lon)
    # calculate the curvature voriticty
    curve_vort_levels = calc_curve_vort_numba(
        common_object.lat, u_levels, v_levels, rel_vort_levels)

    if smooth_field:
        curve_vort_levels = radially_averaged_2d(common_object.lon,
                                                 common_object.lat,
                                                 curve_vort_levels,
                                                 res=common_object.res,
                                                 radius=600)
        rel_vort_levels = radially_averaged_2d(common_object.lon,
                                               common_object.lat,
                                               rel_vort_levels,
                                               res=common_object.res,
                                               radius=600)

    if curve_vort_levels is not None and rel_vort_levels is not None:

        rel_vort_levels = np.float32(rel_vort_levels)
        curve_vort_levels = np.float32(curve_vort_levels)

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
        grad = np.gradient(variable[nlat, :], dx, axis=0)
        # Store the derivative with respect to longitude
        d_dx[nlat, :] = grad

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
    d_dy = np.gradient(variable, dy, axis=0)

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


def radially_averaged_3d(lon, lat, data_file, res, radius=600, n_jobs=-1):
    from joblib import Parallel, delayed
    import numpy as np

    def get_dist_meters(lon, lat):
        earth_circ = 6371 * 2 * np.pi * 1000  # Earth's circumference in meters
        lat_met = earth_circ / 360  # meters per degree latitude
        lat_dist = np.gradient(lat, axis=0) * lat_met
        lon_dist = np.gradient(lon, axis=1) * np.cos(np.deg2rad(lat)) * lat_met
        return lon_dist, lat_dist

    def GetBG(lon, lat, data_file, res, radius=600):
        """
        Computes a background grid averaged over a specified radius
        for each grid point.
        """
        def rad_mask(i, j, dx, dy, radius):
            """Generate a radial mask for averaging within the
            given radius around (i, j)."""
            # Earth's circumference in meters
            earth_circ = 6371 * 2 * np.pi * 1000
            lat_met = earth_circ / 360  # meters per degree latitude
            lat_km_lat = lat_met / 1000
            lat_km_lon = lat_met / 1000 / 2  # Adjusted for latitude effects

            # Define buffer based on radius
            buffer = int(np.ceil(radius / lat_km_lat / res))
            buffer_j = int(np.ceil(radius / lat_km_lon / res))

            boolean_array = np.zeros(np.shape(dx), dtype=bool)

            # Slice out a box region to limit computation
            i_st = max(i - (buffer + 1), 0)
            i_end = min(i + (buffer + 1), dx.shape[0])
            j_st = max(j - (buffer_j + 1), 0)
            j_end = min(j + (buffer_j + 1), dx.shape[1])

            new_i = i - i_st
            new_j = j - j_st

            dy_slc = dy[i_st:i_end, j_st:j_end]
            dx_slc = dx[i_st:i_end, j_st:j_end]

            i_array_sub = np.zeros(np.shape(dy_slc))
            j_array_sub = np.zeros(np.shape(dx_slc))

            # Accumulate distances from the center point (i, j)
            i_array_sub[new_i, :] = 0
            i_array_sub[(new_i + 1):,
                        :] = np.add.accumulate(dy_slc[(new_i + 1):, :])
            i_array_sub[(new_i - 1)::-1,
                        :] = np.add.accumulate(dy_slc[(new_i - 1)::-1, :])

            j_array_sub[:, new_j] = 0
            j_array_sub[:, (new_j + 1):] = \
                np.add.accumulate(dx_slc[:, (new_j + 1):], axis=1)
            j_array_sub[:, (new_j - 1)::-1] = \
                np.add.accumulate(dx_slc[:, (new_j - 1)::-1], axis=1)

            radial_array = (np.sqrt(np.square(i_array_sub) +
                            np.square(j_array_sub)) / 1000) < radius
            boolean_array[i_st:i_end, j_st:j_end] = radial_array

            return boolean_array

        # Calculate distance arrays
        dx, dy = get_dist_meters(lon, lat)

        # Define bounds
        i_bounds = np.arange(int(np.ceil(
            radius / (6371 * 2 * np.pi * 1000 / 360) / res)) + 1,
            np.shape(data_file)[0])
        j_bounds = np.arange(int(np.ceil(
            radius / (6371 * 2 * np.pi * 1000 / 360 / 2) / res)) + 1,
            np.shape(data_file)[1])

        # Output grid for averaged values
        avg_grid = np.zeros(np.shape(data_file))

        # Iterate over grid points
        for y in i_bounds:
            for x in j_bounds:
                out_mask = rad_mask(y, x, dx, dy, radius)
                avg_grid[y, x] = np.mean(data_file[out_mask])
        return avg_grid

    def run_loop(slc_num, radius):
        curv_array = GetBG(lon, lat, data_file[slc_num, :, :], res, radius)
        return curv_array

    results = Parallel(n_jobs=n_jobs)(delayed(run_loop)(i, radius)
                                      for i in np.arange(data_file.shape[0]))

    return np.array(results)


def radially_averaged_2d(lon, lat, data_file, res, radius=600, n_jobs=-1):
    # from joblib import Parallel, delayed
    import numpy as np

    def get_dist_meters(lon, lat):
        earth_circ = 6371 * 2 * np.pi * 1000  # Earth's circumference in meters
        lat_met = earth_circ / 360  # meters per degree latitude
        lat_dist = np.gradient(lat, axis=0) * lat_met
        lon_dist = np.gradient(lon, axis=1) * np.cos(np.deg2rad(lat)) * lat_met
        return lon_dist, lat_dist

    def get_background_grid(lon, lat, data_file, res, radius=600):
        """
        Computes a background grid averaged over a specified radius
        for each grid point.
        """
        def rad_mask(i, j, dx, dy, radius):
            """Generate a radial mask for averaging within the
            given radius around (i, j)."""
            # Earth's circumference in meters
            earth_circ = 6371 * 2 * np.pi * 1000
            lat_met = earth_circ / 360  # meters per degree latitude
            lat_km_lat = lat_met / 1000
            lat_km_lon = lat_met / 1000 / 2  # Adjusted for latitude effects

            # Define buffer based on radius
            buffer = int(np.ceil(radius / lat_km_lat / res))
            buffer_j = int(np.ceil(radius / lat_km_lon / res))

            boolean_array = np.zeros(np.shape(dx), dtype=bool)

            # Slice out a box region to limit computation
            i_st = max(i - (buffer + 1), 0)
            i_end = min(i + (buffer + 1), dx.shape[0])
            j_st = max(j - (buffer_j + 1), 0)
            j_end = min(j + (buffer_j + 1), dx.shape[1])

            new_i = i - i_st
            new_j = j - j_st

            dy_slc = dy[i_st:i_end, j_st:j_end]
            dx_slc = dx[i_st:i_end, j_st:j_end]

            i_array_sub = np.zeros(np.shape(dy_slc))
            j_array_sub = np.zeros(np.shape(dx_slc))

            # Accumulate distances from the center point (i, j)
            i_array_sub[new_i, :] = 0
            i_array_sub[(new_i + 1):,
                        :] = np.add.accumulate(dy_slc[(new_i + 1):, :])
            i_array_sub[(new_i - 1)::-1,
                        :] = np.add.accumulate(dy_slc[(new_i - 1)::-1, :])

            j_array_sub[:, new_j] = 0
            j_array_sub[:, (new_j + 1):] = \
                np.add.accumulate(dx_slc[:, (new_j + 1):], axis=1)
            j_array_sub[:, (new_j - 1)::-1] = \
                np.add.accumulate(dx_slc[:, (new_j - 1)::-1], axis=1)

            radial_array = (np.sqrt(np.square(i_array_sub) +
                            np.square(j_array_sub)) / 1000) < radius
            boolean_array[i_st:i_end, j_st:j_end] = radial_array

            return boolean_array

        # Calculate distance arrays
        dx, dy = get_dist_meters(lon, lat)

        # Define bounds
        i_bounds = np.arange(int(np.ceil(
            radius / (6371 * 2 * np.pi * 1000 / 360) / res)) + 1,
            np.shape(data_file)[0])
        j_bounds = np.arange(int(np.ceil(
            radius / (6371 * 2 * np.pi * 1000 / 360 / 2) / res)) + 1,
            np.shape(data_file)[1])

        # Output grid for averaged values
        avg_grid = np.zeros(np.shape(data_file))

        # Iterate over grid points
        for y in i_bounds:
            for x in j_bounds:
                out_mask = rad_mask(y, x, dx, dy, radius)
                avg_grid[y, x] = np.mean(data_file[out_mask])
        return avg_grid

    # Since data_file is 2D, process it directly
    result = get_background_grid(lon, lat, data_file, res, radius)

    return np.array(result)


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
        for lon_index in range(u.shape[1]):
            for lat_index in range(u.shape[0]):
                lon_index_previous = max(lon_index - 1, 0)
                lon_index_next = min(lon_index + 1, u.shape[1] - 1)
                lat_index_previous = max(lat_index - 1, 0)
                lat_index_next = min(lat_index + 1, u.shape[0] - 1)

                di = 0
                dj = 0
                v1 = ((u_unit_vec[lat_index, lon_index]
                       * u[lat_index, lon_index_previous])
                      + (v_unit_vec[lat_index, lon_index]
                         * v[lat_index, lon_index_previous])) \
                    * v_unit_vec[lat_index, lon_index]
                if lon_index_previous != lon_index:
                    di += 1

                v2 = ((u_unit_vec[lat_index, lon_index]
                       * u[lat_index, lon_index_next])
                      + (v_unit_vec[lat_index, lon_index]
                         * v[lat_index, lon_index_next]))\
                    * v_unit_vec[lat_index, lon_index]
                if lon_index_next != lon_index:
                    di += 1

                u1 = ((u_unit_vec[lat_index, lon_index]
                       * u[lat_index_previous, lon_index])
                      + (v_unit_vec[lat_index, lon_index]
                         * v[lat_index_previous, lon_index]))\
                    * u_unit_vec[lat_index, lon_index]
                if lat_index_previous != lat_index:
                    dj += 1

                u2 = ((u_unit_vec[lat_index, lon_index]
                       * u[lat_index_next, lon_index])
                      + (v_unit_vec[lat_index, lon_index]
                         * v[lat_index_next, lon_index])) \
                    * u_unit_vec[lat_index, lon_index]
                if lat_index_next != lat_index:
                    dj += 1

                if di > 0 and dj > 0:
                    shear_vort[lat_index, lon_index] = (
                        (v2 - v1) / (float(di) * dx[lat_index, lon_index])) \
                        - ((u2 - u1) / (float(dj) * dy[lat_index, lon_index]))

        return shear_vort

    # Calculate shear vorticity
    shear_vort = calc_shear_vort(u.copy(), v.copy(), rel_vort.copy())

    # Calculate curvature vorticity
    curve_vort = rel_vort - shear_vort

    return curve_vort


def get_variables(common_object, date_time, smooth_field=False):
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
    try:
        if common_object.model == 'ERA5':
            u_levels, v_levels, rel_vort_levels, curve_vort_levels = \
                get_ERA5_variables(common_object, date_time,
                                   level=common_object.level,
                                   smooth_field=smooth_field)
        return u_levels, v_levels, rel_vort_levels, curve_vort_levels

    except OSError as e:
        # Handle the case when the file doesn't exist or can't be opened
        print(f"Error: {e}")
        # Optionally, log the error to a file or take other actions

        # You can return None or other default values if the file doesn't exist
        return None, None, None, None


def COMPUTE_CURV_VORT_NON_DIV_UPDATE(data_in, data_out, res=1, radius=600,
                                     njobs=1, nondiv=False, SAVE_IMAGE=False,
                                     SAVE_OUTPUT=True):
    # IMPORT STATEMENTS
    import datetime
    import os
    import time as tm

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import imageio
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import xarray as xr
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    from joblib import Parallel, delayed
    from joblib.externals.loky import set_loky_pickler
    from netCDF4 import Dataset, num2date
    from numpy import dtype
    set_loky_pickler("dill")

    import warnings

    warnings.simplefilter("ignore", UserWarning)

    dir_ani_frame = 'frames_R'+str(radius)

    # ### IMPORTANT DIRECTORIES AND CUSTOMIZATIONS
    gif_dir = 'CURV_VORT/GIF/'
    data_dir = 'CURV_VORT/TEMP_DATA/'
    data_in = data_in
    data_out = data_out

    print('Setting Up | Output to:'+data_out)
    try:
        os.mkdir(gif_dir+dir_ani_frame)
    except:
        print('Directory could not be created -- may already exist')

    save_dir = gif_dir+dir_ani_frame+'/'

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        # print(idx)
        return idx, array[idx]

    def get_dist_meters(lon, lat):
        earth_circ = 6371*2*np.pi*1000  # earth's circumference in meters
        # get the number of meters in a degree latitude (ignoring "bulge")
        lat_met = earth_circ/360
        lat_dist = np.gradient(lat, axis=0)*lat_met
        lon_dist = np.gradient(lon, axis=1)*np.cos(np.deg2rad(lat))*lat_met
        return lon_dist, lat_dist

    def curv_vort(u, v, dx, dy):
        V_2 = (u**2+v**2)
        curv_vort_raw = (1/V_2)*(u*u*np.gradient(v, axis=1)/dx
                                 - v*v*np.gradient(u, axis=0)/dy
                                 - v*u*np.gradient(u, axis=1)/dx
                                 + u*v*np.gradient(v, axis=0)/dy)
        return curv_vort_raw

    def GetBG(lon, lat, data_file, res, radius):
        """
        Computes a background grid averaged over a specified radius
        for each grid point.

        Parameters:
        - lon: 2D array of longitudes
        - lat: 2D array of latitudes
        - data_file: 2D array of data values to be averaged
        - res: resolution of the grid (in degrees)
        - radius: radius (in kilometers) to average over

        Returns:
        - avg_grid: 2D array of averaged values
        """

        import time
        import numpy as np

        start_time = time.time()

        def get_dist_meters(lon, lat):
            """Calculate the distance in meters between grid points based on lat/lon."""
            earth_circ = 6371 * 2 * np.pi * 1000  # Earth's circumference in meters
            lat_met = earth_circ / 360  # Meters per degree latitude
            lat_dist = np.gradient(lat, axis=0) * lat_met
            lon_dist = np.gradient(lon, axis=1) * \
                np.cos(np.deg2rad(lat)) * lat_met
            return lon_dist, lat_dist

        # Step 1: Calculate distance arrays (dx and dy) for longitude and latitude
        dx, dy = get_dist_meters(lon, lat)

        # Step 2: Define the buffer size based on the specified radius
        earth_circ = 6371 * 2 * np.pi * 1000  # Earth's circumference in meters
        lat_met = earth_circ / 360  # Meters per degree latitude
        lat_km_lat = lat_met / 1000
        lat_km_lon = lat_met / 1000 / 2  # Adjusted for latitude effects

        buffer = int(np.ceil(radius / lat_km_lat / res))
        buffer_lon = int(np.ceil(radius / lat_km_lon / res))

        # Step 3: Define bounds for iteration, considering the buffer
        i_bounds = np.arange(buffer + 1, np.shape(data_file)[0] - buffer)
        j_bounds = np.arange(
            buffer_lon + 1, np.shape(data_file)[1] - buffer_lon)

        # Step 4: Initialize the output grid for averaged values
        avg_grid = np.zeros(np.shape(data_file))

        def rad_mask(i, j, dx, dy, radius):
            """Generate a radial mask for averaging within the given radius around (i, j)."""
            earth_circ = 6371 * 2 * np.pi * 1000
            lat_met = earth_circ / 360
            lat_km_lat = lat_met / 1000
            lat_km_lon = lat_met / 1000 / 2
            res = 1

            buffer = int(np.ceil(radius / lat_km_lat / res))
            buffer_j = int(np.ceil(radius / lat_km_lon / res))

            boolean_array = np.zeros(np.shape(dx), dtype=bool)

            # Slice out a box region to limit computation based on max distance
            i_st = i - (buffer + 1)
            i_end = i + (buffer + 1)
            j_st = j - (buffer_j + 1)
            j_end = j + (buffer_j + 1)

            new_i = i - i_st
            new_j = j - j_st

            dy_slc = dy[i_st:i_end, j_st:j_end]
            dx_slc = dx[i_st:i_end, j_st:j_end]

            i_array_sub = np.zeros(np.shape(dy_slc))
            j_array_sub = np.zeros(np.shape(dx_slc))

            # Accumulate distances from the center point (i, j)
            i_array_sub[new_i, :] = 0
            i_array_sub[(new_i + 1):,
                        :] = np.add.accumulate(dy_slc[(new_i + 1):, :])
            i_array_sub[(new_i - 1)::-1,
                        :] = np.add.accumulate(dy_slc[(new_i - 1)::-1, :])

            j_array_sub[:, new_j] = 0
            j_array_sub[:, (new_j + 1):] = \
                np.add.accumulate(dx_slc[:, (new_j + 1):], axis=1)
            j_array_sub[:, (new_j - 1)::-1] = \
                np.add.accumulate(dx_slc[:, (new_j - 1)::-1], axis=1)

            # Create radial mask where distances are less than the radius
            radial_array = (np.sqrt(np.square(i_array_sub) +
                            np.square(j_array_sub)) / 1000) < radius
            boolean_array[i_st:i_end, j_st:j_end] = radial_array

            return boolean_array

        # Step 5: Iterate over all points and compute the average within the radius
        for y in i_bounds:
            for x in j_bounds:
                out_mask = rad_mask(y, x, dx, dy, radius)
                avg_grid[y, x] = np.mean(data_file[out_mask])

        end_time = time.time()
        print(f'Time elapsed: {end_time - start_time:.2f} seconds')

        return avg_grid

    ### ----------------------- ACTUAL COMPUTATIONAL BLOCK --------------------------- ###
    # First, import the data
    print('Starting Computation of Radial Averaged CV...')
    nc_file = xr.open_dataset(data_in)
    nc_file_alt = Dataset(data_in, 'r')

    time = nc_file_alt.variables['time'][:]
    time_units = nc_file_alt.variables['time'].units
    nclat = nc_file['latitude'].values
    nclon = nc_file['longitude'].values

    # Find, slice out only the area we are interested in (to reduce file size and prevent memory overuse/dumps!)
    lat = nclat
    lon = nclon

    if nondiv == False:
        u_wnd = nc_file['u'].values
        v_wnd = nc_file['v'].values
    else:
        u_wnd = nc_file['upsi'].values
        v_wnd = nc_file['vpsi'].values

    LON, LAT = np.meshgrid(lon, lat)
    dx, dy = get_dist_meters(LON, LAT)

    out_array = np.zeros(
        (np.shape(time)[0], np.shape(LAT)[0], np.shape(LAT)[1]))

    def run_loop(slc_num, radius):
        # file_list = []
        print('Timestep number: '+str(slc_num))
        out_name = data_dir + 'curv_temp_data_'+str(slc_num)+'.npy'
        curv_vort_data = curv_vort(
            u_wnd[slc_num, :, :], v_wnd[slc_num, :, :], dx, dy)
        set_radius = radius
        curv_array = GetBG(LON, LAT, curv_vort_data, res, set_radius)

        if SAVE_OUTPUT == True:
            np.save(out_name, curv_array)

    start = tm.time()
    Parallel(n_jobs=njobs)(delayed(run_loop)(i, radius)
                           for i in np.arange(len(time)))
    end = tm.time()
    print('Time to run computation: ' + str(end - start))

    if SAVE_IMAGE == True:
        file_list = []
        for i in np.arange(len(time)):
            temp_file = save_dir+'curv_vort_helmholz_R' + \
                str(radius)+'_'+str(i)+'.png'
            file_list.append(temp_file)

        with imageio.get_writer(gif_dir+'curv_vort_helmholtz_ani_R'+str(radius)+'.gif', mode='I') as writer:
            for filename in file_list:
                image = imageio.imread(filename)
                writer.append_data(image)

    if SAVE_OUTPUT == True:
        for i in np.arange(len(time)):
            in_file = data_dir + 'curv_temp_data_'+str(i)+'.npy'
            temp_file = np.load(in_file)
            out_array[i, :, :] = temp_file[:, :]
            os.remove(in_file)
        # Finally, save the np file
        nlat = np.size(lat)
        nlon = np.size(lon)
        # open a netCDF file to write
        ncout = Dataset(data_out, 'w', format='NETCDF4')

        # define axis size
        ncout.createDimension('time', None)  # unlimited
        ncout.createDimension('latitude', nlat)
        ncout.createDimension('longitude', nlon)

        # create time axis
        time_data = ncout.createVariable(
            'time', dtype('float64').char, ('time',))
        time_data.long_name = 'time'
        time_data.units = time_units
        time_data.calendar = 'gregorian'
        time_data.axis = 'T'

        # create latitude axis
        latitude = ncout.createVariable(
            'latitude', dtype('float64').char, ('latitude'))
        latitude.standard_name = 'latitude'
        latitude.long_name = 'latitude'
        latitude.units = 'degrees_north'
        latitude.axis = 'Y'

        # create longitude axis
        longitude = ncout.createVariable(
            'longitude', dtype('float64').char, ('longitude'))
        longitude.standard_name = 'longitude'
        longitude.long_name = 'longitude'
        longitude.units = 'degrees_east'
        longitude.axis = 'X'

        # create variable array
        curvout = ncout.createVariable('curv_vort', dtype(
            'float64').char, ('time', 'latitude', 'longitude'))
        curvout.long_name = 'Radially Averaged Curvature Vorticity'
        curvout.units = 's**-1'

        # copy axis from original dataset
        time_data[:] = time[:]
        longitude[:] = lon[:]
        latitude[:] = lat[:]
        curvout[:, :, :] = out_array[:, :, :]
        ncout.close()
