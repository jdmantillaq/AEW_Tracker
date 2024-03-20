# %%
from __future__ import division  # makes division not round with integers
from numba import jit
import pygrib
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import wrf as wrf

'''
Pull_data_2.py contains functions that are used by AEW_Tracks.py.
All of the functions in the pulling of variables and formatting
them to be in a common format so AEW_Tracks.py does not need to know what
model the data came from. This contains the original sources considered
in the repository
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

    if common_object.model == 'WRF':
        dt = 6  # time between files
        # get the latitude and longitude and the north, south, east, and west
        # indices of a rectangle over Africa and the Atlantic
        file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/Historical/'\
            'wrfout_d01_2008-07-01_00_00_00'
        data = Dataset(file_location)
        # get lat and lon values
        # get the latitude and longitude at a single time
        # (since they don't change with time)
        lat = wrf.getvar(data, "lat", meta=False)  # ordered lat, lon
        lon = wrf.getvar(data, "lon", meta=False)  # ordered lat, lon

        # get north, south, east, west indices
        lon_index_west, lat_index_south = wrf.ll_to_xy(
            data, south_lat, west_lon, meta=False)
        lon_index_east, lat_index_north = wrf.ll_to_xy(
            data, north_lat, east_lon, meta=False)
        # lat_crop = lat.values[lat_index_south:lat_index_north+1,
        # lon_index_west:lon_index_east+1]
        # lon_crop = lon.values[lat_index_south:lat_index_north+1,
        # lon_index_west:lon_index_east+1]

        # the following two lines are to correct for the weird negative
        # indexing that comes back from the wrf.ll_to_xy function
        lon_index_west = lon.shape[1] + lon_index_west
        lon_index_east = lon.shape[1] + lon_index_east

        # the total number of degrees in the longitude dimension
        lon_degrees = 360.

    elif common_object.model == 'MERRA2':
        dt = 3  # time between files
        # get the latitude and longitude and the north, south, east,
        # and west indices of a rectangle over Africa and the Atlantic
        file_location = \
            '/global/cscratch1/sd/ebercosh/MERRA2/U1000_20170701.nc'
        data = xr.open_dataset(file_location)

        # get lat and lon values
        # get the latitude and longitude at a single time
        # (since they don't change with time)
        lat_1d = data.lat.values  # ordered lat
        lon_1d = data.lon.values  # ordered lon

        # make the lat and lon arrays from the GCM 2D (ordered lat, lon)
        lon = np.tile(lon_1d, (lat_1d.shape[0], 1))
        lat_2d = np.tile(lat_1d, (len(lon_1d), 1))
        lat = np.rot90(lat_2d, 3)
        # switch lat and lon arrays to float32 instead of float64
        lat = np.float32(lat)
        lon = np.float32(lon)
        # make lat and lon arrays C continguous
        lat = np.asarray(lat, order='C')
        lon = np.asarray(lon, order='C')

        # get north, south, east, west indices
        lat_index_north = (np.abs(lat_1d - north_lat)).argmin()
        lat_index_south = (np.abs(lat_1d - south_lat)).argmin()
        lon_index_west = (np.abs(lon_1d - west_lon)).argmin()
        lon_index_east = (np.abs(lon_1d - east_lon)).argmin()

    elif common_object.model == 'CAM5':
        # dt = 3 # time between files
        dt = 6  # time between files to compare with WRF
        # get the latitude and longitude and the north, south, east, and west
        # indices of a rectangle over Africa and the Atlantic
        file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/'\
            'Historical/run3/2006/U_CAM5-1-0.25degree_All-Hist_est1_v3_run3.'\
            'cam.h4.2006-07-01-00000_AEW.nc'
        data = xr.open_dataset(file_location, decode_times=False)

        # get lat and lon values
        # get the latitude and longitude at a single time
        # (since they don't change with time)
        lat_1d = data.lat.values  # ordered lat
        lon_1d = data.lon.values  # ordered lon
        # make the lat and lon arrays from the GCM 2D (ordered lat, lon)
        lon = np.tile(lon_1d, (lat_1d.shape[0], 1))
        lat_2d = np.tile(lat_1d, (len(lon_1d), 1))
        lat = np.rot90(lat_2d, 3)
        # switch lat and lon arrays to float32 instead of float64
        lat = np.float32(lat)
        lon = np.float32(lon)
        # make lat and lon arrays C continguous
        lat = np.asarray(lat, order='C')
        lon = np.asarray(lon, order='C')

        # get north, south, east, west indices
        lat_index_north = (np.abs(lat_1d - north_lat)).argmin()
        lat_index_south = (np.abs(lat_1d - south_lat)).argmin()
        lon_index_west = (np.abs(lon_1d - west_lon)).argmin()
        lon_index_east = (np.abs(lon_1d - east_lon)).argmin()

        # the total number of degrees in the longitude dimension
        lon_degrees = np.abs(lon[0, 0] - lon[0, -1])

    elif common_object.model == 'ERA5':
        # dt for ERA5 is 1 hour (data is hourly),
        # but set dt to whatever the dt is for the dataset to be compared with
        # Eg dt=3 to compare with CAM5 or dt=6 to compare with WRF
        dt = 6  # time between files
        file_location = '/mnt/ERA5/202007/ERA5_PL-20200701_0000.grib'

        fileidx = pygrib.open(file_location)

        # Select the specific variable for U and V components of wind
        grb = fileidx.select(name='U component of wind',
                             typeOfLevel='isobaricInhPa', level=850)[0]

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

    elif common_object.model == 'ERAI':
        dt = 6  # time between files
        file_location = '/global/cscratch1/sd/ebercosh/Reanalysis/ERA-I/'\
            'ei.oper.an.pl.regn128uv.2010113000'
        grbs = pygrib.open(file_location)
        grb = grbs.select(name='U component of wind')[23]
        # lat and lon are 2D, ordered lat, lon
        # the lat goes from north to south (so 90, 89, 88, .....-88, -89, -90),
        # and lon goes from 0-360 degrees
        lat_2d_n_s, lon_2d_360 = grb.latlons()
        # make the lat array go from south to north
        lat = np.flip(lat_2d_n_s, axis=0)
        # make the longitude go from -180 to 180 degrees
        lon = lon_2d_360 - 180.

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

    # set dt in the common_object
    common_object.dt = dt
    # set lat and lon in common_object
    common_object.lat = lat  # switch from float64 to float32
    common_object.lon = lon  # switch from float64 to float32
    # set the lat and lon indices in common_object
    common_object.lat_index_north = lat_index_north
    common_object.lat_index_south = lat_index_south
    common_object.lon_index_east = lon_index_east
    common_object.lon_index_west = lon_index_west
    # set the total number of degrees longitude in common_object
    common_object.total_lon_degrees = lon_degrees
    print(common_object.total_lon_degrees)
    return


def get_WRF_variables(common_object, scenario_type, date_time):
    """
    Retrieve WRF variables required for tracking atmospheric features.

    This function takes the common_object containing latitude and
    longitude information, the scenario_type, and the date and time
    for the desired file. It opens the WRF file corresponding to the
    specified scenario type and date_time and retrieves u, v, relative
    vorticity, and curvature vorticity on specific pressure levels.

    Parameters:
        common_object (object): An object containing common properties
            such as lat/lon information.
        scenario_type (str): Type of scenario, e.g., Historical, Forecast, etc.
        date_time (datetime): Date and time for the desired WRF file.

    Returns:
        tuple: A tuple containing four arrays representing u, v,
            relative vorticity, and
        curvature vorticity on specific pressure levels.
    """

    # location of WRF file
    file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/' + \
        scenario_type + '/' + date_time.strftime('%Y') + '/wrfout_d01_crop_'
    # open file
    data = Dataset(file_location +
                   date_time.strftime("%Y-%m-%d_%H_%M_%S") + '.nc')
    # get u, v, and p
    print("Pulling variables...")
    p_3d = wrf.getvar(data, 'pressure')  # pressure in hPa
    u_3d = wrf.getvar(data, 'ua')  # zonal wind in m/s
    v_3d = wrf.getvar(data, 'va')  # meridional wind in m/s

    # get u and v at the pressure levels 850, 700, and 600 hPa
    u_levels = calc_var_pres_levels(p_3d, u_3d)
    v_levels = calc_var_pres_levels(p_3d, v_3d)

    # calculate the relative vorticity
    rel_vort_levels = calc_rel_vort(
        u_levels.values, v_levels.values, common_object.lat, common_object.lon)
    # calculate the curvature vorticity
    curve_vort_levels = calc_curve_vort(
        common_object, u_levels.values, v_levels.values, rel_vort_levels)

    return u_levels.values, v_levels.values, rel_vort_levels, curve_vort_levels


def calc_var_pres_levels(p, var):
    """
    Interpolate WRF variables to specific pressure levels.

    This function takes the pressure (p) and the variable (var)
    to be interpolated. It interpolates the variable to specified pressure
    levels and returns a three-dimensional array ordered
    as lev (pressure), lat, lon.

    Parameters:
        p (xarray.DataArray): Pressure levels.
        var (xarray.DataArray): Variable to be interpolated.

    Returns:
        xarray.DataArray: A three-dimensional array representing the
            interpolated variable
        at specific pressure levels, ordered lev (pressure), lat, lon.
    """
    # Pressure levels needed
    pressure_levels = [850., 700., 600.]
    # Interpolate the variable to the above pressure levels
    # Returns an array with the lev dimension the length of pressure_levels
    var_levels = wrf.interplevel(var, p, pressure_levels)
    # Get rid of any NaNs
    # Linearly interpolate the missing values
    mask = np.isnan(var_levels.values)
    var_levels.values[mask] = np.interp(np.flatnonzero(
        mask), np.flatnonzero(~mask), var_levels.values[~mask])

    return var_levels


def get_MERRA2_variables(common_object, date_time):
    """
    Get the MERRA2 variables required for tracking.

    This function takes the common_object that holds lat/lon information,
    the scenario type, and the date and time for the desired file.
    It returns u, v, relative vorticity, and curvature
    vorticity on specific pressure levels.

    Parameters:
        common_object: Object containing common attributes such as
            lat/lon information.
        date_time (datetime.datetime): Date and time for the desired file.

    Returns:
        tuple: A tuple containing u, v, relative vorticity, and
            curvature vorticity on specific pressure levels.
    """
    # Location of MERRA-2 files
    u_file_location = '/global/cscratch1/sd/ebercosh/MERRA2/U1000_'
    v_file_location = '/global/cscratch1/sd/ebercosh/MERRA2/V1000_'
    # Open files
    u_data = xr.open_dataset(
        u_file_location + date_time.strftime("%Y%m%d") + '.nc')
    v_data = xr.open_dataset(
        v_file_location + date_time.strftime("%Y%m%d") + '.nc')

    # Get u and v
    print("Pulling variables...")
    time_dict = {'00': 0, '03': 1, '06': 2,
                 '09': 3, '12': 4, '15': 5, '18': 6, '21': 7}
    u_3d = u_data.U[time_dict[date_time.strftime("%H")], :, :, :]
    v_3d = v_data.V[time_dict[date_time.strftime("%H")], :, :, :]

    # Get u and v only on the levels 850, 700, and 600 hPa
    lev_list = [53, 56, 63]
    u_levels = np.zeros([3, u_3d.shape[1], u_3d.shape[2]])
    v_levels = np.zeros([3, v_3d.shape[1], v_3d.shape[2]])
    for level_index in range(0, 3):
        u_levels[level_index, :, :] = u_3d.sel(lev=lev_list[level_index])
        v_levels[level_index, :, :] = v_3d.sel(lev=lev_list[level_index])

    # Interpolate to fill any NaNs
    if np.isnan(u_levels).any():
        mask_u = np.isnan(u_levels)
        u_levels[mask_u] = np.interp(np.flatnonzero(
            mask_u), np.flatnonzero(~mask_u), u_levels[~mask_u])
    if np.isnan(v_levels).any():
        mask_v = np.isnan(v_levels)
        v_levels[mask_v] = np.interp(np.flatnonzero(
            mask_v), np.flatnonzero(~mask_v), v_levels[~mask_v])

    # Calculate the relative vorticity
    rel_vort_levels = calc_rel_vort(
        u_levels, v_levels, common_object.lat, common_object.lon)
    # Calculate the curvature vorticity
    curve_vort_levels = calc_curve_vort(
        common_object, u_levels, v_levels, rel_vort_levels)

    # Switch the arrays to be float32 instead of float64
    u_levels = np.float32(u_levels)
    v_levels = np.float32(v_levels)
    rel_vort_levels = np.float32(rel_vort_levels)
    curve_vort_levels = np.float32(curve_vort_levels)

    # Make the arrays C contiguous
    # (will need this later for the wrapped C smoothing function)
    u_levels = np.asarray(u_levels, order='C')
    v_levels = np.asarray(v_levels, order='C')
    rel_vort_levels = np.asarray(rel_vort_levels, order='C')
    curve_vort_levels = np.asarray(curve_vort_levels, order='C')

    return u_levels, v_levels, rel_vort_levels, curve_vort_levels


def get_CAM5_variables(common_object, scenario_type, date_time):
    """
    Get the CAM5 variables required for tracking.

    This function takes the common_object that holds lat/lon information,
    the scenario type, and the date and time for the desired file.
    It returns u, v, relative vorticity, and curvature vorticity on specific
    pressure levels.

    Parameters:
        common_object: Object containing common attributes such as lat/lon
            information.
        scenario_type (str): Type of scenario, e.g., 'Historical' or 'Plus30'.
        date_time (datetime.datetime): Date and time for the desired file.

    Returns:
        tuple: A tuple containing u, v, relative vorticity, and curvature
            vorticity on specific pressure levels.
    """
    # location of CAM5 files
    if scenario_type == 'Historical':
        u_file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/' \
            + scenario_type + \
            '/run3/' + \
            date_time.strftime(
                '%Y') + '/U_CAM5-1-0.25degree_All-Hist_est1_v3_run3.cam.h4.'
        v_file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/' \
            + scenario_type + \
            '/run3/' + \
            date_time.strftime(
                '%Y') + '/V_CAM5-1-0.25degree_All-Hist_est1_v3_run3.cam.h4.'
    elif scenario_type == 'Plus30':
        u_file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/' \
            + scenario_type + \
            '/run3/' + date_time.strftime('%Y') + \
            '/U_fvCAM5_UNHAPPI30_run003.cam.h4.'
        v_file_location = '/global/cscratch1/sd/ebercosh/CAM5/Africa/' \
            + scenario_type + \
            '/run3/' + date_time.strftime('%Y') + \
            '/V_fvCAM5_UNHAPPI30_run003.cam.h4.'
    # open files
    u_data = xr.open_dataset(
        u_file_location + date_time.strftime("%Y-%m-%d") + '-00000_AEW.nc',
        decode_times=False)
    v_data = xr.open_dataset(
        v_file_location + date_time.strftime("%Y-%m-%d") + '-00000_AEW.nc',
        decode_times=False)
    # get u and v
    print("Pulling variables...")
    # the CAM5 data has 8 times in one file (data is 3 hourly),
    # so pull only the hour that matches the current date_time
    # unfortunately xarray has trouble decoding the times in the file so use
    # the following dictionary to get the correct time index
    # using the time from the time loop
    time_dict = {'00': 0, '03': 1, '06': 2,
                 '09': 3, '12': 4, '15': 5, '18': 6, '21': 7}
    # some of the CAM5 data is missing the last time in the file
    # (so 7 times in a file instead of 8). Use try and except to catch these
    # rare cases and then use the previous time step when the last time step
    # is missing. Because this happens so infrequently, using the previous
    # time step does lead to any issues.
    try:
        u_3d = u_data.U[time_dict[date_time.strftime("%H")], :, :, :]
    except IndexError:
        u_3d = u_data.U[time_dict[date_time.strftime("%H")]-1, :, :, :]
    try:
        v_3d = v_data.V[time_dict[date_time.strftime("%H")], :, :, :]
    except IndexError:
        v_3d = v_data.V[time_dict[date_time.strftime("%H")]-1, :, :, :]
    # get u and v only on the levels 850, 700, and 600 hPa
    lev_list = [850, 700, 600]
    u_levels = np.zeros([3, u_3d.shape[1], u_3d.shape[2]])
    v_levels = np.zeros([3, v_3d.shape[1], v_3d.shape[2]])
    for level_index in range(0, 3):
        u_levels[level_index, :, :] = u_3d.sel(plev=lev_list[level_index])
        v_levels[level_index, :, :] = v_3d.sel(plev=lev_list[level_index])

    # get rid of any NANs
    if np.isnan(u_levels).any():
        mask_u = np.isnan(u_levels)
        u_levels[mask_u] = np.interp(np.flatnonzero(
            mask_u), np.flatnonzero(~mask_u), u_levels[~mask_u])
    if np.isnan(v_levels).any():
        mask_v = np.isnan(v_levels)
        v_levels[mask_v] = np.interp(np.flatnonzero(
            mask_v), np.flatnonzero(~mask_v), v_levels[~mask_v])

    # calculate the relative vorticity
    rel_vort_levels = calc_rel_vort(
        u_levels, v_levels, common_object.lat, common_object.lon)
    # calculate the curvature voriticty
    curve_vort_levels = calc_curve_vort(
        common_object, u_levels, v_levels, rel_vort_levels)

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


def get_ERA5_variables(common_object, date_time):
    """
    Get the ERA5 variables required for tracking.

    This function takes the common_object that holds lat/lon information,
    the date and time for the desired file. It returns u, v, relative
    vorticity, and curvature vorticity on specific pressure levels.

    Parameters:
        common_object: Object containing common attributes such as lat/lon
            information.
        date_time (datetime.datetime): Date and time for the desired file.

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

    # pressure list (hPa)
    lev_list = [850, 700, 600]

    # Select the specific variable for U and V components of wind
    u_grbs = [fileidx.select(name='U component of wind',
                             typeOfLevel='isobaricInhPa',
                             level=level)[0] for level in lev_list]
    v_grbs = [fileidx.select(name='V component of wind',
                             typeOfLevel='isobaricInhPa',
                             level=level)[0] for level in lev_list]

    # Extract latitudes and longitudes
    latd, lond = u_grbs[0].values.shape

    # Initialize arrays for U and V component of wind at each pressure level
    u_levels = np.zeros([len(lev_list), latd, lond])
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
        common_object, u_levels, v_levels, rel_vort_levels)

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


def get_ERAI_variables(common_object, date_time):
    """
    Get the ERAI variables required for tracking.

    This function takes the common_object that holds lat/lon information,
    the date and time for the desired file. It returns u, v, relative
    vorticity, and curvature vorticity on specific pressure levels.

    Parameters:
        common_object: Object containing common attributes such as lat/lon
            information.
        date_time (datetime.datetime): Date and time for the desired file.

    Returns:
        tuple: A tuple containing u, v, relative vorticity, and curvature
        vorticity on specific pressure levels.
    """

    # location of ERA-Interim data, which is in GRIB format
    file_location = '/global/cscratch1/sd/ebercosh/Reanalysis/ERA-I/'\
        'ei.oper.an.pl.regn128uv.'
    # open file
    grbs = pygrib.open(file_location + date_time.strftime("%Y%m%d%H"))

    # pressure list
    # 30, 25, and 23 correspond to 850, 700, and 600 hPa, respectively
    lev_list = [30, 25, 23]

    u_levels_360 = np.zeros(
        [3, grbs.select(name='U component of wind')[23].values.shape[0],
         grbs.select(name='U component of wind')[23].values.shape[1]])
    v_levels_360 = np.zeros_like(u_levels_360)
    # get the desired pressure levels
    for level_index in range(0, 3):
        # the ERA-Interim data goes from north to south. Use flip to flip it
        # 180 degrees in the latitude dimension so that
        # the array now goes from south to north like the other datasets.
        u_levels_360[level_index, :, :] = np.flip(
            grbs.select(name='U component of wind')[
                lev_list[level_index]].values, axis=0)
        v_levels_360[level_index, :, :] = np.flip(grbs.select(
            name='V component of wind')[lev_list[level_index]].values, axis=0)

    # need to roll the u and v variables on the longitude axis because the
    # longitudes were changed from
    # 0-360 to -180 to 180
    u_levels = np.roll(u_levels_360, int(u_levels_360.shape[2]/2), axis=2)
    v_levels = np.roll(v_levels_360, int(v_levels_360.shape[2]/2), axis=2)

    # calculate the relative vorticity
    rel_vort_levels = calc_rel_vort(
        u_levels, v_levels, common_object.lat, common_object.lon)
    # calculate the curvature voriticty
    curve_vort_levels = calc_curve_vort(
        common_object, u_levels, v_levels, rel_vort_levels)

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


@jit
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


@jit
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


@jit
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
    # the gradient function will return a list of arrays of the same
    # dimensions as the WRF variable, where each array is a derivative with
    # respect to one of the dimensions
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


def calc_curve_vort_numba(common_object, u, v, rel_vort):
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
        common_object.lat[2, 0] - common_object.lat[1, 0]))
    earth_radius = 6367500.0

    @jit
    def calc_shear_vort(u, v, rel_vort):
        dy = np.full((common_object.lat.shape[0],
                      common_object.lat.shape[1]), earth_radius * dlat)
        dx = np.cos(np.radians(common_object.lat)) * (dy)

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
    if common_object.model == 'WRF':
        u_levels, v_levels, rel_vort_levels, curve_vort_levels = \
            get_WRF_variables(common_object, scenario_type, date_time)
    elif common_object.model == 'MERRA2':
        u_levels, v_levels, rel_vort_levels, curve_vort_levels = \
            get_MERRA2_variables(common_object, date_time)
    elif common_object.model == 'CAM5':
        u_levels, v_levels, rel_vort_levels, curve_vort_levels =\
            get_CAM5_variables(common_object, scenario_type, date_time)
    elif common_object.model == 'ERA5':
        u_levels, v_levels, rel_vort_levels, curve_vort_levels = \
            get_ERA5_variables(common_object, date_time)
    elif common_object.model == 'ERAI':
        u_levels, v_levels, rel_vort_levels, curve_vort_levels = \
            get_ERAI_variables(common_object, date_time)

    return u_levels, v_levels, rel_vort_levels, curve_vort_levels

# %%
