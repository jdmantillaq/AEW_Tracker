# %%
from __future__ import division  # makes division not round with integers
import numpy as np
import math as math
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
from math import inf
# from scipy import inf
from collections.abc import Iterable
from numba import jit, njit

'''
Tracking_functions.py contains all of the tracking-related functions used by
AEW_Tracks.py. This includes finding the starting points, correcting the
starting points, accounting for duplicate locations, filtering out tracks,
and advecting tracks.
'''


class filter_result:
    """
    A class representing filter_result objects.

    These objects are returned by the filter function and have two boolean
    attributes:

    1) reject_track:
        - Indicates whether the track is completely removed from the
          AEW_track_list.

    2) finished_track:
        - Indicates whether the track has ended and should be removed from
          AEW_track_list and added to the permanent finished_AEW_tracks_list.
    """

    def __init__(self):
        self.reject_track = bool
        self.finished_track = bool


def get_starting_targets(common_object, curve_vort_smooth):
    """
    Get the starting latitude and longitude values for AEW tracks.

    Parameters:
        common_object (object): The common object.
        curve_vort_smooth (array): Smoothed curvature vorticity.

    Returns:
        list: A list of lat/lon pairs as tuples representing the locations
        of unique vorticity maxima.
    """
    max_indices = peak_local_max(
        curve_vort_smooth[:, :],
        min_distance=5,
        threshold_abs=common_object.min_threshold)

    lon = common_object.lon
    lat = common_object.lat

    # get differences between neighboring lat and lon points
    dlon = lon[0, 1] - lon[0, 0]
    dlat = lat[1, 0] - lat[0, 0]

    unique_max_locs = []
    # delta = int((common_object.radius / (111*common_object.res))*0.6)
    for max_index in max_indices:
        # Extract the lat/lon indices of the peak location
        max_lat_index, max_lon_index = max_index

        # Compute the latitude and longitude values corresponding to the peak
        max_lat = lat[max_lat_index, max_lon_index] + dlat / 2
        max_lon = lon[max_lat_index, max_lon_index] + dlon / 2

        # # get new max lat and lon indices using the adjusted max_lat and
        # # max_lon valus above and adding or subtracting delta
        # # Calculate the current max lat and lon indices using absolute differences
        # max_lat_index = np.abs(lat[:, 0] - max_lat).argmin()
        # max_lon_index = np.abs(lon[0, :] - max_lon).argmin()

        # # Add and subtract the delta to/from the max indices to get a range
        # # around the max points
        # max_lat_index_plus_delta = max_lat_index + delta
        # max_lat_index_minus_delta = max_lat_index - delta
        # max_lon_index_plus_delta = max_lon_index + delta
        # max_lon_index_minus_delta = max_lon_index - delta

        # # create a cropped version of the variable array, lat and lon arrays
        # # using the delta modified lat/lon indices above
        # var_crop = curve_vort_smooth[
        #     max_lat_index_minus_delta:max_lat_index_plus_delta,
        #     max_lon_index_minus_delta:max_lon_index_plus_delta]

        # # if np.median(var_crop) > common_object.min_threshold:
        # # Append the location as a tuple
        unique_max_locs.append((max_lat, max_lon))

    # if len(unique_max_locs) > 1:
    #     unique_max_locs = unique_locations(
    #         unique_max_locs, common_object.radius, 99999999)

    return unique_max_locs


@jit
def great_circle_dist_km(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two lat/lon points.

    Parameters:
        lon1 (float): Longitude of the first point.
        lat1 (float): Latitude of the first point.
        lon2 (float): Longitude of the second point.
        lat2 (float): Latitude of the second point.

    Returns:
        float: The distance between the two points in kilometers.
    """
    # switch degrees to radians
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # dlon = lon2 - lon1
    # dlat = lat2 - lat1
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # note that everything was already switched to radians a few lines above
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * \
        np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    dist_km = 6371. * c  # 6371 is the radius of the Earth in km

    return dist_km


def unique_locations(max_locations, radius, unique_loc_number):
    """
    Average locations that are within a specified radius in kilometers of
    each other.

    This function processes a list of latitude/longitude locations
    (as tuples) and averages those that fall within a given radius.
    It uses recursion to ensure that all locations 
    within the radius are accounted for until no more duplicates are found. 

    Parameters:
        max_locations (list): List of latitude/longitude locations as tuples.
        radius (float): The radius in kilometers for averaging nearby locations.
        unique_loc_number (int): A large integer for initial comparison.

    Returns:
        list: A list of averaged latitude/longitude tuples
                    representing unique locations.
    """

    # Sort locations by latitude for efficient processing from south to north
    max_locations.sort(key=lambda x: x[0])

    # Convert radius from kilometers to degrees (approximate)
    max_distance = radius/111.  # convert to degrees

    # Create a KD-Tree for efficient spatial searches
    tree = cKDTree(max_locations)

    # List to hold averaged neighboring points
    point_neighbors_list_avg = []
    first_time = True
    # Iterate through all maximum locations
    for max_loc in max_locations:
        # check to see if it's the first time
        if first_time:
            first_time = False
        else:
            # Skip already processed nearby points
            if max_loc in point_neighbors:
                continue

        # Query the KD-Tree for neighbors within the specified distance
        distances, indices = tree.query(max_loc, len(
            max_locations), p=10, distance_upper_bound=max_distance)

        # Ensure distances and indices are iterable
        if not isinstance(distances, Iterable) and \
                not isinstance(indices, Iterable):
            distances = [distances]
            indices = [indices]
        # Collect close points based on the distances
        point_neighbors = []
        for index, distance in zip(indices, distances):
            if distance == inf:
                break  # Ignore points beyond the max distance
            point_neighbors.append(max_locations[index])

        # Average the neighboring points and store the result
        point_neighbors_list_avg.append(tuple(np.mean(point_neighbors,
                                                      axis=0)))

    # Count unique locations, removing duplicates
    new_unique_loc_number = len(list(set(point_neighbors_list_avg)))

    # Check if the number of unique locations has stabilized
    if new_unique_loc_number == unique_loc_number:
        return list(set(point_neighbors_list_avg))
    else:
        # Recursively call the function with updated unique location number
        return unique_locations(point_neighbors_list_avg, radius,
                                new_unique_loc_number)


def unique_track_locations(track_object, unique_max_locs,
                           current_time_index, radius):
    """
    Ensures unique track locations by averaging nearby points within a
    specified radius.

    Modifies the track object by averaging lat/lon pairs that are within the 
    given radius of the track's current location, preventing duplicates. The 
    close lat/lon pairs from unique_max_locs are removed after averaging.

    Parameters:
        track_object (object): The track object being updated.
        unique_max_locs (list): List of vorticity maximum lat/lon locations.
        current_time_index (int): Index for the current time step.
        radius (float): Distance threshold in kilometers for averaging.

    Returns:
        None
    """

    # Get the current lat/lon pair (track) for the AEW
    current_latlon_pair_list = track_object.latlon_list[
        current_time_index: min(current_time_index + 1,
                                len(track_object.latlon_list))]

    # loop through the lat/lon locations in the unique_max_locs list
    for latlon_loc in list(unique_max_locs):
        for current_latlon_pair in current_latlon_pair_list:

            dist_km = great_circle_dist_km(
                current_latlon_pair[1], current_latlon_pair[0],
                latlon_loc[1], latlon_loc[0])
            # If the distance is less than the radius, average
            # the lat/lon and update
            if dist_km < radius:
                track_object.latlon_list = [
                    ((latlon_loc[0]+current_latlon_pair[0])/2,
                     (latlon_loc[1] + current_latlon_pair[1])/2)
                    if x == current_latlon_pair else x for x in
                    track_object.latlon_list]

                unique_max_locs.remove(latlon_loc)
                continue
    return


def calculate_weighted_mean(near_max_locs, var_values, current_latlon_pair):
    """Calculate weighted mean lat/lon, giving more weight to westward and
    latitudinally close points."""

    # Extract current latitude and longitude
    current_lat, current_lon = current_latlon_pair

    # Calculate latitudinal and longitudinal differences
    lat_diff = np.abs(near_max_locs[:, 0] - current_lat)
    # Positive if westward (smaller longitude)
    lon_diff = current_lon - near_max_locs[:, 1]

    # Give more weight to points west of the current location (positive lon_diff)
    # Increase weight by 1.5x if westward
    lon_weight = np.where(lon_diff > 0, 1.8, 1.0)

    # Give more weight to points closer in latitude (smaller lat_diff)
    # Prevent division by zero with a small epsilon
    lat_weight = 1 / (lat_diff + 1e-6)

    # Final weights: combine variable values with lat and lon weights
    combined_weights = np.abs((var_values) * lon_weight * lat_weight)

    if np.sum(combined_weights) != 0:
        # Weighted mean of lat/lon based on the combined weights
        lat_pon = np.sum(
            near_max_locs[:, 0] * combined_weights) / np.sum(combined_weights)
        lon_pon = np.sum(
            near_max_locs[:, 1] * combined_weights) / np.sum(combined_weights)
    else:
        # Default to the mean if combined weights sum to zero
        lat_pon = np.mean(near_max_locs[:, 0])
        lon_pon = np.mean(near_max_locs[:, 1])

    return lat_pon, lon_pon


def unique_track_locations_2(common_object, track_object, unique_max_locs,
                             current_time_index, var):
    """
    Update track locations based on proximity to nearby vorticity maxima.

    This function modifies the track object's location by averaging nearby
    vorticity maximum points within a specified radius. The weighted mean 
    of lat/lon is calculated based on the variable values at these locations.

    Parameters:
        common_object (object): Object containing geographic and threshold data.
        track_object (object): Track object to be updated.
        unique_max_locs (list): List of vorticity maximum lat/lon locations.
        current_time_index (int): Current time step index.
        var (ndarray): Variable field (e.g., vorticity) for weighting.

    Returns:
        None
    """

    unique_max_locs_array = np.array(unique_max_locs)

    # Get the current lat/lon pair (track) for the AEW
    current_latlon_pair_list = track_object.latlon_list[
        current_time_index: min(current_time_index + 1,
                                len(track_object.latlon_list))]

    for current_latlon_pair in current_latlon_pair_list:
        # Calculate distances from the current lat/lon pair
        # to each unique max location
        dist_km = np.array([great_circle_dist_km(
            current_latlon_pair[1], current_latlon_pair[0],
            latlon_loc[1], latlon_loc[0])
            for latlon_loc in unique_max_locs_array])

        # Find the indices of locations within the specified radius
        idx_distance = np.where(dist_km <= common_object.radius)[0]
        unique_max_locs_copy = unique_max_locs_array.copy()

        if len(idx_distance) > 0:
            # Locations within the radius
            near_max_locs = unique_max_locs_array[idx_distance]

            # # Append the current lat/lon pair to the near_max_locs
            # near_max_locs = np.vstack([near_max_locs, current_latlon_pair])
            # dist_km  = np.hstack([dist_km[idx_distance], 1])

            # Extract variable values at the corresponding grid points
            # for near_max_locs
            var_values = []
            for latlon_loc in near_max_locs:
                # Find the nearest indices in the lat/lon grids
                lat_index = np.abs(
                    common_object.lat[:, 0] - latlon_loc[0]).argmin()
                lon_index = np.abs(
                    common_object.lon[0, :] - latlon_loc[1]).argmin()
                var_values.append(var[lat_index, lon_index])

            var_values = np.abs(np.array(var_values))
            var_values *= 1.5

            # Calculate weighted mean lat/lon with the modified weighting scheme
            lat_pon, lon_pon = calculate_weighted_mean(
                near_max_locs, var_values, current_latlon_pair)

           # Adjust new location if it moves eastward too much
            if lon_pon < current_latlon_pair[1]:
                weight_new = 1
                weight_current = 1
                weight_sum = weight_new + weight_current
                lat_pon = (lat_pon*weight_new
                           + current_latlon_pair[0]*weight_current)/weight_sum
                lon_pon = (lon_pon*weight_new
                           + current_latlon_pair[1]*weight_current)/weight_sum
            else:
                lat_pon, lon_pon = current_latlon_pair

             # Snap lat/lon to the nearest grid point
            lat_index = np.abs(common_object.lat[:, 0] - lat_pon).argmin()
            lon_index = np.abs(common_object.lon[0, :] - lon_pon).argmin()

            lat_pon = common_object.lat[lat_index, 0]
            lon_pon = common_object.lon[0, lon_index]

            # Update the track with the new snapped lat/lon pair
            track_object.latlon_list = [
                (lat_pon, lon_pon)
                if x == current_latlon_pair else x for x in
                track_object.latlon_list
            ]
            # Remove the used locations from unique_max_locs
            for idx in idx_distance:
                unique_max_locs.remove(tuple((unique_max_locs_copy[idx])))
    return


def assign_magnitude(common_object, curve_vort_smooth, track_object):
    """
    Assign vorticity magnitude to the last location in an AEW track.

    Parameters:
        common_object (object): Object containing latitude and longitude grids.
        curve_vort_smooth (ndarray): Smoothed curvature vorticity field.
        track_object (object): The AEW track object to update.

    Returns:
        None
    """
    # Ensure there is a new location in the track to assign a magnitude
    if len(track_object.latlon_list) - len(track_object.magnitude_list) == 1:
        # Get the latest lat/lon pair
        lat_lon_pair = track_object.latlon_list[-1]

        # Find the nearest grid indices for the lat/lon pair
        lat_index = (
            np.abs(common_object.lat[:, 0] - lat_lon_pair[0])).argmin()
        lon_index = (
            np.abs(common_object.lon[0, :] - lat_lon_pair[1])).argmin()

        # Assign the corresponding vorticity magnitude
        track_object.add_magnitude(curve_vort_smooth[lat_index, lon_index])
        return
    else:
        return


def filter_tracks(common_object, track_object, AEW_tracks_list):
    """
    Filter AEW tracks based on directional movement, vorticity, latitude limits,
    and proximity to other tracks.

    Parameters:
        common_object (object): Contains thresholds and region data.
        track_object (object): The AEW track being evaluated.
        AEW_tracks_list (list): List of other AEW tracks for proximity checks.

    Returns:
        filter_result_object (object): Indicates whether the track passes
        or fails filtering criteria.
    """
    # Initialize the filter result object
    filter_result_object = filter_result()

    # 1. Filter tracks moving eastward (tracks should move west)
    if len(track_object.latlon_list) > 4:
        longitudes = [point[1] for point in track_object.latlon_list[-5:]]

        # Check if track is moving eastward
        if all(longitudes[i] < longitudes[i + 1] for i in range(len(longitudes) - 1)):
            filter_result_object.reject_track_direction = True
        else:
            filter_result_object.reject_track_direction = False
    else:
        filter_result_object.reject_track_direction = False

    # 2. Filter based on vorticity magnitude (more than 25% negative values)
    if len(track_object.latlon_list) > 4:
        magnitudes = np.array(track_object.magnitude_list)
        negative_count = np.sum(magnitudes < 0)

        # Check if more than 25% of the magnitudes are negative
        if negative_count > 0.25 * len(magnitudes):
            filter_result_object.magnitude_finish_track = True
        else:
            filter_result_object.magnitude_finish_track = False
    else:
        filter_result_object.magnitude_finish_track = False

     # 3. Filter weak tracks (more than 4 weak points for tracks > 6 points)
    if len(track_object.latlon_list) > 6:
        weak_points = sum(1 for magnitude in track_object.magnitude_list
                          if magnitude < common_object.min_threshold)
        filter_result_object.magnitude_finish_track = weak_points > 4
    else:
        filter_result_object.magnitude_finish_track = False

     # 4. Filter tracks exceeding latitude limits
    last_latitude = track_object.latlon_list[-1][0]
    if last_latitude > 40. or last_latitude < -5.:
        filter_result_object.latitude_finish_track = True
    else:
        filter_result_object.latitude_finish_track = False

    # 5. Filter based on proximity to other AEW tracks' last locations
    last_lat, last_lon = track_object.latlon_list[-1]

    # Initialize distances with a high value
    dist_km = np.full(len(AEW_tracks_list), 9999.0)

    for i, other_track in enumerate(AEW_tracks_list):
        # Skip comparing the track with itself
        if other_track.id == track_object.id:
            continue
        try:
            # Calculate the distance to other tracks' last positions
            other_last_lat, other_last_lon = other_track.latlon_list[-1]

            dist_km[i] = great_circle_dist_km(
                last_lat, last_lon, other_last_lat, other_last_lon)
        except (IndexError, ValueError):
            # Handle cases where the time index is not present
            # or lat/lon is unavailable
            continue

    # Reject if any other AEW track's last point is within the specified radius
    if np.any(dist_km <= common_object.radius):
        filter_result_object.reject_due_to_proximity = True
    else:
        filter_result_object.reject_due_to_proximity = False

    return filter_result_object


def circle_avg_m_point(common_object, var, lat_lon_pair):
    """
    Compute a smoothed average for a variable over a circular region
    centered on a lat/lon point.

    Parameters:
        common_object (object): Contains regional grid information and
                            parameters.
        var (numpy.ndarray): The variable (e.g., u or v wind) to smooth.
        lat_lon_pair (tuple): Latitude and longitude of the center point.

    Returns:
        smoothed_var (numpy.ndarray): Smoothed variable at the specified
        lat/lon point.
    """
    # Take cos of lat in radians
    cos_lat = np.cos(np.radians(common_object.lat))

    R = 6371.  # Earth radius in km

    # Compute the number of grid points within the smoothing radius (in km)
    radius_gridpts = int(common_object.radius*(common_object.lat.shape[1]/(
        (common_object.total_lon_degrees/360)*2*np.pi*R)))

    # create a copy of the var array
    smoothed_var = np.copy(var)

    # Find the nearest lat/lon indices for the point of interest
    lat_index_maxima = (
        np.abs(common_object.lat[:, 0] - lat_lon_pair[0])).argmin()
    lon_index_maxima = (
        np.abs(common_object.lon[0, :] - lat_lon_pair[1])).argmin()
    # take circle average
    tempv = 0.0
    divider = 0.0

    # Perform circular averaging around the lat/lon point
    for radius_index in range(-radius_gridpts, radius_gridpts+1):

        # Skip if out of latitude bounds
        if (lat_index_maxima+radius_index) < 0 \
            or (lat_index_maxima+radius_index) \
                > (common_object.lat.shape[1]-1):
            continue

        lat1 = common_object.lat[lat_index_maxima,
                                 lon_index_maxima]  # center of circle
        # vertical distance from circle center
        lat2 = common_object.lat[lat_index_maxima +
                                 radius_index, lon_index_maxima]

        # Calculate angular distance and convert to longitudinal grid points
        angle_rad = np.arccos(
            -((np.sin(np.radians(lat1))*np.sin(np.radians(lat2)))
                - np.cos(common_object.radius/R))/(np.cos(np.radians(lat1))
                                                   * np.cos(
                                                       np.radians(lat2))))
        lon_gridpts = int((angle_rad/0.0174533) *
                          (common_object.lat.shape[1]/360.))

        # Average over the circle's longitudinal extent
        for lon_circle_index in range(lon_index_maxima-lon_gridpts,
                                      lon_index_maxima+lon_gridpts+1):
            # the following conditionals handle the cases where the longitude
            # index is out of bounds
            # (from the Albany code that had global data)
            cyclic_lon_index = lon_circle_index
            if cyclic_lon_index < 0:
                cyclic_lon_index = cyclic_lon_index+common_object.lat.shape[1]
            if cyclic_lon_index > common_object.lat.shape[1]-1:
                cyclic_lon_index = cyclic_lon_index-common_object.lat.shape[1]

            tempv = tempv + (cos_lat[lat_index_maxima+radius_index,
                                     lon_index_maxima]
                             * var[(lat_index_maxima+radius_index),
                                   cyclic_lon_index])
            divider = divider + \
                cos_lat[lat_index_maxima+radius_index, lon_index_maxima]

    smoothed_var[lat_index_maxima, lon_index_maxima] = tempv/divider

    return smoothed_var


@jit
def circle_avg_m_point_numba(var, lat_lon_pair,
                             lat, lon, radius, total_lon_degrees,
                             ):
    """
    Calculate a circle average at a specific point.

    This function calculates the circle smoothed variable at a specified
    lat/lon point.  Unlike the c smoothing, which smooths a larger domain,
    this function focuses on smoothing  centered around the lat/lon point of
    interest.

    Parameters:
        common_object (object): The common object containing region indices
            and other common parameters.
        var (numpy.ndarray): The variable (u or v) to be smoothed.
        lat_lon_pair (tuple): The lat/lon pair from an AEW track.

    Returns:
        smoothed_var (numpy.ndarray): The circle smoothed variable at the
            specified lat/lon point.
    """
    # Take cos of lat in radians
    cos_lat = np.cos(np.radians(lat))

    R = 6371.  # Earth radius in km
    # Get the number of gridpoints equivalent to the radius being used for
    # the smoothing. To convert the smoothing radius in km to gridpoints,
    # multiply the radius (in km) by the total number of longitude
    # gridpoints = var.shape[2] for the whole domain divided by the degrees
    # of longitude in the domain divided by 360 times the circumference of
    # the Earth = 2piR.
    # The degrees of longitude/360 * circumference is to
    # scale the circumference to account for non-global data. This is also a
    # rough approximation since we're not quite at the equator.
    #
    # radius_gridpts = radius (in km) *
    #            (longitude gridpoints / scaled circumference of Earth (in km))
    #
    # Make radius_gridpts an int so it can be used as a loop index later.
    radius_gridpts = int(radius*(lat.shape[1]/(
        (total_lon_degrees/360)*2*np.pi*R)))

    # create a copy of the var array
    smoothed_var = np.copy(var)

    # get the indices for the lat/lon pairs of the maxima
    lat_index_maxima = (
        np.abs(lat[:, 0] - lat_lon_pair[0])).argmin()
    lon_index_maxima = (
        np.abs(lon[0, :] - lat_lon_pair[1])).argmin()
    # take circle average
    tempv = 0.0
    divider = 0.0
    # work wayup circle
    for radius_index in range(-radius_gridpts, radius_gridpts+1):

        # make sure we're not goint out of bounds, and if we are go to the
        # next iteration of the loop
        if (lat_index_maxima+radius_index) < 0 \
            or (lat_index_maxima+radius_index) \
                > (lat.shape[1]-1):
            continue

        lat1 = lat[lat_index_maxima,
                   lon_index_maxima]  # center of circle
        # vertical distance from circle center
        lat2 = lat[lat_index_maxima +
                   radius_index, lon_index_maxima]
        # make sure that lat2, which has the radius added, doesn't go off the
        # grid (either off the top or the bottom)

        # need to switch all angles from degrees to radians (haversine trig)
        angle_rad = np.arccos(
            -((np.sin(np.radians(lat1))*np.sin(np.radians(lat2)))
                - np.cos(radius/R))/(np.cos(np.radians(lat1))
                                     * np.cos(
                    np.radians(lat2))))

        # convert angle from radians to degrees and then from degrees to
        # gridpoints divide angle_rad by pi/180 = 0.0174533 to convert radians
        # to degrees multiply by lat.shape[1]/360 which is the lon gridpoints
        # over the total 360 degrees around the globe the conversion to
        # gridpoints comes from (degrees)*(gridpoints/degrees) = gridpoints
        # lon_gridpts defines the longitudinal grid points for each lat
        lon_gridpts = int((angle_rad/0.0174533) *
                          (lat.shape[1]/360.))

        # work across circle
        for lon_circle_index in range(lon_index_maxima-lon_gridpts,
                                      lon_index_maxima+lon_gridpts+1):
            # the following conditionals handle the cases where the longitude
            # index is out of bounds
            # (from the Albany code that had global data)
            cyclic_lon_index = lon_circle_index
            if cyclic_lon_index < 0:
                cyclic_lon_index = cyclic_lon_index+lat.shape[1]
            if cyclic_lon_index > lat.shape[1]-1:
                cyclic_lon_index = cyclic_lon_index-lat.shape[1]

            tempv = tempv + (cos_lat[lat_index_maxima+radius_index,
                                     lon_index_maxima]
                             * var[(lat_index_maxima+radius_index),
                                   cyclic_lon_index])
            divider = divider + \
                cos_lat[lat_index_maxima+radius_index, lon_index_maxima]

    smoothed_var[lat_index_maxima, lon_index_maxima] = tempv/divider

    return smoothed_var


@njit
def circle_avg_m_point_numba_optimized(var, lat_lon_pair, lat, lon,
                                       radius, total_lon_degrees):
    """
    Optimized calculation of circle average at a specific lat/lon
    point using Numba.

    This function calculates the smoothed variable centered at a specific
    lat/lon point, using a radius defined in kilometers and converts it to
    grid points for smoother computation.

    Parameters:
        var (numpy.ndarray): Variable to be smoothed (e.g., u or v component).
        lat_lon_pair (tuple): The latitude and longitude pair of interest.
        lat (numpy.ndarray): Latitude array.
        lon (numpy.ndarray): Longitude array.
        radius (float): Radius in kilometers for the smoothing.
        total_lon_degrees (float): The total degrees of longitude in the domain.

    Returns:
        smoothed_var (numpy.ndarray): Circle-averaged variable centered
                    at the lat/lon point.
    """
    cos_lat = np.cos(np.radians(lat))
    R = 6371.0  # Earth radius in km

    # Convert radius from km to grid points
    radius_gridpts = \
        int(radius * (lat.shape[1] /
            ((total_lon_degrees / 360) * 2 * np.pi * R)))

    # Create a copy of the variable to store smoothed values
    smoothed_var = np.copy(var)

    # Find the closest grid point for the specified lat/lon pair
    lat_index_maxima = np.abs(lat[:, 0] - lat_lon_pair[0]).argmin()
    lon_index_maxima = np.abs(lon[0, :] - lat_lon_pair[1]).argmin()

    tempv = 0.0
    divider = 0.0

    # Loop through the circle's radius to compute the average
    for radius_index in range(-radius_gridpts, radius_gridpts + 1):
        lat_index = lat_index_maxima + radius_index
        if lat_index < 0 or lat_index >= lat.shape[0]:
            continue  # Skip if out of bounds

        lat1 = lat[lat_index_maxima, lon_index_maxima]
        lat2 = lat[lat_index, lon_index_maxima]

        # Compute distance in radians using the haversine formula
        cos_lat1, cos_lat2 = np.cos(np.radians(lat1)), np.cos(np.radians(lat2))
        sin_lat1, sin_lat2 = np.sin(np.radians(lat1)), np.sin(np.radians(lat2))

        # Calculate angle in radians between lat1 and lat2
        angle_rad = np.arccos(sin_lat1 * sin_lat2 +
                              cos_lat1 * cos_lat2 * np.cos(radius / R))

        # Convert the angle from radians to grid points
        lon_gridpts = int((angle_rad / 0.0174533) * (lat.shape[1] / 360.))

        for lon_circle_index in range(lon_index_maxima - lon_gridpts,
                                      lon_index_maxima + lon_gridpts + 1):
            cyclic_lon_index = lon_circle_index % lat.shape[1]
            # Handle cyclic longitude indices

            # Accumulate weighted variable value
            tempv += cos_lat[lat_index, lon_index_maxima] * \
                var[lat_index, cyclic_lon_index]
            divider += cos_lat[lat_index, lon_index_maxima]

    # Assign the average value to the central point
    if divider != 0:
        smoothed_var[lat_index_maxima, lon_index_maxima] = tempv / divider
    return smoothed_var


def advect_tracks(common_object, u_2d, v_2d, track_object, times, time_index):
    """
    Advects the last lat/lon point in the track using wind to estimate the
    next position.

    Parameters:
        common_object (object): Contains grid and region parameters.
        u_2d (numpy.ndarray): Zonal wind component (u).
        v_2d (numpy.ndarray): Meridional wind component (v).
        track_object (object): The track object to update.
        times (numpy.ndarray): Array of time steps.
        time_index (int): Current time step index.

    Returns:
        None
    """
    # Get the last lat/lon pair from the track
    lat_lon_pair = track_object.latlon_list[-1]

    # Smooth the u and v wind fields around the current lat/lon location
    attr_prop_obj = {i: getattr(common_object, i)
                     for i in ['lat', 'lon',
                               'radius',
                               'total_lon_degrees',
                               ]}

    try:
        u_2d_smooth = circle_avg_m_point_numba_optimized(u_2d, lat_lon_pair,
                                                         **attr_prop_obj)
        v_2d_smooth = circle_avg_m_point_numba_optimized(v_2d, lat_lon_pair,
                                                         **attr_prop_obj)

    except:
        print(lat_lon_pair)

    # Find the nearest grid indices for the lat/lon pair
    lat_index = (np.abs(common_object.lat[:, 0] - lat_lon_pair[0])).argmin()
    lon_index = (np.abs(common_object.lon[0, :] - lat_lon_pair[1])).argmin()

    # Calculate the new lat/lon using the wind fields and time step (dt)
    new_lat_value = lat_lon_pair[0] + (
        (v_2d_smooth[lat_index, lon_index] * 60. * 60. * common_object.dt) /
        111120.)
    new_lon_value = lat_lon_pair[1] + (
        (u_2d_smooth[lat_index, lon_index] * 60. * 60. *
         common_object.dt) / 111120.) * np.cos(np.radians(lat_lon_pair[0]))

    # Ensure the next time step is within bounds
    if time_index + 1 < times.shape[0]:
        # add the next time to the end of the track_objects time list
        track_object.add_time(times[time_index + 1])

        # Check for NaN or out-of-bounds lat/lon values
        if np.any(np.isnan([new_lat_value, new_lon_value]))\
                or np.any(np.isinf([new_lat_value, new_lon_value])):
            return
        if not is_in_data(new_lat_value, new_lon_value,
                          common_object.lat_min,
                          common_object.lat_max,
                          common_object.lon_min,
                          common_object.lon_max):
            return

        # Add the new lat/lon pair to the track
        track_object.add_latlon((new_lat_value, new_lon_value))

    return


@jit
def is_in_data(lat_i, lon_i, lat_min, lat_max, lon_min, lon_max):
    return (lat_min <= lat_i <= lat_max) and (lon_min <= lon_i <= lon_max)
