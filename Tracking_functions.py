# %%
from __future__ import division  # makes division not round with integers
import numpy as np
import math as math
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
from math import inf
# from scipy import inf
from collections.abc import Iterable
# %%
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
    # get a list of indices (lat/lon index pairs) of local maxima in the
    # smoothed curvature vorticity field
    max_indices = peak_local_max(
        curve_vort_smooth[
            1,
            common_object.lat_index_south:common_object.lat_index_north+1,
            common_object.lon_index_west:common_object.lon_index_east+1],
        min_distance=1,
        threshold_abs=common_object.min_threshold
    )  # indx come out ordered lat, lon

    # get a list of weighted averaged lat/lon index pairs
    weighted_max_indices = get_mass_center(
        common_object, curve_vort_smooth[1, :, :], max_indices)

    # Remove duplicate locations. This checks to see if new starting targets
    # actually belong to existing tracks. The 99999999 is the starting value
    # for unique_loc_number; it just needs to be way bigger than the possible
    # number of local maxima in weighted_max_indices. This function
    # recursively calls itself until the value for unique_loc_number doesn't
    # decrease anymore.
    # Only need to check for duplicate locations if there is more than one
    # location (meaning len(weighted_max_indices) > 1).
    # If there is only one member of the the list,
    # then unique_max_locs = weighted_max_indices.
    if len(weighted_max_indices) > 1:
        unique_max_locs = unique_locations(
            weighted_max_indices, common_object.radius, 99999999)
    else:
        unique_max_locs = weighted_max_indices

    return unique_max_locs


def get_mass_center(common_object, var, max_indices):
    """
    Calculate the weighted average lat/lon indices of vorticity maxima.

    Parameters:
        common_object (object): The common object.
        var (array): A variable (vorticity).
        max_indices (list): A list of lat/lon indices of vorticity maxima.

    Returns:
        list: A list of lat/lon values.
    """
    # set a minimum radius in km
    min_radius_km = 100.  # km

    # lists to store the weighted lats and lons
    weight_lat_list = []
    weight_lon_list = []

    # get differences between neighboring lat and lon points
    dlon = common_object.lon[0, 1] - common_object.lon[0, 0]
    dlat = common_object.lat[1, 0] - common_object.lat[0, 0]

    # this is the approximate number of degrees covered by the common_object
    # radius + 4 to have a buffer ceil rounds up to the nearest int
    delta = int(math.ceil(common_object.radius/111.)+4)

    # loop through all the max indices and calculate weighted lat and lon
    # values for the max locations
    for max_index in max_indices:
        # to get the max lat and lon indices, we need to add lat_index_south
        # and lon_index_west because the max indices were found using a dataset
        # that was cropped over the region of interest in get_starting_targets
        max_lat_index = max_index[0] + \
            common_object.lat_index_south  # this is an index
        max_lon_index = max_index[1] + \
            common_object.lon_index_west  # this is an index

        # get the max lat and lon values from the max lat and lon indices and
        # then add dlat/2 and dlon/2 to nudge the points
        # this is the actual latitude value
        max_lat = common_object.lat[max_lat_index, max_lon_index] + (dlat/2.)
        # this is the actual longitude value
        max_lon = common_object.lon[max_lat_index, max_lon_index] + (dlon/2.)

        # get new max lat and lon indices using the adjusted max_lat and
        # max_lon valus above and adding or subtracting delta
        max_lat_index_plus_delta = (
            np.abs(common_object.lat[:, 0] - (max_lat+delta))).argmin()
        max_lat_index_minus_delta = (
            np.abs(common_object.lat[:, 0] - (max_lat-delta))).argmin()
        max_lon_index_plus_delta = (
            np.abs(common_object.lon[0, :] - (max_lon+delta))).argmin()
        max_lon_index_minus_delta = (
            np.abs(common_object.lon[0, :] - (max_lon-delta))).argmin()

        # create a cropped version of the variable array, lat and lon arrays
        # using the delta modified lat/lon indices above
        var_crop = var[max_lat_index_minus_delta:max_lat_index_plus_delta,
                       max_lon_index_minus_delta:max_lon_index_plus_delta]
        lat_crop = common_object.lat[
            max_lat_index_minus_delta:max_lat_index_plus_delta,
            max_lon_index_minus_delta:max_lon_index_plus_delta]
        lon_crop = common_object.lon[
            max_lat_index_minus_delta:max_lat_index_plus_delta,
            max_lon_index_minus_delta:max_lon_index_plus_delta]

        # Find mass center over large area first, and then progressively make
        # the radius smaller to hone in on the center using a weighted average.
        weight_lat, weight_lon = subgrid_location_km(
            var_crop, max_lat, max_lon, lat_crop,
            lon_crop, common_object.radius)

        # now find mass center over a smaller area by dividing
        # common_object.radius by n (defined below). Keep doing this while
        # common_object.radius/n is greater than the min radius defined at
        # the beginning of this function.
        n = 2
        while common_object.radius/float(n) > min_radius_km:
            weight_lat, weight_lon = subgrid_location_km(
                var_crop, weight_lat, weight_lon, lat_crop, lon_crop,
                common_object.radius/float(n))
            n += 1
        # one last round of taking a weighted average, this time using the
        # minimum radius.
        weight_lat, weight_lon = subgrid_location_km(
            var_crop, weight_lat, weight_lon, lat_crop,
            lon_crop, min_radius_km)

        weight_lat_list.append(weight_lat)
        weight_lon_list.append(weight_lon)
        del weight_lat
        del weight_lon

    # zip the weighted lat and lon lists together to get a list of ordered
    # pairs [(lat1,lon1), (lat2,lon2), etc.]
    weight_lat_lon_list = list(zip(weight_lat_list, weight_lon_list))

    return weight_lat_lon_list


def subgrid_location_km(var_crop, max_lat, max_lon, lat, lon, radius):
    """
    Find the weighted maxima lat/lon point from a subset of data.

    Parameters:
        var_crop (array): Cropped variable (cropping comes from the
            common_object).
        max_lat (float): The maximum latitude value.
        max_lon (float): The maximum longitude value.
        lat (array): Array of latitude values.
        lon (array): Array of longitude values.
        radius (float): The radius to use for the weights.

    Returns:
        tuple: A tuple containing the weighted average lat and lon values.
    """
    # replace all values less than zero with zero
    var_crop[var_crop < 0] = 0.0
    # calculate the great circle distance in km between the max lat/lon point
    # and all of the lat and lon values from the cropped lat and lon arrays
    gc_dist = great_circle_dist_km(max_lon, max_lat, lon, lat)
    # calculate the weights using a generous barnes-like weighting function
    # (from Albany) and then use the weights on var_crop
    weights = np.exp(-1 * ((gc_dist**2)/(radius**2)))
    var_crop = (var_crop**2)*weights
    # flatten the var_crop array
    var_crop_1d = var_crop.flatten()
    # set any values in the flattened var_crop array less than zero
    # equal to zero
    var_crop_1d[var_crop_1d < 0] = 0.0
    # check to see if all the values in var_crop_1d are equal to zero.
    # If they are, then return the original max_lat and max_lon that were
    # passed in as parameters. If not, then use a weighted average with
    # var_crop_1d on the lat and lon arrays to get new lat and lon values.
    if var_crop_1d.all() == 0:
        return max_lat, max_lon
    else:
        weight_lat = np.average(lat.flatten(), weights=var_crop_1d)
        weight_lon = np.average(lon.flatten(), weights=var_crop_1d)

    return weight_lat, weight_lon


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
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # note that everything was already switched to radians a few lines above
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist_km = 6371. * c  # 6371 is the radius of the Earth in km

    return dist_km


def unique_locations(max_locations, radius, unique_loc_number):
    """
    Average locations that are within radius in km of each other.

    This function takes a list of lat/lon locations as tuples in a list.
    These are locations of local maxima. In this function, locations that are
    within the radius of each other are averaged. To make sure that no
    duplicate locations are returned, the function is recursively called until
    the number of unique locations no longer decreases. The function takes the
    list of lat/lon locations, the radius used to check for duplicates,
    and a unique_loc_number as parameters and returns a list of lat/lon tuples
    of the unique locations. The unique_loc_number is originally set to a very
    large value (99999999) when this function is called, and then this is what
    is compared with for recursive call (eg. is the new number of unique
    locations, say 20, less than 99999999? If yes, then go through
    unique_locations again, but this time unique_loc_number is set to 20.
    Only when the new number of unique locations equals the previous
    unique_loc_number, is the recursion over).

    Parameters:
        max_locations (list): List of lat/lon locations as tuples.
        radius (float): The radius in km.
        unique_loc_number (int): A large integer for comparison.

    Returns:
        list: A list of lat/lon tuples of the unique locations.
    """
    # sort the list by the latitudes, so the lat/lon paris go from south to
    # north
    max_locations.sort(key=lambda x: x[0])

    # convert the radius to its approximate degree equivalent by dividing the
    # radius in km by 111 km. This is because 1 degree is approximately
    # 111 km (at the equator).
    max_distance = radius/111.  # convert to degrees

    # make a tree out of the lat/lon points in the list
    tree = cKDTree(max_locations)
    # make a list to hold the average of closely neighboring points
    point_neighbors_list_avg = []
    first_time = True
    # loop through all the lat/lon max locations and find the points that
    # are within the radius/max_distance of each other
    for max_loc in max_locations:
        # check to see if it's the first time
        if first_time:
            first_time = False
        else:
            # skip points that have just been found as near neighbors
            # so if a point was just in point_neighbors, move on to the next
            # point
            if max_loc in point_neighbors:
                # print("continue")
                continue

        # calculate the distance and indices between the max location lat/lon
        # points
        distances, indices = tree.query(max_loc, len(
            max_locations), p=10, distance_upper_bound=max_distance)
        # p=10 seems to be a good choice
        # the following conditional catches the cases where distances and
        # indices are a float and integer, respectively. In this case, the
        # zip a few lines below won't work because floats and ints aren't
        # interable. Check and see if distance and indices are not iterable
        # and if they're not, make them lists so that they are iterable.
        if not isinstance(distances, Iterable) and \
                not isinstance(indices, Iterable):
            distances = [distances]
            indices = [indices]
        # store the neighboring points that are close to each in a list
        point_neighbors = []
        for index, distance in zip(indices, distances):
            # points that aren't close are listed at a distance of inf so
            # ignore those
            if distance == inf:
                break
            point_neighbors.append(max_locations[index])

        # the points_neighbors list has all the lat/lon points that are close
        # to each other average those points, keep them as a tuple
        # (so a new averaged lat/lon point), and append that to
        # point_neighbors_list_avg
        point_neighbors_list_avg.append(
            tuple(np.mean(point_neighbors, axis=0)))

    # the number of unique locations is the length of the
    # point_neighbors_list_avg list use the list set option here to make sure
    # any duplicates aren't counted
    new_unique_loc_number = len(list(set(point_neighbors_list_avg)))

    # check to see if the number of unique locations is the same as it was the
    # last time through the function. If it is the same, return
    # point_neighbors_list_avg, which is the unique lat/lon locations.
    # If it is not the same, recursively go back through unique_locations to
    # weed out any more locations that are within the radius/max_distance of
    # each other.
    if new_unique_loc_number == unique_loc_number:
        # use the list set option here to make sure any duplicates are
        # not counted
        return list(set(point_neighbors_list_avg))
    else:
        return unique_locations(point_neighbors_list_avg, radius,
                                new_unique_loc_number)


def unique_track_locations(track_object, combined_unique_max_locs,
                           current_time_index, radius):
    """
    Modify the existing track object to ensure unique track locations.

    This function takes a track object, the list of vorticity maximum lat/lon
    locations, the current time index, and the radius in km. The lat/lon
    location of the track object at the current time is found. That lat/lon is
    then compared with all of the lat/lon pairs in the list. If any of the
    lat/lons in the list are within the radius of the track object's lat/lon,
    this lat/lon pair replaces the lat/lon pair in the track object by
    averaging the old and new lat/lon pairs.
    The lat/lon pair in the list that was too close to the track object is
    then removed. The whole point of this function is to make sure no
    duplicate track objects are created by weeding out possible track
    locations that are already represented in existing tracks. This function
    doesn't return anything, it just modifies the existing track object and
    lat/lon list.

    Parameters:
        track_object (object): The track object.
        combined_unique_max_locs (list): The list of vorticity maximum lat/lon
            locations.
        current_time_index (int): The current time index.
        radius (float): The radius in km.

    Returns:
        None
    """
    # get the lat/lon pair for the track object at the current time using
    # current_time_index
    current_latlon_pair_list = []
    for time_index in range(current_time_index,
                            min(current_time_index+1,
                                len(track_object.latlon_list))):
        current_latlon_pair_list.append(track_object.latlon_list[time_index])

    # loop through the lat/lon locations in the combined_unique_max_locs list
    for latlon_loc in list(combined_unique_max_locs):
        for current_latlon_pair in current_latlon_pair_list:
            # get the distance between the track object's lat/lon and the
            # lat/lon pair from the list
            dist_km = great_circle_dist_km(
                current_latlon_pair[1], current_latlon_pair[0], latlon_loc[1],
                latlon_loc[0])
            # check and see if the distance is less than the radius. If it is,
            # replace the track_object lat/lon pair with the average of the
            # existing track_object lat/lon pair and the newly found lat/lon
            # pair, remove the new pair from the combined_unique_max_locs list,
            # and continue to the next pair.
            if dist_km < radius:
                track_object.latlon_list = [
                    ((latlon_loc[0]+current_latlon_pair[0])/2,
                     (latlon_loc[1] + current_latlon_pair[1])/2)
                    if x == current_latlon_pair else x for x in
                    track_object.latlon_list]
                # remove the lat/lon pair from the list
                combined_unique_max_locs.remove(latlon_loc)
                continue
    return


def get_multi_positions(common_object, curve_vort_smooth, rel_vort_smooth,
                        unique_max_locs):
    """
    Generate new unique maximum locations combining curvature and relative
    vorticity data.

    This function combines the existing points with a weighted average.
    The new points are then run through unique_locations to check for
    duplicates.

    Parameters:
        common_object (object): The common object containing region indices.
        curve_vort_smooth (array): Smoothed curvature vorticity array.
        rel_vort_smooth (array): Smoothed relative vorticity array.
        unique_max_locs (list): List of unique maximum locations.

    Returns:
        list: A new list of unique maximum locations.
    """
    # Create a variable list that contains the smoothed curvature vorticity
    # and the relative vorticity
    var_list = [curve_vort_smooth, rel_vort_smooth]
    # Create a list for the new max lat/lon indices
    new_weighted_max_indices_list = []

    # Loop through the two variables in var_list
    for var_number in range(2):
        # Loop through the pressure levels where
        # 0 = 850 hPa, 1 = 700 hPa, and 2 = 600 hPa
        for p_level in range(3):
            # For var_number 0 (curvature vorticity), skip level 1 (700 hPa)
            if var_number == 0 and p_level == 1:
                continue
            # For var_number 1 (relative vorticity), skip level 0 (850 hPa)
            if var_number == 1 and p_level == 0:
                continue
            # Get the local maxima lat/lon indices over Africa and the
            # Atlantic for the var in var_list.
            new_max_indices = peak_local_max(
                var_list[var_number][p_level,
                                     common_object.lat_index_south:
                                     common_object.lat_index_north+1,
                                     common_object.lon_index_west:
                                     common_object.lon_index_east+1
                                     ], min_distance=1)

            # Loop through all of the unique locations of local maxima from
            # unique_max_locs
            # Check to see if the lat/lon pairs in new_max_indices are within
            # the radius of the unique locations
            # Only keep the locations in new_max_indices that are within the
            # radius of the location from unique_max_locs
            for max_loc in unique_max_locs:
                weighted_max_indices = is_maxima(
                    common_object, var_list[var_number][p_level, :, :],
                    max_loc, new_max_indices)
                # Check to make sure there wasn't an empty list coming back
                # from is_maxima
                if weighted_max_indices != 9999999:
                    new_weighted_max_indices_list.append(weighted_max_indices)

    # Flatten new_weighted_max_indices_list
    flattened_indices = [
        item for sublist in new_weighted_max_indices_list for item in sublist]

    # Pass the flattened version of new_weighted_max_indices_list to
    # unique_locations
    if len(flattened_indices) > 1:
        unique_max_locs = unique_locations(flattened_indices, 350., 99999999)
    else:
        unique_max_locs = flattened_indices

    return unique_max_locs


def is_maxima(common_object, var, max_loc, new_max_indices):
    """
    Find the weighted maxima indices close to an existing max_loc location.

    This function takes a list of lat/lon indices that correspond to vorticity
    maxima and checks to see if they are close to an existing lat/lon pair from
    get_starting_targets. If any lat/lon points are close to the existing
    point, they are added to a list, then a weighted average is taken from
    that list and is returned.

    Parameters:
        common_object (object): The common object containing region indices.
        var (array): The variable (curvature or relative vorticity).
        max_loc (tuple): The existing maximum location.
        new_max_indices (list): A list of new lat/lon indices.

    Returns:
        tuple or int: Weighted lat/lon indices if found, 9999999 if no
            indices are found.
    """
    # this is a list for all the lat/lon pairs that are close to the max_loc
    # location
    max_indices_near_max_loc = []
    # loop through all the possible new local maxima locations and find the
    # ones that are close to the max_loc location
    for max_index in new_max_indices:
        # get the distance between teh new max lat/lon index and the existing
        # max_loc need to add lat_index_south and lon_index_west to
        # max_index[0] and max_index[1], respectively, because the max_indices
        # were calculated over a cropped region so the indices will not match
        # the full lat/lon arrays.
        dist_km = great_circle_dist_km(
            common_object.lon[max_index[0]+common_object.lat_index_south,
                              max_index[1]+common_object.lon_index_west],
            common_object.lat[max_index[0]+common_object.lat_index_south,
                              max_index[1]+common_object.lon_index_west],
            max_loc[1], max_loc[0])
        # if the distance is less than the common_object radius, then append
        # that max index to max_indices_near_max_loc
        if dist_km < common_object.radius:
            max_indices_near_max_loc.append(tuple(max_index))

    # check to make sure max_indices_near_max_loc is not an empty list
    # If max_indices_near_max_loc is not empty, use the function
    # get_mass_center to find the weighted averaged of the
    # max_indices_near_max_loc lat/lon pairs.
    # If it is empty, it will skip and set weighted_max_indices to 9999999
    # If max_indices_near_max_loc is not empty, in get_multi_positions,
    # the weighted_max_indices returned here are are then appended to a list.
    # If max_indices_near_max_loc is empty, that append step is skipped.
    if max_indices_near_max_loc:
        weighted_max_indices = get_mass_center(
            common_object, var, max_indices_near_max_loc)
    else:
        weighted_max_indices = 9999999

    return weighted_max_indices


def assign_magnitude(common_object, curve_vort_smooth, rel_vort_smooth,
                     track_object):
    """
    Assign a vorticity magnitude value to an AEW track object.

    This function assigns an AEW track object a vorticity magnitude value
    that corresponds with the last lat/lon location of the track object.

    Parameters:
        common_object (object): The common object containing region indices.
        curve_vort_smooth (array): The smoothed curvature vorticity.
        rel_vort_smooth (array): The smoothed relative vorticity.
        track_object (object): The AEW track object.

    Returns:
        None
    """
    # the length of the track object's latlon_list should be one longer than
    # the magnitude_list since the magnitude hasn't been appended yet
    if len(track_object.latlon_list) - len(track_object.magnitude_list) == 1:
        # get the last lat/lon pair in the latlon_list
        lat_lon_pair = track_object.latlon_list[-1]
        # get the indices for the lat/lon point
        lat_index = (
            np.abs(common_object.lat[:, 0] - lat_lon_pair[0])).argmin()
        lon_index = (
            np.abs(common_object.lon[0, :] - lat_lon_pair[1])).argmin()

        # the magnitude is whatever is largest, curvature or relative voriticy
        # at 850., 700., 600. or 700., 600., respectively
        magnitude = max(curve_vort_smooth[0, lat_index, lon_index],
                        curve_vort_smooth[1, lat_index, lon_index],
                        curve_vort_smooth[2, lat_index, lon_index],
                        rel_vort_smooth[1, lat_index, lon_index],
                        rel_vort_smooth[2, lat_index, lon_index])
        # add the magnitude to the track object
        track_object.add_magnitude(magnitude)
        return
    else:
        return


def filter_tracks(common_object, track_object):
    """
    Filter out AEW tracks that don't meet specific criteria.

    This function catches and filters out any tracks that don't make sense
    for actual AEW tracks (e.g., moving east instead of west). 
    It evaluates various conditions based on the common_object and the
    track_object and returns a filter_result_object indicating 
    whether the track should be filtered out.

    Parameters:
        common_object (object): The common object containing region indices
            and thresholds.
        track_object (object): The track object to be filtered.

    Returns:
        filter_result_object (object): An object containing filtering results
            indicating whether the track should be rejected based on various
            criteria.
    """
    # Create an instance of filter_result to keep track of whether or not a
    # given track_object needs to be filtered out
    filter_result_object = filter_result()

    # Get rid of any tracks that are going east instead of west
    if len(track_object.latlon_list) > 4 and \
        (track_object.latlon_list[-2][1]
         - track_object.latlon_list[-1][1] < 0) and \
        (track_object.latlon_list[-3][1]
         - track_object.latlon_list[-2][1] < 0) and \
        (track_object.latlon_list[-4][1]
         - track_object.latlon_list[-3][1] < 0) and \
        (track_object.latlon_list[-5][1]
         - track_object.latlon_list[-4][1] < 0):
        filter_result_object.reject_track_direction = True
    else:
        filter_result_object.reject_track_direction = False

    # If the track lasts more than 6 times and the intensities are too weak
    # (less than min_threshold) for more than three times, then return True
    # and the track object is removed and added to a finished track list
    if len(track_object.latlon_list) > 6 and \
            sum(1 for x in track_object.magnitude_list
                if x < common_object.min_threshold) > 3:
        filter_result_object.magnitude_finish_track = True
    else:
        filter_result_object.magnitude_finish_track = False

    # If the track goes too far north or south, remove it and add it to a
    # finished track list
    if track_object.latlon_list[-1][0] > 40. \
            or track_object.latlon_list[-1][0] < -5.:
        filter_result_object.latitude_finish_track = True
    else:
        filter_result_object.latitude_finish_track = False

    return filter_result_object


def circle_avg_m_point(common_object, var, lat_lon_pair):
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
    cos_lat = np.cos(np.radians(common_object.lat))

    R = 6371.  # Earth radius in km
    # Get the number of gridpoints equivalent to the radius being used for
    # the smoothing. To convert the smoothing radius in km to gridpoints,
    # multiply the radius (in km) by the total number of longitude
    # gridpoints = var.shape[2] for the whole domain divided by the degrees
    # of longitude in the domain divided by 360 times the circumference of
    # the Earth = 2piR. The degrees of longitude/360 * circumference is to
    # scale the circumference to account for non-global data. This is also a
    # rough approximation since we're not quite at the equator.
    #
    # radius_gridpts = radius (in km) *
    #            (longitude gridpoints / scaled circumference of Earth (in km))
    #
    # Make radius_gridpts an int so it can be used as a loop index later.
    radius_gridpts = int(common_object.radius*(common_object.lat.shape[1]/(
        (common_object.total_lon_degrees/360)*2*np.pi*R)))

    # create a copy of the var array
    smoothed_var = np.copy(var)

    # get the indices for the lat/lon pairs of the maxima
    lat_index_maxima = (
        np.abs(common_object.lat[:, 0] - lat_lon_pair[0])).argmin()
    lon_index_maxima = (
        np.abs(common_object.lon[0, :] - lat_lon_pair[1])).argmin()
    # take circle average
    tempv = 0.0
    divider = 0.0
    # work wayup circle
    for radius_index in range(-radius_gridpts, radius_gridpts+1):

        # make sure we're not goint out of bounds, and if we are go to the
        # next iteration of the loop
        if (lat_index_maxima+radius_index) < 0 \
            or (lat_index_maxima+radius_index) \
                > (common_object.lat.shape[1]-1):
            continue

        lat1 = common_object.lat[lat_index_maxima,
                                 lon_index_maxima]  # center of circle
        # vertical distance from circle center
        lat2 = common_object.lat[lat_index_maxima +
                                 radius_index, lon_index_maxima]
        # make sure that lat2, which has the radius added, doesn't go off the
        # grid (either off the top or the bottom)

        # need to switch all angles from degrees to radians (haversine trig)
        angle_rad = np.arccos(
            -((np.sin(np.radians(lat1))*np.sin(np.radians(lat2)))
                - np.cos(common_object.radius/R))/(np.cos(np.radians(lat1))
                                                   * np.cos(
                                                       np.radians(lat2))))

        # convert angle from radians to degrees and then from degrees to
        # gridpoints divide angle_rad by pi/180 = 0.0174533 to convert radians
        # to degrees multiply by lat.shape[1]/360 which is the lon gridpoints
        # over the total 360 degrees around the globe the conversion to
        # gridpoints comes from (degrees)*(gridpoints/degrees) = gridpoints
        # lon_gridpts defines the longitudinal grid points for each lat
        lon_gridpts = int((angle_rad/0.0174533) *
                          (common_object.lat.shape[1]/360.))

        # work across circle
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


def advect_tracks(common_object, u_3d, v_3d, track_object, times, time_index):
    """
    Advects a track object's last lat/lon point using the average of the
    zonal (u) and meridional (v) wind  between 850 and 600 hPa to obtain the
    next lat/lon point in time.

    Parameters:
        common_object (object): The common object containing region indices
            and other common parameters.
        u_3d (numpy.ndarray): The zonal wind component (u) at different
            pressure levels.
        v_3d (numpy.ndarray): The meridional wind component (v) at different
            pressure levels.
        track_object (object): The track object to be advected.
        times (numpy.ndarray): Array containing time values.
        time_index (int): The current time index.

    Returns:
        None
    """
    # get the last lat/lon tuple in the track object's latlon_list
    # We want the last lat/lon tuple because that will be the last one in time
    # and we want to advect the last lat/lon point to get the
    # next point in time.
    lat_lon_pair = track_object.latlon_list[-1]

    # calculate the u and v wind averaged over the two steering levels;
    # the steering levels are 850 and 600 hPa
    u_2d = (u_3d[0, :, :] + u_3d[2, :, :]) / \
        2.  # take average of the 850 (index 0) and 600 (index 2) hPa levels
    v_2d = (v_3d[0, :, :] + v_3d[2, :, :]) / \
        2.  # take average of the 850 (index 0) and 600 (index 2) hPa levels

    # smooth the u and v arrays at the lat/lon points that have been
    # identified as unique maxima the lat/lon pairs come in through
    # track_locations as tuples in a list. The smoothing is done using the
    # python circle_avg_m_point function.
    u_2d_smooth = circle_avg_m_point(common_object, u_2d, lat_lon_pair)
    v_2d_smooth = circle_avg_m_point(common_object, v_2d, lat_lon_pair)

    # find the indices for the lat/lon pairs
    lat_index = (np.abs(common_object.lat[:, 0] - lat_lon_pair[0])).argmin()
    lon_index = (np.abs(common_object.lon[0, :] - lat_lon_pair[1])).argmin()

    # get new lat/lon values for the next time step by advecting the existing
    # point using u and v multiply dt (in hours) by 60*60 to get seconds;
    # 111120. is the approximate meters in one degree on Earth
    # the *60*60*dt / 111120 converts the m/s from u and v into degrees
    new_lat_value = lat_lon_pair[0] + (
        (v_2d_smooth[lat_index, lon_index] * 60. * 60. * common_object.dt) /
        111120.)
    new_lon_value = lat_lon_pair[1] + (
        (u_2d_smooth[lat_index, lon_index] * 60. * 60. *
         common_object.dt) / 111120.) * np.cos(np.radians(lat_lon_pair[0]))
    # to switch to radians

    # make sure that the next time step is actually within our timeframe by
    # checking to see if time_index+1 (the next time)
    # is less than times.shape[0], otherwise there wil be an index error
    if time_index + 1 < times.shape[0]:
        # add the new lat/lon pair to the end of the track_object's latlon_list
        track_object.add_latlon((new_lat_value, new_lon_value))
        # add the next time to the end of the track_objects time list
        track_object.add_time(times[time_index + 1])

    return
