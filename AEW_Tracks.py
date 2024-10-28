# %%
from __future__ import division  # makes division not round with integers
import os
from tqdm import tqdm
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from numba import jit
import pandas as pd
import xarray as xr
import Tracking_functions as track
import Pull_data as pull
import wrf as wrf
import utilities as util
import warnings
warnings.filterwarnings("ignore")


"""
AEW_Tracks.py - Main program for the African Easterly Wave (AEW) tracking algorithm.

This script tracks AEW events based on specified model types, scenario types, 
and years, which are parsed from the command line. The tracking is typically 
configured to run from May to October, but this timeframe can be modified in 
the main function. 

Upon execution, the program generates AEW track objects and produces a figure 
visualizing all tracks found over the specified time period.

Usage:
    To run the program, use the following command:
    python AEW_Tracks.py --model 'ERA5' --year '2010'
    python AEW_Tracks.py --year '2010'

Command Line Arguments:
    --model str       : The model type to be used for tracking (e.g., 'ERA5').
    --year int        : The year for which AEW tracks are to be analyzed.

Output:
    - AEW track object
    - Figure displaying all tracks found during the specified period.
"""


# Define the path for the data directory relative to the script directory
path_data = os.path.join(os.path.dirname(__file__), 'Data/')
path_fig = os.path.join(os.path.dirname(__file__), 'Figures/')

# Create the directories (if it doesn't exist)
util.create_path(path_data)
util.create_path(path_fig)


class Common_track_data:
    """
    Class containing common information used for tracking.
    """

    def __init__(self, model=None, lat=None, lon=None, dt=None,
                 min_threshold=None, radius=None):
        self.model = model
        self.lat = lat
        self.lon = lon
        self.lat_index_north = None
        self.lat_index_south = None
        self.lon_index_east = None
        self.lon_index_west = None
        self.lat_index_north_crop = None
        self.lat_index_south_crop = None
        self.lon_index_east_crop = None
        self.lon_index_west_crop = None
        self.total_lon_degrees = None
        self.dt = dt
        self.min_threshold = min_threshold
        self.radius = radius

    def add_model(self, model_type):
        self.model = model_type

    def __repr__(self):
        lat = self.lat[:, 0]
        lon = self.lon[0, :]
        text = f'Model: {self.model}\n'\
            f'Minimum threshold (CV): {self.min_threshold}\n'\
            f'Radius [km]: {self.radius}\n'\
            f'dt [hours]: {self.dt}\n'\
            f'Spatial resolution: {self.res}°\n'\
            f'lat [dim: {self.lat.shape}]: {lat.min()} to {lat.max()}\n'\
            f'lon [dim: {self.lon.shape}]: {lon.min()} to {lon.max()}'
        return text


class AEW_track:
    """
    Class representing an African Easterly Wave (AEW) track.
    Each instance/object of AEW_track represents an AEW track
    and contains corresponding latitude/longitude points and magnitudes of the
    vorticity at those points.

    Attributes:
        latlon_list (list): List of latitude/longitude tuples representing the
            track's path.
        magnitude_list (list): List of magnitudes of vorticity at each
            latitude/longitude point.
        time_list (list): List of timestamps for each location in the track.
        id (int): Unique identifier for the track.
    """

    # Class-level counter for assigning unique IDs
    id_counter = 0

    def __init__(self):
        self.latlon_list = []  # List for lat/lon tuples
        self.magnitude_list = []  # List for magnitudes of vorticity
        self.time_list = []  # List for timestamps

        # Assign a unique ID to the track and increment the class-level counter
        self.id = AEW_track.id_counter
        AEW_track.id_counter += 1

    def add_latlon(self, latlon):
        self.latlon_list.append(latlon)

    def remove_latlon(self, latlon):
        self.latlon_list.remove(latlon)

    def add_magnitude(self, magnitude):
        self.magnitude_list.append(magnitude)

    def add_time(self, time):
        self.time_list.append(time)

    def __repr__(self):
        return f"AEW_track(id={self.id}, latlon_list={self.latlon_list}, " \
               f"time_list={self.time_list})"


def plot_points_map(aew_track_list, lat_step=15, lon_step=30,
                    map_resolution=50, point_size_start=15, point_size_end=40,
                    cmap='viridis', title=None):
    """
    Generate a figure with connected latitude/longitude points overlaid on a
    map to visualize AEW tracks. The point sizes increase linearly for each track.

    Parameters:
        aew_track_list (list): List of AEW_track objects representing AEW
            tracks to be plotted.
        lat_step (int): Latitude step for gridlines and ticks.
        lon_step (int): Longitude step for gridlines and ticks.
        map_resolution (int): Map resolution for coastlines.
        point_size_start (int): Starting size of the scatter points.
        point_size_end (int): Ending size of the scatter points.

    Returns:
        None
    """
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    # Plot information
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    
    if title is not None:
        plt.title(title, loc='left')

    # Create color palette
    colors = sns.color_palette(cmap, len(aew_track_list))

    for ii, aew_track in enumerate(aew_track_list):
        track_color = colors[ii]

        # Unzip list of lat/lon tuples into a list of two lists.
        track_latlons = [list(t) for t in zip(*aew_track.latlon_list)]
        track_lats = track_latlons[0]
        track_lons = track_latlons[1]

        # Create an array of linearly increasing point sizes based on track length
        track_length = len(track_lats)
        point_sizes = np.linspace(
            point_size_start, point_size_end, track_length)

        # Scatter points with increasing size
        ax.scatter(track_lons, track_lats, color=track_color, s=point_sizes,
                   linewidth=0.25, marker='o', transform=ccrs.PlateCarree())
        # Plot lines connecting the points
        ax.plot(track_lons, track_lats, color=track_color, linewidth=1,
                label=aew_track.time_list[0].strftime('%Y-%m-%d_%H'),
                transform=ccrs.PlateCarree())

        del track_latlons
        del track_lats
        del track_lons

    ax.set_yticks(np.arange(-90, 91, lat_step), crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-180, 180, lon_step), crs=ccrs.PlateCarree())

    # Set tick labels and gridlines
    ax.tick_params(axis='both', which='major', labelsize=12, color="#434343")
    lon_formatter = LongitudeFormatter(
        zero_direction_label=True, number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_axisbelow(False)

    # Add gridlines
    ax.grid(which='major', linestyle='--', linewidth='0.6',
            color='gray', alpha=0.8, zorder=9)

    # Add coastlines
    ax.coastlines(resolution=f'{map_resolution}m',
                  color='k', alpha=0.78, lw=0.6, zorder=10)

    ax.set_extent([-90., 30., -6., 30.],
                  ccrs.PlateCarree(central_longitude=0))
    ax.set_aspect('auto')

    return fig


def plot_potential_loc_map(var_values, common_object, value_limit,
                           treshold, points=None, aew_list=None,
                           label=None,
                           lat_step=15, lon_step=30,
                           map_resolution=50,
                           cmap='RdBu_r', title=None):
    """
    Generate a figure with connected latitude/longitude points overlaid on a
    map to visualize AEW tracks. The point sizes increase linearly for each track.

    Parameters:
        aew_track_list (list): List of AEW_track objects representing AEW
            tracks to be plotted.
        lat_step (int): Latitude step for gridlines and ticks.
        lon_step (int): Longitude step for gridlines and ticks.
        map_resolution (int): Map resolution for coastlines.
        point_size_start (int): Starting size of the scatter points.
        point_size_end (int): Ending size of the scatter points.

    Returns:
        None
    """
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.patches import Circle

    lat = common_object.lat
    lon = common_object.lon

    # Plot information
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())

    # Define the contour levels for the temperature plot
    levels = np.linspace(-value_limit, value_limit, 21)

    # Define the colormap for the plot
    cmap = sns.color_palette(cmap, as_cmap=True)

    # # Plot the temperature data for the current time slice
    cs = ax.contourf(lon, lat, var_values[:, :], levels,
                     cmap=cmap, extend='both', transform=ccrs.PlateCarree())

    ax.contour(lon, lat, var_values[:, :], [treshold],
               linewidths=0.5, colors='black', transform=ccrs.PlateCarree())

    if points is not None:
        for ii, (y, x) in enumerate(points):
            # Scatter points with increasing size
            ax.scatter(x, y, color='magenta', linewidth=0.25, marker='o', s=45,
                       transform=ccrs.PlateCarree())
    if title is not None:
        plt.title(title, loc='left')

    if aew_list is not None:
        # Create color palette
        # colors = sns.color_palette(cmap, len(aew_list))
        colors = 'k'

        for ii, aew_track in enumerate(aew_list):
            track_color = colors

            # Unzip list of lat/lon tuples into a list of two lists.
            track_latlons = [list(t) for t in zip(*aew_track.latlon_list)]
            track_lats = track_latlons[0]
            track_lons = track_latlons[1]

            # Create an array of linearly increasing point sizes based on track length
            track_length = len(track_lats)
            # point_sizes = np.linspace(
            #     point_size_start, point_size_end, track_length)

            # # Scatter points with increasing size
            ax.scatter(track_lons, track_lats, color='k', s=25,
                       linewidth=0.25, marker='o', transform=ccrs.PlateCarree())
            # Plot lines connecting the points
            ax.plot(track_lons, track_lats, color=track_color, linewidth=1,
                    label=aew_track.time_list[0].strftime('%Y-%m-%d_%H'),
                    transform=ccrs.PlateCarree())

            center = (track_lons[-1], track_lats[-1])
            radius = common_object.radius / 111
            circle = Circle(center, radius, fc='None', ec='r', )
            ax.add_patch(circle)

            del track_latlons
            del track_lats
            del track_lons

    ax.set_yticks(np.arange(-90, 91, lat_step), crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-180, 180, lon_step), crs=ccrs.PlateCarree())

    # Set tick labels and gridlines
    ax.tick_params(axis='both', which='major', labelsize=12, color="#434343")
    lon_formatter = LongitudeFormatter(
        zero_direction_label=True, number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_axisbelow(False)

    # Add gridlines
    ax.grid(which='major', linestyle='--', linewidth='0.6',
            color='gray', alpha=0.8, zorder=9)

    # Add coastlines
    ax.coastlines(resolution=f'{map_resolution}m',
                  color='k', alpha=0.78, lw=0.6, zorder=10)

    ax.set_extent([-90., 30., -6., 30.],
                  ccrs.PlateCarree(central_longitude=0))
    # ax.set_aspect('auto')

    plt.colorbar(cs, label=label)

    return fig


def load_or_fetch_data(path_data_exp, file_exp_name, common_object,
                       times, time_index, smooth_field):
    """
    Loads data from a file if it exists,
    otherwise fetches it and saves it to the file.

    Args:
        path_data_exp: Path to the experiment data directory.
        file_exp_name: Name of the experiment data file.
        common_object: Object used for fetching data
        (details depend on your implementation).
        times: List of time points.
        time_index: Index of the desired time point.
        smooth_field: Boolean indicating whether to smooth the field data.

    Returns:
        u_2d, v_2d, rel_vort_3d, curve_vort_3d: Loaded or fetched data variables.
    """

    file_path = f'{path_data_exp}{file_exp_name}'

    if util.file_in_folder(path_data_exp, file_exp_name):
        # Load data from file
        try:
            with open(file_path, 'rb') as file:
                u_2d, v_2d, rel_vort_3d, curve_vort_3d = pickle.load(file)
        except (pickle.UnpicklingError, FileNotFoundError, OSError) as e:
            print(f"Error loading data from file: {e}")
            # Handle the error appropriately, e.g., fetch data or return default values

    else:
        # Fetch data and save to file
        u_2d, v_2d, rel_vort_3d, curve_vort_3d = pull.get_variables_par(
            common_object, times[time_index], smooth_field=smooth_field
        )

        try:
            with open(file_path, 'wb') as file:
                pickle.dump([u_2d, v_2d, rel_vort_3d, curve_vort_3d], file)
        except (pickle.PicklingError, OSError) as e:
            print(f"Error saving data to file: {e}")
            # Handle the error appropriately

    return u_2d, v_2d, rel_vort_3d, curve_vort_3d


def process_clean_tracks(path_data_exp, path_fig_out, times, finished_AEW_tracks_list,
                         common_object, min_threshold, save_fig=True, smooth_field=True):
    """
    Function to process and plot AEW tracks over a range of time steps.

    Parameters:
        path_data_exp (str): Path to the experimental data.
        path_fig_out (str): Path to save output figures.
        times (list): List of time points.
        finished_AEW_tracks_list (list): List of finished AEW tracks.
        common_object (object): Object containing common track data.
        min_threshold (float): Threshold value for identifying AEWs.
        save_fig (bool): Whether to save the figures (default: True).
        smooth_field (bool): Whether to smooth the data fields (default: True).

    Returns:
        None
    """
    path_exp_clean_tracks = f'{path_fig_out}clean_tracks/'
    util.create_path(path_exp_clean_tracks)

    for time_index, time_i in tqdm(enumerate(times), desc="Processing dates clean AEW",
                                   total=len(times), leave=True):

        date_ii = time_i.strftime('%Y-%m-%d %H:%M:%S')
        print(f"{'-'*66}")
        print(f"{date_ii:^66}")
        print(f"{'-'*66}")

        # Load data for the current time step
        file_exp_name = f"{time_i.strftime('%Y-%m-%d_%H:%M:%S')}.obj"
        u_2d, v_2d, rel_vort_3d, curve_vort_3d = load_or_fetch_data(
            path_data_exp, file_exp_name, common_object, times,
            time_index, smooth_field)

        active_aew_list = []
        for track_object in finished_AEW_tracks_list:
            try:
                current_time_index = track_object.time_list.index(
                    times[time_index])
                # Create a new AEW_track object for the cut track
                new_aew_track = AEW_track()

                # Copy latitude/longitude, magnitude, and time data up to current time index
                new_aew_track.latlon_list = track_object.latlon_list[:current_time_index + 1]
                new_aew_track.magnitude_list = track_object.magnitude_list[:current_time_index + 1]
                new_aew_track.time_list = track_object.time_list[:current_time_index + 1]

                # Append the new "cut" AEW track to the active list
                active_aew_list.append(new_aew_track)

            except ValueError:
                continue

        # Plot and save the results for the current time step
        fig = plot_potential_loc_map(curve_vort_3d, common_object, aew_list=active_aew_list,
                                     treshold=min_threshold, value_limit=0.000_02,
                                     label='Curvature Vorticity', title=date_ii)

        if save_fig:
            fig.savefig(f'{path_exp_clean_tracks}{time_index:04}.png',
                        dpi=200, bbox_inches='tight', transparent=False, facecolor='white')
            plt.close('all')


def save_aew_tracks(path_data_out, file_name, AEW_tracks_list):
    """
    Save the finished AEW tracks list to a file using pickle.

    Parameters:
        path_data_out (str): The directory path where the output file will
        be saved.
        file_name (str): name of the output file.
        AEW_tracks_list (list): The list of finished AEW track objects to save.

    Returns:
        None
    """

    # Open the file in write-binary mode and save the data using pickle
    with open(f"{path_data_out}{file_name}", 'wb') as tracks_file:
        pickle.dump(AEW_tracks_list, tracks_file)

    print(f"AEW tracks saved to {file_name}")


def print_dates(start_date, end_date):
    """
    Print the starting and finishing dates in a beautiful format.

    Parameters:
        start_date (datetime): The starting date.
        end_date (datetime): The finishing date.

    Returns:
        None
    """
    # Format the dates into a more readable form
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Print the dates in a beautiful format
    print(f"{'='*66}")
    print(f"Start Date: {start_str}")
    print(f"End Date:   {end_str}")
    print(f"{'='*66}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AEW Tracker.')
    parser.add_argument('--year', dest='year',
                        help='Get the year of interest', required=True)
    args = parser.parse_args()

    # set the year that is parsed from the command line
    year = int(args.year)
    
    # year = 1996    
    model = 'ERA5'
    

    # Define the output path for experiment results (AEW tracks)
    path_data_out = os.path.join(os.path.dirname(__file__), 'experiments/')

    # Flags to control whether figures should be plotted and saved
    plot_fig = False
    save_fig = False

    # Experiment flag (indicates this is an experimental run)
    experiment = False

    # Define paths for experiment-specific data and output figures
    path_data_exp = os.path.join(os.path.dirname(__file__), 'data_exp/')
    
    if experiment:
        path_fig = os.path.join(os.path.dirname(__file__), 'Fig_exp/')
        util.create_path(path_fig)
        
        path_fig = os.path.join(path_fig, 'Exp1/')
        util.create_path(path_fig)
        

    # Set the radius (in kilometers) used for smoothing fields and
    # identifying points that belong to the same AEW track
    radius_km = 600  # km

    # Set the threshold value for identifying AEWs based on curvature vorticity
    # Based on the study by Brannan and Martin (2019)
    min_threshold = 0.000_002

    # Restrict the tracking of AEW initiation to longitudes east of 20°W
    # This ensures only perturbations east of -20 longitude are tracked
    limit_long_init = True
    min_lon = -20

    # Set a flag to indicate if the curvature vorticity fields should be smoothed
    smooth_field = True

    # Create an instance of the Common_track_data class to hold shared
    # information needed for AEW tracking (such as model type, lat/lon, etc.)
    common_object = Common_track_data(model=model,
                                      min_threshold=min_threshold,
                                      radius=radius_km)

    # Fetch the common track data (such as latitude, longitude, and time steps)
    # and assign them to the common_object
    pull.get_common_track_data(common_object)

    # Set the start and end dates for tracking AEWs (from May to November)
    start_date = pd.Timestamp(year, 5, 1)
    end_date = pd.Timestamp(year, 11, 1)

    # Print the start and end dates for the tracking period
    print_dates(start_date, end_date)

    # Generate the list of timestamps for the tracking period with a frequency
    # based on the time step defined in common_object (e.g., every 6 hours)
    times = pd.date_range(start=start_date, end=end_date,
                          freq=f'{common_object.dt}h')

    # Initialize lists to hold the AEW tracks for the current time step
    # and finished AEW tracks that have been fully processed
    AEW_tracks_list = []
    finished_AEW_tracks_list = []

    for time_index, time_i in tqdm(enumerate(times),
                                   desc="Processing dates", total=len(times),
                                   leave=True):

        date_ii = time_i.strftime('%Y-%m-%d %H:%M:%S')
        print(f"{'-'*66}")
        print(f"{date_ii:^66}")
        print(f"{'-'*66}")

        # Fetch variables for the current time step from experimentdata or directly
        if experiment:
            file_exp_name = f"{time_i.strftime('%Y-%m-%d_%H:%M:%S')}.obj"
            u_2d, v_2d, rel_vort_smooth, curve_vort_smooth = load_or_fetch_data(
                path_data_exp, file_exp_name, common_object, times,
                time_index, smooth_field)
        else:
            u_2d, v_2d, rel_vort_smooth, curve_vort_smooth = \
                pull.get_variables_par(
                    common_object, times[time_index], level=700,
                    smooth_field=True)

        # Skip this time step if data is missing
        if u_2d is None:
            continue

        # Raise an error if curvature vorticity data is missing
        if curve_vort_smooth is None:
            raise ValueError('Error: Curvature Vorticity data is missing')

        # Identify new starting points (vorticity maxima) at the 700 hPa level
        unique_max_locs = track.get_starting_targets(common_object,
                                                     curve_vort_smooth)

        # fig = plot_potential_loc_map(curve_vort_smooth, common_object,
        #                              treshold=min_threshold,
        #                              value_limit=0.000_02,
        #                              label='Curvature Vorticity',
        #                              points=unique_max_locs)

        # Update track locations if the list of existing AEW tracks is not empty
        if AEW_tracks_list:
            for track_object in AEW_tracks_list:
                # get the index of current time from the time list associated
                # with the track object
                current_time_index = track_object.time_list.index(
                    times[time_index])
                # Ensure new locations do not overlap with existing tracks
                track.unique_track_locations_2(common_object,
                                               track_object,
                                               unique_max_locs,
                                               current_time_index,
                                               curve_vort_smooth)
                # track.unique_track_locations(
                #     track_object, unique_max_locs,
                #     current_time_index, common_object.radius)

        # Remove duplicate locations by averaging those within a set radius
        if len(unique_max_locs) > 1:
            unique_max_locs = track.unique_locations(
                unique_max_locs, common_object.radius, 99999999)

        # If unique_max_locs (local vorticity maxima) isn't empty,
        # create new AEW track objects
        if unique_max_locs:  # Proceed only if the list isn't empty
            for lat_lon_pair in unique_max_locs:
                lat, lon = lat_lon_pair

                # Ensure the lat/lon pair contains valid numbers and
                # is within data bounds
                if np.all(np.isreal(lat_lon_pair)) and track.is_in_data(
                        lat, lon,
                        common_object.lat_min, common_object.lat_max,
                        common_object.lon_min, common_object.lon_max):

                    # Check if the point is near other AEWs
                    if AEW_tracks_list:
                        dist_km = np.full(len(AEW_tracks_list), 9999.0)
                        for i, track_object in enumerate(AEW_tracks_list):
                            try:
                                # Find the index for the current time
                                # in the AEW track object
                                current_time_index = track_object.time_list.index(
                                    times[time_index])
                                aew_current_pos = track_object.latlon_list[
                                    current_time_index]

                                # Calculate the distance between the new point
                                # and the current AEW point
                                dist_km[i] = track.great_circle_dist_km(
                                    lat, lon,
                                    aew_current_pos[0],
                                    aew_current_pos[1])
                            except (IndexError, ValueError):
                                continue

                        # Skip this location if it is within the specified
                        # radius of another AEW
                        if np.any(dist_km <= common_object.radius):
                            continue
                    # Skip if the longitude is below the minimum
                    # allowed longitude
                    if limit_long_init and lat_lon_pair[1] < min_lon:
                        continue

                    # Create a new AEW track, add lat/lon and time, and
                    # append to AEW_tracks_list
                    aew_track = AEW_track()
                    aew_track.add_latlon(lat_lon_pair)
                    aew_track.add_time(times[time_index])
                    AEW_tracks_list.append(aew_track)
                    del aew_track

        # Loop through existing AEW tracks, assign magnitudes,
        # filter, and advect them
        for track_object in list(AEW_tracks_list[::-1]):

            if track_object not in AEW_tracks_list:
                continue
            # Assign a magnitude from vorticity to each lat/lon point
            track.assign_magnitude(common_object, curve_vort_smooth,
                                   track_object)

            # Filter out non-AEW tracks based on specific conditions
            filter_result = track.filter_tracks(common_object, track_object,
                                                AEW_tracks_list)

            if filter_result.reject_track_direction or \
                    filter_result.reject_due_to_proximity:
                AEW_tracks_list.remove(track_object)
                continue

            # Move completed tracks to the finished list
            elif filter_result.magnitude_finish_track \
                    or filter_result.latitude_finish_track:
                finished_AEW_tracks_list.append(track_object)
                AEW_tracks_list.remove(track_object)
                continue

            # If no rejection, advect the tracks using wind data
            track.advect_tracks(common_object, u_2d, v_2d,
                                track_object, times, time_index)

        # Print the number of active and finished AEW tracks
        print("\t\tLength of AEW_tracks_list: ", len(AEW_tracks_list))
        print("\t\tLength of finished_AEW_tracks_list: ",
              len(finished_AEW_tracks_list))

        # Plot and save the results if the respective flags are enabled
        if plot_fig:
            fig = plot_potential_loc_map(curve_vort_smooth,
                                         common_object,
                                         points=unique_max_locs,
                                         aew_list=AEW_tracks_list,
                                         treshold=min_threshold,
                                         value_limit=0.000_02,
                                         label='Curvature Vorticity',
                                         title=date_ii)

            if save_fig:
                fig.savefig(f'{path_fig}{time_index:04}.png',
                            dpi=200, bbox_inches='tight',
                            transparent=False, facecolor='white')

                plt.close('all')

    # Combine finished and active tracks, ensuring no duplicates
    finished_AEW_tracks_list = list(
        set(AEW_tracks_list + finished_AEW_tracks_list))
    print("\t\tTotal number of AEW tracks: ", len(finished_AEW_tracks_list))

    # Filter out AEW tracks that do not meet certain criteria, such as
    # insufficient duration, insufficient distance traveled,
    # or incorrect location.
    discarted = []
    for aew_track in list(finished_AEW_tracks_list):
        # Remove tracks that lasted less than two days (i.e., fewer than
        # 48 hours / dt time steps).
        if len(aew_track.latlon_list) < ((48/common_object.dt)+1):
            discarted.append(aew_track)
            finished_AEW_tracks_list.remove(aew_track)
            continue

        # Separate the latitude/longitude pairs of the track into a numpy array
        # for easier computation.
        track_latlons = np.array([list(t)
                                  for t in zip(*aew_track.latlon_list)])

        # If the total longitudinal distance covered by the track is less than
        # 15 degrees, discard the track as it is too short.
        if np.abs(track_latlons[1][0] - track_latlons[1][-1]) < 15:
            discarted.append(aew_track)
            finished_AEW_tracks_list.remove(aew_track)
            continue

        # Ensure that the track has moved sufficiently far west. Using a
        # conservative estimate of 2.25 degrees/day (based on AEW movement of
        # ~250 km/day), calculate if the track covered enough longitudinal
        # distance.
        if abs(aew_track.latlon_list[0][1] - aew_track.latlon_list[-1][1])\
                < (len(aew_track.latlon_list)/(24./common_object.dt))*2.25:
            discarted.append(aew_track)
            finished_AEW_tracks_list.remove(aew_track)
            continue

        # If the track remains too far south (maximum latitude is less than 5°N),
        # discard the track, as it does not exhibit typical AEW movement.
        if np.amax(track_latlons[0]) < 5:
            # print("\t\t\t\ttoo far south")
            discarted.append(aew_track)
            finished_AEW_tracks_list.remove(aew_track)
            continue

    # Print the total number of valid AEW tracks remaining after filtering.
    print(f"{'='*66}")
    print(f"\t\tTracking Period: {start_date.strftime('%Y-%m-%d')} to "
          f"{end_date.strftime('%Y-%m-%d')}")
    print(
        f"\t\tTotal number of valid AEW tracks: {len(finished_AEW_tracks_list)}")
    print(f"{'='*66}")

    date_name = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    name_root = f'AEW_tracks_{radius_km}km_{date_name}'
    
    date_name_title = f"{start_date.strftime('%Y-%m-%d')} to "\
        f"{end_date.strftime('%Y-%m-%d')}"
    fig_title = f'{model} AEW: {date_name_title}'
    
    fig = plot_points_map(finished_AEW_tracks_list, title=fig_title)
    fig.savefig(f'{path_fig}{name_root}.png',
                dpi=200, bbox_inches='tight',
                transparent=False, facecolor='white')
    plt.close('all')

    # Save tracks to file to a prickel object    
    name_file_out = f"{model}_{name_root}.obj"
    save_aew_tracks(path_data, name_file_out, finished_AEW_tracks_list)

    if experiment and save_fig:
        process_clean_tracks(path_data_exp, path_fig, times,
                             finished_AEW_tracks_list,
                             common_object, min_threshold,
                             save_fig=save_fig,
                             smooth_field=True)
        print(f"{'='*66}")
        print("\t\tTotal number of AEW tracks: ", len(finished_AEW_tracks_list))
        print(f"{'='*66}")


# %%
