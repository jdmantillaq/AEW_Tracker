
# AEW_Tracker

This project contains a tracking algorithm for African Easterly Waves (AEWs) using atmospheric data from sources like ERA5. The main goal of this project is to analyze AEW tracks based on wind and vorticity data, with options to visualize and store the results.


This repository is a fork from [github.com/mehtut/AEW_Tracker](https://github.com/mehtut/AEW_Tracker). All credits for the original project go to her.

## Project Structure

    AEW_Tracker/
    │
    ├── AEW_Tracks.py         # Main script for running the AEW tracking algorithm
    ├── Tracking_functions.py # Functions for identifying and correcting AEW tracks
    ├── Pull_data.py          # Functions for pulling data from ERA5 and other sources
    ├── utilities.py          # Utility functions for file management and directories
    ├── Data/                 # Folder for storing input data
    ├── Figures/              # Folder for storing output figures
    └── README.md             # Project overview and instructions
    

## Usage

1. Tracking AEWs: The main script AEW_Tracks.py runs the algorithm for tracking AEWs over a specified time period (May-October).

2. Visualization: After tracking AEWs, the script generates visualizations showing the paths of AEWs on a map, using the plot_points_map function from AEW_Tracks.py.

3. Data Fetching: The Pull_data.py script handles loading and formatting ERA5 or WRF data for analysis, converting them into a common format for the AEW tracker to process.


### Example:

    python AEW_Tracks.py --model 'ERA5' --year '2020'


## Requirements

The AEW Tracker requires the following dependencies, which can be installed using the provided `requirements.txt` file:

`conda create --name aewtrack --file requirements.txt`

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it according to the terms of this license.
