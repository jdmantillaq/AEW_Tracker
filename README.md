
# AEW_Tracker

The African Easterly Wave (AEW) Tracker is a program designed to identify and track AEWs in atmospheric data. AEWs are important meteorological phenomena that can lead to the formation of tropical cyclones and influence weather patterns over large regions, particularly in the Atlantic basin.

This repository is a fork from [github.com/mehtut/AEW_Tracker](https://github.com/mehtut/AEW_Tracker). All credits for the original project go to her.

## Description

The AEW Tracker algorithm consists of several components:

-   **AEW_Tracks.py**: This is the main program that executes the tracking algorithm. It takes input parameters such as the model type (WRF, CAM5, or ERA5), scenario type (Historical, late_century, Plus30), and the year to analyze. It parses these arguments from the command line and orchestrates the tracking process.
    
-   **C_circle_functions.c**: This C program contains the circular smoothing algorithm used for data processing. It needs to be compiled into a shared library (`C_circle_functions.so`) before running the main program.
    
-   **Pull_data.py**: This script handles the retrieval of data required for AEW tracking. It may need adjustments to the data locations depending on where the data is stored.

-   **Tracking_functions.py**: This file contains all the functions related to the identification and tracking algorithms of the AEW. It includes functions for smoothing, finding starting points, combining potential locations, filtering tracks, advecting tracks, and more.
    

## Usage

To run the AEW Tracker, use the following command format:

`python AEW_Tracks.py --model 'WRF' --scenario 'late_century' --year '2010'` 

Make sure to adjust the model type, scenario type, and year according to your analysis needs.

Before running the tracker, compile the C program using the provided compilation command. Additionally, ensure that the data locations in `Pull_data.py` are correct before executing the main program.

## Requirements

The AEW Tracker requires the following dependencies, which can be installed using the provided `requirements.txt` file:

`conda create --name aewtrack --file requirements.txt` 

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it according to the terms of this license.