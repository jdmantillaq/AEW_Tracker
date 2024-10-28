def create_path(path):
    """
    Create a directory at the specified path if it does not already exist.

    Parameters:
    - path (str): The path of the directory to be created.
    """
    import os
    os.makedirs(path, exist_ok=True)


def get_files_in_folder(path, pattern):
    """
    Get a sorted array of file paths matching the specified pattern
    within a folder.

    Parameters:
    - path (str): The path of the folder containing the files.
    - pattern (str): The pattern to match the filenames.

    Returns:
    - numpy.ndarray: A sorted array of file paths.
    """
    import os
    import numpy as np
    import glob
    return np.sort(glob.glob(os.path.join(path, pattern)))


def walk_path_target(path, target):
    """
    Walk through the directory path and find file(s) matching the
    specified target.

    Parameters:
    - path (str): The path of the directory to search.
    - target (str): The target file(s) to search for.

    Returns:
    - numpy.ndarray: A sorted array of file paths matching the target.
    """
    import os
    import numpy as np
    import glob
    # function to walk through the directory path and find target file(s)
    return np.sort([y for x in os.walk(path) for y
                    in glob.glob(os.path.join(x[0], target))])
    


def file_in_folder(folder_path, file_name):
    """
    Check if a file exists within a folder.

    Parameters:
        folder_path (str or Path): Path to the folder where to check for the file.
        file_name (str): Name of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    from pathlib import Path
    folder = Path(folder_path)
    file_path = folder / file_name
    return file_path.exists()
