# -*- coding: utf-8 -*-
"""
Utility module for locating dataset directories dynamically.
"""

import os

def get_dataset_path(
    folder_name: str = "dataset", 
    path: str = "", 
    max_depth: int = 10
) -> str:
    """
    Searches upwards through the directory tree to find a specific folder.

    Parameters
    ----------
    folder_name : str, optional
        The name of the folder to search for. Defaults to "dataset".
    path : str, optional
        The starting path for the search. If left empty, the search begins at the 
        parent directory of the current working directory. Defaults to "".
    max_depth : int, optional
        The maximum number of parent directories to traverse upwards. Defaults to 10.

    Returns
    -------
    str
        The absolute path to the targeted folder. If the folder is not found within 
        the maximum depth, it returns the joined path of the highest reached 
        directory and the folder name.
    """
    depth = 0
    
    # Initialize the starting directory
    if not path:
        # Note: This starts at the PARENT of the current working directory
        current_dir = os.path.dirname(os.getcwd())
    else:
        current_dir = path

    while depth < max_depth:
        target_path = os.path.join(current_dir, folder_name)
        
        # Check if the directory exists directly (faster and safer than os.listdir)
        if os.path.isdir(target_path):
            return target_path
            
        # Move one directory level up
        parent_dir = os.path.dirname(current_dir)
        
        # Safety break: Stop if we have reached the root directory (e.g., 'C:\\' or '/')
        # to prevent infinite loops if max_depth is set extremely high.
        if current_dir == parent_dir:
            break
            
        current_dir = parent_dir
        depth += 1
        
    return os.path.join(current_dir, folder_name)