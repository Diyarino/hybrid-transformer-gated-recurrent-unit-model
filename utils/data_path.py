# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:12:41 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""


# %% imports

import os

# %% Get Dataset Path


def get_dataset_path(folder_name="dataset", path="", max_depth=10):
    """
    Parameters.

    ----------
    folder_name : str, optional
        Folder name to look for, first occurence is returned. The default is "dataset".
    max_depth : int, optional
        Maximum number of parental directories to search. The default is 10.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    depth = 0
    if path == "":
        current_dir = os.path.dirname(os.getcwd())
    else:
        current_dir = path

    while depth < max_depth:
        if folder_name in os.listdir(current_dir):
            break
        current_dir = os.path.dirname(current_dir)
        depth += 1
    return os.path.join(current_dir, folder_name)
