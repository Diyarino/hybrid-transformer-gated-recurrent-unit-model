# -*- coding: utf-8 -*-
"""
Experiment Folder Generation Module

This module provides utilities to automatically generate, structure, and 
manage directories for training machine learning models, ensuring organized 
storage for setups, data, images, and model weights.
"""

import os
import time
from typing import Tuple, Optional, Union, Any


class GenerateFolder:
    """
    Generates and manages a structured directory hierarchy for training trials.
    """

    def __init__(self, generate_all: bool = False) -> None:
        """
        Initializes the folder generation class.

        Parameters
        ----------
        generate_all : bool, optional
            If True, instantly generates the train, setup, and data folders 
            upon instantiation. Defaults to False.
        """
        self.trainfolder: Optional[str] = None
        self.setupfolder: Optional[str] = None
        self.imgfolder: Optional[str] = None
        self.datafolder: Optional[str] = None
        self.netfolder: Optional[str] = None
        self.tablefolder: Optional[str] = None
        
        self.setup_idx: int = 1
        
        if generate_all:
            self.generate_train_folder(generate=True)
            self.generate_setup_folder(generate=True)
            self.generate_data_folder(generate=True)
            
    def _validate_location(self, location: Any) -> Optional[str]:
        """
        Internal helper to ensure the location is a string. 
        Extracts the trainfolder path if a GenerateFolder object was passed by mistake.
        """
        if location is None:
            return None
        
        # Bugfix for the TypeError you encountered in main.py
        if isinstance(location, GenerateFolder):
            if location.trainfolder is None:
                raise ValueError("Passed a GenerateFolder object as location, but its trainfolder is None.")
            return location.trainfolder
            
        if not isinstance(location, (str, os.PathLike)):
            raise TypeError(f"Location must be a string or PathLike object, got {type(location).__name__}")
            
        return str(location)

    def generate_train_folder(
        self, 
        generate: bool = False, 
        location: Optional[Union[str, 'GenerateFolder']] = None, 
        name: str = ''
    ) -> 'GenerateFolder':
        """
        Defines and optionally creates the main training directory based on current time.

        Parameters
        ----------
        generate : bool, optional
            Directly creates the folder on the disk. Defaults to False.
        location : str or GenerateFolder, optional
            Alternative base path. Defaults to None (uses CWD).
        name : str, optional
            Suffix string to append to the folder name. Defaults to ''.

        Returns
        -------
        GenerateFolder
            Returns self for method chaining.
        """
        time_str = time.strftime("%y_%m_%d__%H_%M_%S") + name
        
        valid_location = self._validate_location(location)
        if valid_location is None:
            valid_location = os.getcwd()
            
        self.trainfolder = os.path.join(valid_location, "training", time_str)
        
        if generate:
            os.makedirs(self.trainfolder, exist_ok=True)
            
        return self
    
    def generate_setup_folder(
        self, 
        generate: bool = False, 
        location: Optional[Union[str, 'GenerateFolder']] = None
    ) -> 'GenerateFolder':
        """
        Defines and optionally creates a specific setup sub-directory.

        Parameters
        ----------
        generate : bool, optional
            Directly creates the folder on the disk. Defaults to False.
        location : str or GenerateFolder, optional
            Alternative base path. Defaults to None (uses trainfolder).

        Returns
        -------
        GenerateFolder
            Returns self for method chaining.
        """        
        valid_location = self._validate_location(location)
        if valid_location is None:
            if self.trainfolder is None:
                raise ValueError("Cannot create setup folder: trainfolder is not initialized.")
            valid_location = self.trainfolder
            
        self.setupfolder = os.path.join(valid_location, str(self.setup_idx).zfill(3) + "_setup")
        
        if generate:
            os.makedirs(self.setupfolder, exist_ok=True)
            
        return self
    
    def generate_data_folder(
        self, 
        generate: bool = False, 
        location: Optional[Union[str, 'GenerateFolder']] = None
    ) -> Tuple[str, str, str, str]:
        """
        Defines and optionally creates the data, image, model, and table sub-directories.

        Parameters
        ----------
        generate : bool, optional
            Directly creates the folders on the disk. Defaults to False.
        location : str or GenerateFolder, optional
            Alternative base path. Defaults to None (uses setupfolder).

        Returns
        -------
        Tuple[str, str, str, str]
            Paths for (datafolder, imgfolder, netfolder, tablefolder).
        """
        valid_location = self._validate_location(location)
        if valid_location is None:
            if self.setupfolder is None:
                raise ValueError("Cannot create data folders: setupfolder is not initialized.")
            valid_location = self.setupfolder
            
        self.datafolder = os.path.join(valid_location, "data")
        self.imgfolder = os.path.join(valid_location, "img")
        self.netfolder = os.path.join(valid_location, "model")
        self.tablefolder = os.path.join(valid_location, "table")
        
        if generate:
            os.makedirs(self.datafolder, exist_ok=True)
            os.makedirs(self.imgfolder, exist_ok=True)
            os.makedirs(self.netfolder, exist_ok=True)
            os.makedirs(self.tablefolder, exist_ok=True)
            
        return self.datafolder, self.imgfolder, self.netfolder, self.tablefolder

    def get_folder(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Outputs all current folder paths tracked by the instance.

        Returns
        -------
        Tuple[str, str, str, str, str, str]
            Paths for (trainfolder, setupfolder, datafolder, imgfolder, netfolder, tablefolder).
        """
        return (self.trainfolder, self.setupfolder, self.datafolder, 
                self.imgfolder, self.netfolder, self.tablefolder)
    
    def __call__(self, setup_step: bool = False) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Updates the internal setup index and returns all active folders.

        Parameters
        ----------
        setup_step : bool, optional
            If True, increments the setup index and creates a new setup folder. Defaults to False.

        Returns
        -------
        Tuple[str, str, str, str, str, str]
            All tracked folder paths.
        """
        if setup_step:
            self.setup_idx += 1
            # Note: We don't automatically trigger data_folder generation here.
            # You might want to call self.generate_data_folder(generate=True) 
            # after this if you want new data folders per setup step!
            self.generate_setup_folder(generate=True)
            
        return (self.trainfolder, self.setupfolder, self.datafolder, 
                self.imgfolder, self.netfolder, self.tablefolder)

    # --- Backwards Compatibility Aliases ---
    # These map your old uppercase method calls (e.g., from main.py) to the new PEP 8 methods.
    GenerateTrainFolder = generate_train_folder
    GenerateSetupFolder = generate_setup_folder
    GenerateDataFolder = generate_data_folder
    GetFolder = get_folder


def generate_trainfolder(
    generate: bool = False, 
    location: Optional[str] = None
) -> Tuple[str, str, str, str, str]:
    """
    Standalone function to quickly generate a standard training directory structure.

    Parameters
    ----------
    generate : bool, optional
        Directly creates the folders on the disk. Defaults to False.
    location : str, optional
        Alternative base path. Defaults to None (uses CWD).

    Returns
    -------
    Tuple[str, str, str, str, str]
        Paths for (train_folder, img_folder, config_folder, model_folder, data_folder).
    """
    time_str = time.strftime("%y_%m_%d__%H_%M_%S")
    
    if location is None:
        location = os.getcwd()
        
    train_folder = os.path.join(location, "training", time_str)
    img_folder = os.path.join(train_folder, "img")
    config_folder = os.path.join(train_folder, "config")
    model_folder = os.path.join(train_folder, "model")
    data_folder = os.path.join(train_folder, "data")
    
    if generate:
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(config_folder, exist_ok=True)
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(data_folder, exist_ok=True)
        
    return train_folder, img_folder, config_folder, model_folder, data_folder