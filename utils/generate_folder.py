# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:15:06 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - improve the class
"""

# %% imports

import os
import time

#%% Generate Train Folder class

class GenerateFolder():
    """
    Generate folder and save every path of train trials.

    Returns
    -------
    None.

    """    
    
    def __init__(self, GenerateAll = False):

        self.trainfolder = None
        self.setupfolder = None
        self.imgfolder = None
        self.datafolder = None
        self.netfolder = None
        self.tablefolder = None
        
        self.setup_idx = 1
        
        if GenerateAll:
            self.GenerateTrainFolder(generate = True)
            self.GenerateSetupFolder(generate = True)
            self.GenerateDataFolder(generate = True)
        
    def GenerateTrainFolder(self, generate = False, location = False):
        r"""
        Return the name of the training folder.
    
        Parameters
        ----------
        generate : bool, optional
            Enable to directly create the folder instead of only returning the path.\n
            Default is False.
        location : str, optional
            Alternative location of the folder. 01_Training-folder will be\n
            automatically created at this location. The default is false (using cwd).
    
        Returns
        -------
        str
            Train folder.
    
        """
        time_str=time.strftime("%y_%m_%d__%H_%M_%S")
        
        if not location:
            location = os.getcwd()
        self.trainfolder = os.path.join(location, "training", time_str)
        if generate:
            if not os.path.exists(self.trainfolder):
                os.makedirs(self.trainfolder)
        return self.trainfolder
    
    def GenerateSetupFolder(self, generate = False, location = False):
        r"""
        Return the name of the training folder.
    
        Parameters
        ----------
        generate : bool, optional
            Enable to directly create the folder instead of only returning the path.\n
            Default is False.
        location : str, optional
            Alternative location of the folder. 01_Training-folder will be\n
            automatically created at this location. The default is false (using cwd).
    
        Returns
        -------
        str
            Train folder.
    
        """        
        if not location:
            location = self.trainfolder
        self.setupfolder = os.path.join(location, str(self.setup_idx).zfill(3) + "_Setup")
        if generate:
            if not os.path.exists(self.setupfolder):
                os.makedirs(self.setupfolder)
        return self
    
    def GenerateDataFolder(self, generate = False, location = False):
        r"""
        Return the name of the training folder.
    
        Parameters
        ----------
        generate : bool, optional
            Enable to directly create the folder instead of only returning the path.\n
            Default is False.
        location : str, optional
            Alternative location of the folder. 01_Training-folder will be\n
            automatically created at this location. The default is false (using cwd).
    
        Returns
        -------
        str
            Train folder.
    
        """
        if not location:
            location = self.setupfolder
        self.datafolder = os.path.join(location, "data")
        self.imgfolder = os.path.join(location, "img")
        self.netfolder = os.path.join(location, "model")
        self.tablefolder = os.path.join(location, "table")
        if generate:
            if not os.path.exists(self.datafolder):
                os.makedirs(self.datafolder)
            if not os.path.exists(self.imgfolder):
                os.makedirs(self.imgfolder)
            if not os.path.exists(self.netfolder):
                os.makedirs(self.netfolder)
            if not os.path.exists(self.tablefolder):
                os.makedirs(self.tablefolder)
        return self.datafolder, self.imgfolder, self.netfolder, self.tablefolder

    def GetFolder(self):
        """
        Output all trainfolders.

        Returns
        -------
        trainfolder, setupfolder, datafolder, imgfolder, netfolder
            All trainfolders.

        """
        return self.trainfolder, self.setupfolder, self.datafolder, self.imgfolder, self.netfolder, self.tablefolder
    
    def __call__(self, setup_step = False):
        """
        Create and update the current folder.

        Parameters
        ----------
        setup_step : bool, optional
            If new setup folder should be created. The default is False.

        Returns
        -------
        trainfolder, setupfolder, datafolder, imgfolder, netfolder
            All trainfolders.

        """
        if setup_step:
            self.setup_idx += 1
            self.GenerateSetupFolder(generate=True)
        return self.trainfolder, self.setupfolder, self.datafolder, self.imgfolder, self.netfolder, self.tablefolder

# %% generate folder function

def generate_trainfolder(generate = False, location = False):
    """
    Returns the name of the training folder.

    Parameters
    ----------
    generate : bool, optional
        Enable to directly create the folder instead of only returning the path.\n
        Default is False.
    location : str, optional
        Alternative location of the folder. 01_Training-folder will be\n
        automatically created at this location. The default is false (using cwd).

    Returns
    -------
    str
        Train folder.

    """
    time_str=time.strftime("%y_%m_%d__%H_%M_%S")
    
    if not location:
        location = os.getcwd()
    train_folder = os.path.join(location, "training", time_str)
    img_folder = os.path.join(train_folder, "img")
    config_folder = os.path.join(train_folder, "config")
    model_folder = os.path.join(train_folder, "model")
    data_folder = os.path.join(train_folder, "data")
    if generate:
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        if not os.path.exists(config_folder):
            os.makedirs(config_folder)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
    return train_folder, img_folder, config_folder, model_folder, data_folder