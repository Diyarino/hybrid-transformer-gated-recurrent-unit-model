# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:06:10 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - ...
"""

# %% Imports

import os
import torch
import pandas as pd
import numpy as np

# %% create windows from data


def create_windows(data, window_length, window_step=None):
    """
    Split up the data-tensor into multiple (overlaping) windows.

    Parameters
    ----------
    data : torch.tensor
        Torch Tensor of shape [M x ...] to be split up into specified window size.
    window_length : int
        Specifying length of each window. Cannot be greater than len(data).
    window_step : int, optional
        Specifying distance between windows. The default is None, meaning window_step = window_length.

    Returns
    -------
    Sequences : torch.tensor
        Torch tensor of shape [N x L x ...].

    """
    window_step = window_step if window_step else window_length
    num_samples = len(data)

    # calc amount of windows
    if num_samples < window_length:
        raise AttributeError(
            "Incorrect window_length, can't create a window with size %i for %i data-points." % (window_length, num_samples))
    elif window_length <= 1:
        return data
    else:
        num_windows = (num_samples-window_length)//window_step

    sequences = torch.zeros(num_windows, window_length, *data.shape[1:])

    idx = window_length
    for n in range(num_windows):
        sequences[n, :, :] = data[idx-window_length: idx]
        idx += window_step

    return sequences

# %% dataste


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Generate a iterable Dataset.

    Parameters
    ----------
    Data : iterable
        Input data best used as tensor.
    Label : iterable
        Output data best used as tensor.

    Returns
    -------
    None.

    """

    def __init__(self, path, window_length=5, window_step=None, mode = 'train', temp = False):

        window_step = window_step if window_step else window_length
        
        self.num_days_forcast = 2
        
        self.window_step = window_step
        self.window_length = window_length
        
        self.dataset_raw = self.load_csv_to_tensor(os.path.join(path, 'Data_test.csv'))
        self.dataset_clean = self.remove_nan(self.dataset_raw).float()
        self.dataset_normalized, self.mean, self.std = self.extract_mean_std(self.dataset_clean)
        
        self.windowed_data = self.create_windows(self.dataset_normalized.flatten(end_dim = 1), input_size=96*self.num_days_forcast)
        self.inputs, self.labels = self.create_input_label_pairs____(self.dataset_normalized, window_length)
        
        # self.inputs, self.labels = self.create_input_label_pairs(self.dataset_normalized, window_length)
        # self.inputs = self.inputs.flatten(start_dim = 1, end_dim = 2)
        # self.labels = self.labels.flatten(start_dim = 1, end_dim = 2)
        
        # self.inputs = torch.cat([self.inputs, self.labels[:,:,4].unsqueeze(-1)], dim = -1)
        self.inputs = self.inputs
        if temp:
            self.labels = self.labels[:, :, [4, 7]]
        else:
            self.labels = self.labels[:, :, [7]]
        
        indices = torch.randperm(self.inputs.shape[0])
        num_train_samples = self.inputs.shape[0]//4*3
        train_inidces, test_inidces = indices[:num_train_samples], indices[num_train_samples:]
        
        if mode == 'train':
            self.inputs, self.labels = self.inputs[train_inidces], self.labels[train_inidces]
        else:
            self.inputs, self.labels = self.inputs[test_inidces], self.labels[test_inidces]
        
        
        self.data = mode
        
        
    def create_input_label_pairs(self, tensor, input_size=5):
        """Creates input-label pairs from a tensor.
    
        Args:
            tensor: The input tensor.
            input_size: The number of elements in each input sequence.
    
        Returns:
            A tuple of two lists: input tensors and label tensors.
        """
    
        num_pairs = len(tensor) - input_size
    
        inputs = []
        labels = []
        for i in range(num_pairs):
            inputs.append(tensor[i:i+input_size])
            labels.append(tensor[i+1:i+input_size+1])
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        return inputs, labels
    
    def create_input_label_pairs____(self, data, input_size=5, window_size = 96, num_days = 2):
        '''
        Creates input-label pairs from a tensor.

        Parameters
        ----------
        data : torch.tensor
            The input tensor.
        size : int
            The number of elements in each input sequence.

        Returns
        -------
        A tuple of two lists: input tensors and label tensors.

        '''
        data = data.flatten(end_dim = 1)
        num_samples = data.size(0)  # Anzahl der Zeitfenster
        input_length = window_size*(input_size+1) # Anzahl der Zeitfenster inkl. Label
        num_windows = num_samples - num_days*input_length
        
        inputs, labels = [], []
        for i in range(num_windows):
            buffer = []
            for j in range(input_size):
                buffer.append(data[j*window_size +i : j*window_size+window_size +i])
            inputs.append(torch.stack(buffer))
            labels.append(data[(j+1)*window_size+i : (j+1)*window_size+i+(num_days*window_size)])
        inputs, labels = torch.stack(inputs), torch.stack(labels)
        inputs = inputs.flatten(start_dim =1, end_dim = 2)
        return inputs, labels
    
    def remove_nan(self, df_encoded):
        '''
        Remove Nan values from a tensor

        Parameters
        ----------
        df_encoded : pandas.dataframe
            The input data which should be cleaned.

        Returns
        -------
        the new cleaned tensor.

        '''
        data_tensor = torch.tensor(df_encoded.values)
        data_tensor = data_tensor.reshape(1720,96,16)
        mask = ~torch.isnan(data_tensor).flatten(start_dim = 1).any(dim=1)
        filtered_tensor = data_tensor[mask]
        return filtered_tensor
        
    def extract_mean_std(self, data):
        '''
        Normalize the dataset based on z-norm.

        Parameters
        ----------
        data : torch.tensor
            The input data which should be normalized.

        Returns
        -------
        None.

        '''
        shape_buffer = data.shape
        data = data.flatten(start_dim = 0, end_dim = 1)
        mean, std = data.mean(dim = 0), data.std(dim = 0) 
        data_norm = (data-mean)/std
        data_norm = data_norm.reshape(shape_buffer)
        return data_norm, mean, std
    
    def create_windows(self, data, input_size=5):
        '''
        Creates input-label pairs from a tensor.

        Parameters
        ----------
        data : torch.tensor
            The input tensor.
        size : int
            The number of elements in each input sequence.

        Returns
        -------
        A tuple of two lists: input tensors and label tensors.

        '''
        num_pairs = len(data) - input_size
        inputs = []
        for i in range(num_pairs):
            inputs.append(data[i:i+input_size])
        inputs = torch.stack(inputs)
        return inputs
        
    def load_csv_to_tensor(self, file_path):
        """
        Loads data from a CSV file into a PyTorch tensor.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        
        Returns
        -------
        str
            A PyTorch tensor containing the data from the CSV file.
        
        """
        df = pd.read_csv(file_path, on_bad_lines='skip', delimiter = ';', low_memory=False)
        
        # remove all the bullshit and encode the data
        df_without_column = df.iloc[1: , :]
        df_without_column = df_without_column.drop('Column5', axis=1)
        df_without_column = df_without_column.drop('Column1', axis=1)
        # df_without_column = df_without_column.drop('Column9', axis=1)
        # df_without_column = df_without_column.drop('Column10', axis=1)
        # df_without_column = df_without_column.drop('Column12', axis=1)
        df_encoded = pd.get_dummies(df_without_column, columns=['Column2'], prefix='', prefix_sep='')
        df_encoded.replace('? 0,00', np.nan, inplace=True)
        df_encoded.replace('? 0,0', np.nan, inplace=True)
        df_encoded = df_encoded.astype(float) 
        
        return df_encoded
    

    def __str__(self):
        """
        Define the describtion of the class.

        Returns
        -------
        str
            The class describtion.

        """
        return "Simple Dataset"
    
    def __repr__(self):
        """
        Represens the name of the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(window_length={}, window_step={})'.format(
            self.window_length, self.window_step)

    def __len__(self):
        """
        Define the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.

        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Get the following item.

        Parameters
        ----------
        idx : int
            Index of item.

        Returns
        -------
        temp : tuple
            Tuple of Datapoint and corresponding label.

        """
        return self.inputs[idx], self.labels[idx]

# %% test


if __name__ == '__main__':
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    dataset = TimeSeriesDataset(dataset_path)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    camera, joints = next(iter(dataloader))
