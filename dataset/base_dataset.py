# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:31:39 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch


# %% TestDataset

class Dataset(torch.utils.data.Dataset):
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

    def __init__(self, data, label, transforms=None):

        self.data = data
        self.label = label
        self.transforms = transforms

    def __str__(self):
        """
        Define the describtion of the class.

        Returns
        -------
        str
            The class describtion.

        """
        return "Simple Dataset"

    def __len__(self):
        """
        Define the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.

        """
        return len(self.data)

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
        if torch.rand(1) > 0.5 and self.transforms:
            self.data[idx] = self.transforms(self.data[idx])
        return self.data[idx], self.label[idx]

# %% Dataloader


class DataLoader():
    """

    Parameters.

    ----------
    data : tensor, required
        Dataset as Itarable Object which gives tuple of inputs and outputs.
    batch_size : int, optional
        Defines the size of one batch. The default is 1.
    shuffle : bool, optional
        If shuffle is set True all Datapoint are mixed. The default is None.

    Raises
    ------
    Stop Iteration
        If all Datapoints are processed

    Returns
    -------
    Iterable Object.

    """

    def __init__(self, data, batch_size=1, shuffle=False):

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.len = len(data)
        n_batches, remainder = divmod(self.len, self.batch_size)
        self.n_batches = n_batches

    def __str__(self):
        """
        Define the describtion of the class.

        Returns
        -------
        str
            The class describtion.

        """
        return "Simple Dataloader"

    def __iter__(self):
        """
        Iterate over a Dataset.

        Returns
        -------
        None

        """
        if self.shuffle:
            indices = torch.randperm(self.len)
            self.input_tensors, self.targets_tensors = self.data[indices]
        else:
            indices = torch.arange(self.len)
            self.input_tensors, self.targets_tensors = self.data[indices]
        self.i = 0
        return self

    def __next__(self):
        """
        Get next batch.

        Raises
        ------
        StopIteration
            If iteration is at the end of the dataset.

        Returns
        -------
        batch_input : double, required
            Input batch of a classifier.
        batch_target : double, required
            Ground truth (correct) target values.

        """
        if self.i >= self.len - self.n_batches:
            raise StopIteration
        batch_input = self.input_tensors[self.i:self.i+self.batch_size]
        batch_target = self.targets_tensors[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        return batch_input, batch_target

    def __len__(self):
        """
        Give the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.

        """
        return self.len
