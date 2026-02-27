# -*- coding: utf-8 -*-
"""
Module for processing and loading Stadtwerke time-series data.

This module provides tools to split tensor data into rolling windows
and a custom PyTorch Dataset for handling the Stadtwerke energy dataset.
"""

import os
from typing import Tuple, Optional

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


def create_windows(
    data: torch.Tensor, 
    window_length: int, 
    window_step: Optional[int] = None
) -> torch.Tensor:
    """
    Split up the data tensor into multiple (overlapping) windows.

    Parameters
    ----------
    data : torch.Tensor
        Torch Tensor of shape [M, ...] to be split up.
    window_length : int
        Specifying the length of each window. Cannot be greater than len(data).
    window_step : int, optional
        Specifying the distance between windows. Defaults to window_length.

    Returns
    -------
    torch.Tensor
        Torch tensor containing the sequences of shape [N, window_length, ...].

    Raises
    ------
    AttributeError
        If the data is shorter than the requested window length.
    """
    window_step = window_step if window_step is not None else window_length
    num_samples = len(data)

    if num_samples < window_length:
        raise AttributeError(
            f"Incorrect window_length: can't create a window of size {window_length} "
            f"for {num_samples} data points."
        )
    if window_length <= 1:
        return data

    num_windows = (num_samples - window_length) // window_step
    
    # Pre-allocate output tensor
    sequences = torch.zeros(num_windows, window_length, *data.shape[1:], dtype=data.dtype)

    idx = window_length
    for n in range(num_windows):
        sequences[n] = data[idx - window_length : idx]
        idx += window_step

    return sequences


class StadtwerkeDataset(Dataset):
    """
    Custom iterable Dataset for Stadtwerke time-series data.

    Parameters
    ----------
    path : str
        Path to the directory containing 'Data_test.csv'.
    window_length : int, optional
        The number of elements in each input sequence. Defaults to 5.
    window_step : int, optional
        The step size between windows. Defaults to window_length.
    mode : str, optional
        Defines the split mode ('train' or 'test'). Defaults to 'train'.
    temp : bool, optional
        If True, includes temperature data in labels. Defaults to False.
    """

    def __init__(
        self, 
        path: str, 
        window_length: int = 5, 
        window_step: Optional[int] = None, 
        mode: str = 'train', 
        temp: bool = False
    ) -> None:
        
        self.window_step = window_step if window_step is not None else window_length
        self.window_length = window_length
        self.num_days_forecast = 2
        self.mode = mode
        
        # 1. Load and clean data
        csv_path = os.path.join(path, 'Data_test.csv')
        self.dataset_raw = self.load_csv_to_tensor(csv_path)
        self.dataset_clean = self.remove_nan(self.dataset_raw).float()
        
        # 2. Normalize
        self.dataset_normalized, self.mean, self.std = self.extract_mean_std(self.dataset_clean)
        
        # 3. Create inputs and labels
        self.windowed_data = self.create_windows(
            self.dataset_normalized.flatten(end_dim=1), 
            input_size=96 * self.num_days_forecast
        )
        self.inputs, self.labels = self._create_input_label_pairs_advanced(
            self.dataset_normalized, 
            window_length
        )
        
        # 4. Filter labels based on 'temp' flag
        if temp:
            self.labels = self.labels[:, :, [4, 7]]
        else:
            self.labels = self.labels[:, :, [7]]
            
        # 5. Train/Test Split (Hardcoded indices as per original logic)
        split_idx = 100000
        if mode == 'train':
            self.inputs = self.inputs[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.inputs = self.inputs[split_idx:]
            self.labels = self.labels[split_idx:]

    def create_input_label_pairs(self, tensor: torch.Tensor, input_size: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates simple input-label pairs from a tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The input tensor.
        input_size : int, optional
            The number of elements in each input sequence. Defaults to 5.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing input tensors and label tensors.
        """
        num_pairs = len(tensor) - input_size
        inputs, labels = [], []
        
        for i in range(num_pairs):
            inputs.append(tensor[i : i + input_size])
            labels.append(tensor[i + 1 : i + input_size + 1])
            
        return torch.stack(inputs), torch.stack(labels)
    
    def _create_input_label_pairs_advanced(
        self, 
        data: torch.Tensor, 
        input_size: int = 5, 
        window_size: int = 96, 
        num_days: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates advanced input-label pairs considering window sizes and forecast days.

        Parameters
        ----------
        data : torch.Tensor
            The input tensor.
        input_size : int, optional
            Number of elements in each input sequence. Defaults to 5.
        window_size : int, optional
            The size of a single temporal window (e.g., 96 quarters per day). Defaults to 96.
        num_days : int, optional
            Number of days to forecast. Defaults to 2.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Flattened input and label tensors.
        """
        data = data.flatten(end_dim=1)
        num_samples = data.size(0)
        input_length = window_size * (input_size + 1)
        num_windows = num_samples - (num_days * input_length)
        
        inputs, labels = [], []
        
        for i in range(num_windows):
            buffer = []
            # 'j' is used after the loop in the original code. 
            # Defining it explicitly here ensures it's available for the label slicing.
            j = 0 
            for j in range(input_size):
                start_idx = j * window_size + i
                end_idx = start_idx + window_size
                buffer.append(data[start_idx:end_idx])
                
            inputs.append(torch.stack(buffer))
            
            label_start = (j + 1) * window_size + i
            label_end = label_start + (num_days * window_size)
            labels.append(data[label_start:label_end])
            
        inputs_tensor = torch.stack(inputs).flatten(start_dim=1, end_dim=2)
        labels_tensor = torch.stack(labels)
        
        return inputs_tensor, labels_tensor
    
    def remove_nan(self, df_encoded: pd.DataFrame) -> torch.Tensor:
        """
        Remove NaN values from the encoded dataframe and convert to a reshaped tensor.

        Parameters
        ----------
        df_encoded : pd.DataFrame
            The input data which should be cleaned.

        Returns
        -------
        torch.Tensor
            The cleaned and reshaped PyTorch tensor.
        """
        data_tensor = torch.tensor(df_encoded.values)
        data_tensor = data_tensor.reshape(1720, 96, 16)
        mask = ~torch.isnan(data_tensor).flatten(start_dim=1).any(dim=1)
        filtered_tensor = data_tensor[mask]
        
        return filtered_tensor
        
    def extract_mean_std(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize the dataset based on z-score normalization.

        Parameters
        ----------
        data : torch.Tensor
            The input data to be normalized.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the normalized tensor, the mean, and the standard deviation.
        """
        shape_buffer = data.shape
        data_flat = data.flatten(start_dim=0, end_dim=1)
        
        mean = data_flat.mean(dim=0)
        std = data_flat.std(dim=0)
        
        data_norm = (data_flat - mean) / std
        data_norm = data_norm.reshape(shape_buffer)
        
        return data_norm, mean, std
    
    def create_windows(self, data: torch.Tensor, input_size: int = 5) -> torch.Tensor:
        """
        Creates windowed sequences from a tensor.

        Parameters
        ----------
        data : torch.Tensor
            The input tensor.
        input_size : int, optional
            The number of elements in each sequence. Defaults to 5.

        Returns
        -------
        torch.Tensor
            A stacked tensor of the created windows.
        """
        num_pairs = len(data) - input_size
        inputs = []
        for i in range(num_pairs):
            inputs.append(data[i : i + input_size])
            
        return torch.stack(inputs)
        
    def load_csv_to_tensor(self, file_path: str) -> pd.DataFrame:
        """
        Loads and pre-processes data from a CSV file into a pandas DataFrame.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        
        Returns
        -------
        pd.DataFrame
            An encoded and cleaned pandas DataFrame.
        """
        df = pd.read_csv(file_path, on_bad_lines='skip', delimiter=';', low_memory=False)
        
        # Remove empty or unnecessary columns
        df_without_column = df.iloc[1:, :].copy()
        df_without_column = df_without_column.drop(columns=['Column5', 'Column1'], errors='ignore')
        
        # Encode categorical variables and handle corrupt numeric formats
        df_encoded = pd.get_dummies(df_without_column, columns=['Column2'], prefix='', prefix_sep='')
        df_encoded.replace(['? 0,00', '? 0,0'], np.nan, inplace=True)
        
        return df_encoded.astype(float)

    def __str__(self) -> str:
        """Return a brief description of the dataset."""
        return "Stadtwerke Time-Series Dataset"
    
    def __repr__(self) -> str:
        """Return the detailed representation of the class instance."""
        return f"{self.__class__.__name__}(window_length={self.window_length}, window_step={self.window_step})"

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the data point and its corresponding label at the given index.

        Parameters
        ----------
        idx : int
            Index of the item.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of the input sequence and the corresponding label.
        """
        return self.inputs[idx], self.labels[idx]


# %% Test Execution

if __name__ == '__main__':
    # Best practice: Avoid executing logic if module is only imported
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    
    # Optional: ensure directory exists before running to avoid obscure errors
    if os.path.exists(dataset_path):
        dataset = StadtwerkeDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        batch_inputs, batch_labels = next(iter(dataloader))
        print(f"Inputs Shape: {batch_inputs.shape}")
        print(f"Labels Shape: {batch_labels.shape}")
    else:
        print(f"Dataset path not found: {dataset_path}. Please adjust the path.")