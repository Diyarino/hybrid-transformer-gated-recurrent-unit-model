# -*- coding: utf-8 -*-
"""
Module for data scaling, normalization, and windowed dataset generation.
"""

import os
from typing import Tuple, List, Optional

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# %% Scaling methods

class NormZScore:
    """Z-score normalization (standardization) for a given tensor."""
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data
        self.data_normalized = (self.data - self.data.mean(dim=0)) / self.data.std(dim=0)

    def __call__(self) -> torch.Tensor:
        return self.data_normalized


class NormLinearScaling:
    """Min-Max linear scaling to a specific range."""
    def __init__(self, data: torch.Tensor, lower_value: float, upper_value: float) -> None:
        self.data = data
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.min = torch.min(self.data)
        self.max = torch.max(self.data)
        
        # Min-Max Scaling formula
        self.data_normalized = (
            (self.upper_value - self.lower_value) * (self.data - self.min) / (self.max - self.min)
        ) + self.lower_value

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data_normalized, self.min, self.max


class DeNormLinearScaling:
    """Reverses the linear scaling to return to the original data range."""
    def __init__(
        self, 
        data: torch.Tensor, 
        lower_value: float, 
        upper_value: float, 
        minimum: torch.Tensor, 
        maximum: torch.Tensor
    ) -> None:
        self.data = data
        self.lower = lower_value
        self.upper = upper_value
        self.min = minimum
        self.max = maximum
        
        # Inverse Min-Max Scaling formula
        self.data_denormalized = (
            (self.max - self.min) * (self.data - self.lower) / (self.upper - self.lower)
        ) + self.min

    def __call__(self) -> torch.Tensor:
        return self.data_denormalized


class NormLogScaling:
    """Logarithmic scaling for a given tensor."""
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data
        self.data_normalized = torch.log(self.data)

    def __call__(self) -> torch.Tensor:
        return self.data_normalized


# %% Normalization

class Normalization:
    """
    Applies a specified normalization method to all sensors (columns) of the data
    and optionally plots the normalized distributions.
    """
    def __init__(
        self, 
        data: torch.Tensor, 
        save_path: str = "", 
        normalize: str = "Z score", 
        low_value_normalization: float = 0.0, 
        upper_value_normalization: float = 1.0, 
        plot_normalized_dataset: bool = True
    ) -> None:
        self.data_raw = data
        self.normalize = normalize
        self.low_value_normalization = low_value_normalization
        self.upper_value_normalization = upper_value_normalization
        self.plot_normalized_dataset = plot_normalized_dataset
        self.save_path = save_path
        self.time = torch.arange(0, self.data_raw.shape[0], 1)
        
        # Pre-allocate variable for the processed dataset
        self.dataset: torch.Tensor = torch.empty(0)
        
        # Execute normalization logic
        self._process_data()

    def _process_data(self) -> 'Normalization':
        """Applies normalization per channel and collects the results."""
        processed_series: List[torch.Tensor] = []
        
        for i in range(self.data_raw.shape[1]):
            sensor = self.data_raw[:, i]
            series = sensor # Fallback
            
            if self.normalize == "Z score":
                series = NormZScore(sensor)()
                title = f"Normalized Z Score - sensor_{i}"
                
            elif self.normalize == "linear scalling": # Kept string for backwards compatibility
                series, _, _ = NormLinearScaling(
                    sensor, self.low_value_normalization, self.upper_value_normalization
                )()
                title = f"Normalized linear scaling sensor_{i}"
                
            elif self.normalize == "log scalling":
                series = NormLogScaling(sensor)()
                title = f"Normalized log scaling sensor_{i}"
            
            # Plotting if required
            self.plot_normalization(
                self.time, series, title=title, run=i, plotting=self.plot_normalized_dataset
            )
            
            # Reshape and append to list instead of slow torch.cat in a loop
            processed_series.append(series.reshape(-1, 1))
            
        # Efficiently concatenate all processed columns at once
        self.dataset = torch.cat(processed_series, dim=1)
        return self

    def __call__(self) -> torch.Tensor:
        return self.dataset

    def plot_normalization(
        self, 
        time: torch.Tensor, 
        series: torch.Tensor, 
        title: str, 
        run: int, 
        plotting: bool = False, 
        line_format: str = "-", 
        start: int = 0, 
        end: Optional[int] = None
    ) -> plt.Figure:
        """Plots the normalized series and saves the figure to disk."""
        plt.style.use("default")
        fig = plt.figure(figsize=(5, 3))
        plt.plot(time[start:end], series[start:end], line_format)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        
        # Safe and cross-platform path building
        data_path = os.path.join(self.save_path, "02_Plots Dataset Generation", f"run{run}")
        os.makedirs(data_path, exist_ok=True)
        
        save_file = os.path.join(data_path, f"{title}.jpg")
        plt.savefig(save_file, bbox_inches="tight")
        
        if plotting:
            plt.show()
        else:
            plt.close(fig)
            
        return fig


# %% Windowing

class PreprocessingDataset(Dataset):
    """
    Creates a windowed PyTorch Dataset from time-series data.
    """
    def __init__(
        self, 
        train: bool, 
        data: torch.Tensor, 
        save: bool = True, 
        save_path: str = "", 
        splittingdataset: bool = False, 
        reshapedata: bool = False, 
        window_size: int = 25, 
        shift: int = 25, 
        selected_channels: Tuple[int, ...] = (0, 1, 2)
    ) -> None:
        self.save = save
        self.save_path = save_path
        self.window_size = window_size
        self.shift = shift
        self.selected_channels = selected_channels
        
        self.sensors = data.shape[1]
        self.data_raw = data
        self.label = torch.empty(0)  # Standardized empty tensor initialization
        
        # Train / Test split (80/20)
        split_idx = int(self.data_raw.shape[0] * 0.8)
        if train:
            self.data_raw = self.data_raw[:split_idx]
        else:
            self.data_raw = self.data_raw[split_idx:]
            
        self.datasplit: torch.Tensor = torch.empty(0)
        self.selected_dataset: torch.Tensor = torch.empty(0)
        
        # Execute preprocessing pipeline
        self.moving_window()
        self.select_channels()
        
        if reshapedata:
            self.catall()
            
    def splitdata(self, initial_split: int = 1) -> None:
        """
        Splits the dataset into exact mathematical chunks. 
        Note: initial_split must be provided to prevent infinite loops.
        """
        self.split = initial_split
        # Ensure division without remainder
        while self.data_raw.shape[0] % self.split != 0:
            self.split += 1

        size_window = int(self.data_raw.shape[0] / self.split)
        split_complete_ds = int(self.data_raw.shape[0] / size_window)
        
        print(f"The dataset will be split into {self.split} parts of {size_window} rows.")
        
        self.datasplit = torch.zeros(split_complete_ds, self.data_raw.shape[1], size_window) 
        
        for j in range(self.sensors):
            k = 0
            for i in range(0, self.data_raw.shape[0], size_window):
                self.datasplit[k, j] = self.data_raw[i : i + size_window, j]
                k += 1
                
    def moving_window(self) -> None:
        """Applies a sliding window over the time-series data."""
        window_split = int((self.data_raw.shape[0] - self.window_size) / self.shift) + 1
        self.datasplit = torch.zeros([window_split, self.sensors, self.window_size]) 
        
        k = 0
        for i in range(window_split):
            self.datasplit[i] = self.data_raw[k : k + self.window_size].permute(1, 0)
            k += self.shift
            
    def select_channels(self) -> None:
        """Filters the windowed dataset for specific sensor channels."""
        # Using list comprehension for performance instead of iterative torch.cat
        channels = [self.datasplit[:, i : i + 1] for i in self.selected_channels]
        if channels:
            self.selected_dataset = torch.cat(channels, dim=1)
        else:
            self.selected_dataset = torch.empty(0)
   
    def catall(self) -> None:
        """Reshapes the selected dataset into a single-channel configuration."""
        batch_size = self.selected_dataset.shape[0]
        num_channels = self.selected_dataset.shape[1]
        window_len = self.selected_dataset.shape[2]
        
        self.selected_dataset = torch.reshape(
            self.selected_dataset, 
            (batch_size * num_channels, 1, window_len)
        )
        
    def __len__(self) -> int:
        return self.selected_dataset.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.selected_dataset[idx]


# %% Backwards Compatibility Aliases
# Keep your old class names working if you import them in other files!
Norm_z_score = NormZScore
Norm_linear_scalling = NormLinearScaling
DeNorm_linear_scalling = DeNormLinearScaling
Norm_log_scalling = NormLogScaling