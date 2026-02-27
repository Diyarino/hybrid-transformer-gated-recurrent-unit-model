# -*- coding: utf-8 -*-
"""
Visualization module for thermo performance and training losses.
"""

from typing import Any, Union, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt


# %% Application Plots

def plot_thermo(
    prediction: Union[torch.Tensor, np.ndarray, list], 
    label: Union[torch.Tensor, np.ndarray, list]
) -> plt.Figure:
    """
    Generate a plot comparing the predicted and real thermo performance. 

    Parameters
    ----------
    prediction : Union[torch.Tensor, np.ndarray, list]
        The estimated thermo performance data.
    label : Union[torch.Tensor, np.ndarray, list]
        The ground truth (real) thermo performance data.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(prediction, color='red', label='Estimated', linestyle='dashed')
    ax.plot(label, color='royalblue', label='Real')
    
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Thermo Performance')
    ax.set_ylim(0, 15)
    
    fig.tight_layout()
    plt.show()
    
    return fig


def plot_thermo_temp(
    prediction: Union[torch.Tensor, np.ndarray], 
    label: Union[torch.Tensor, np.ndarray], 
    temp_pred: Union[torch.Tensor, np.ndarray], 
    temp_label: Union[torch.Tensor, np.ndarray], 
    mean_temp: Optional[float] = None  # Parameter kept for backwards compatibility
) -> plt.Figure:
    """
    Generate a two-panel plot displaying thermo performance and temperature comparisons.

    Parameters
    ----------
    prediction : Union[torch.Tensor, np.ndarray]
        The estimated thermo performance data.
    label : Union[torch.Tensor, np.ndarray]
        The ground truth thermo performance data.
    temp_pred : Union[torch.Tensor, np.ndarray]
        The estimated temperature data.
    temp_label : Union[torch.Tensor, np.ndarray]
        The ground truth temperature data.
    mean_temp : float, optional
        Legacy parameter (currently unused in plotting logic). Defaults to None.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure containing both subplots.
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    
    # Subplot 0: Thermo Performance
    axs[0].plot(prediction, color='red', label='Estimated')
    axs[0].plot(label, color='royalblue', label='Real')
    
    # Use axhline instead of creating an array of ones
    axs[0].axhline(y=float(np.mean(np.array(prediction))), color='red', linestyle='dashed')
    axs[0].axhline(y=float(np.mean(np.array(label))), color='royalblue', linestyle='dashed')
    
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Thermo Performance')
    axs[0].set_ylim(0, 15)
    
    # Subplot 1: Temperature
    axs[1].plot(temp_pred, color='red', label='Estimated')
    axs[1].plot(temp_label, color='royalblue', label='Real')
    
    # Use axhline for temperature means as well
    axs[1].axhline(y=float(np.mean(np.array(temp_pred))), color='red', linestyle='dashed')
    axs[1].axhline(y=float(np.mean(np.array(temp_label))), color='royalblue', linestyle='dashed')
    
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Temperature')
    axs[1].set_ylim(-10, 35)
    
    fig.tight_layout()
    plt.show()
    
    return fig


# %% Loss Plots

def plot_losses(storage: Any) -> plt.Figure:
    """
    Generate a plot showing the training and testing loss curves. 
    

    Parameters
    ----------
    storage : Any
        An object containing a 'StoredValues' dictionary with keys:
        'loss' (training loss), 'Batch' (test batch indices), and 'testloss'.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    
    train_loss = storage.StoredValues['loss']
    
    # Raw training loss
    ax.plot(train_loss, alpha=0.3, color='royalblue')
    
    # Smoothed training loss (Moving Average)
    smoothed_loss = np.convolve(train_loss, np.ones(100) / 100, mode='valid')
    ax.plot(smoothed_loss, alpha=1.0, color='royalblue', label='Training (Smoothed)')
    
    # Testing loss
    test_batches = storage.StoredValues['Batch'][100:]
    test_losses = storage.StoredValues['testloss'][100:]
    ax.plot(test_batches, test_losses, alpha=1.0, color='red', label='Testing')
    
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Batches')
    ax.set_ylabel('Loss')
    ax.set_ylim(0, 0.2)
    
    # fig.tight_layout()
    # plt.show()
    
    return fig


# %% Test Execution

if __name__ == '__main__':
    # Simple dummy execution to test the script
    test_tensor = torch.rand(1)
    print(f"Plotting module loaded successfully. Dummy tensor: {test_tensor}")