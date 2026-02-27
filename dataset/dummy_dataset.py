# -*- coding: utf-8 -*-
"""
Synthetic Data Generator for Pipeline Verification

This script generates dummy datasets with random noise to test and verify 
the Transformer-GRU forecasting pipeline without exposing confidential data.
"""

import os
import torch
from torch.utils.data import Dataset


class DummyTimeSeriesDataset(Dataset):
    """
    A mock dataset mirroring the exact structure, tensor shapes, and 
    normalization attributes of the original time-series dataset.
    """
    def __init__(self, num_samples: int = 500) -> None:
        # Data shape per sample: (480, 16)
        # With batch_size=32 -> torch.Size([32, 480, 16])
        self.data = torch.randn(num_samples, 480, 16)
        
        # Label shape per sample: (192, 2)
        # With batch_size=32 -> torch.Size([32, 192, 2])
        self.labels = torch.randn(num_samples, 192, 2)
        
        # Normalization Attributes
        # main.py accesses indices like train_set.mean[[7]]
        # We provide a tensor of length 16 to match the last dimension of the data
        self.mean = torch.zeros(16)
        self.std = torch.ones(16)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def generate_resources() -> None:
    """Creates the resources directory and saves the synthetic train/test sets."""
    base_dir = os.path.join(os.getcwd(), 'resources')
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Generating synthetic datasets in: {base_dir}")
    
    # Generate mock datasets (1000 samples for training, 200 for testing)
    train_set = DummyTimeSeriesDataset(num_samples=100)
    test_set = DummyTimeSeriesDataset(num_samples=20)
    
    train_path = os.path.join(base_dir, 'train_set_dummy.pt')
    test_path = os.path.join(base_dir, 'test_set_dummy.pt')
    
    torch.save(train_set, train_path)
    torch.save(test_set, test_path)
    
    print("Success! 'train_set.pt' and 'test_set.pt' generated with exact dimensions.")
    print("Expected Data shape per batch (BS=32): torch.Size([32, 480, 16])")
    print("Expected Label shape per batch (BS=32): torch.Size([32, 192, 2])")


if __name__ == '__main__':
    generate_resources()