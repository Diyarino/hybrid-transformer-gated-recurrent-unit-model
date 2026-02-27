# -*- coding: utf-8 -*-
"""
Main training pipeline for Probabilistic Transformer-GRU Forecasting.

This script loads configurations, initializes datasets, builds the 
Transformer-GRU model, and executes the training loop across multiple trials.

Created on Tue Jul 15 09:39:53 2025
@author: Diyar Altinses, M.Sc.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

# Local module imports
from model.transformer_GRU_model import GRUModel, TransformerEncoder
from model.base_variational import Variational
from utils.data_storage import DataStorage
from utils.generate_folder import GenerateFolder
from utils.config_plots import configure_plt
from utils.plots import plot_losses

from dataset.dummy_dataset import DummyTimeSeriesDataset


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Loads the YAML configuration file safely."""
    with open(config_path, encoding='utf-8') as yaml_file:
        # safe_load is industry standard to prevent arbitrary code execution from YAML
        return yaml.safe_load(yaml_file)


def evaluate_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> float:
    """
    Evaluates the model on the test dataset.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    dataloader : DataLoader
        The dataloader containing test/validation data.
    criterion : nn.Module
        The loss function.
    device : torch.device
        The device to perform computations on.

    Returns
    -------
    float
        The average loss over the test dataset.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)[..., 0]
            
            _ = model(data.permute(0, 2, 1))
            
            # Accessing the custom mu_values from the Variational layer (model[-1])
            loss = criterion(model[-1].mu_values, label)
            total_loss += loss.item()
            
    model.train()  # Revert back to training mode
    return total_loss / len(dataloader)


def run_training_pipeline() -> None:
    """Main execution function for setting up and running model training trials."""
    
    # --- Setup & Configuration ---
    config = load_config()
    
    # Robust device handling
    device_str = config['general'].get('device', 'cpu')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")
    
    configure_plt()

    # --- Data Loading ---
    dataset_path = os.path.join(os.getcwd(), 'resources')
    train_path = os.path.join(dataset_path, 'train_set_dummy.pt')
    test_path = os.path.join(dataset_path, 'test_set_dummy.pt')

    print("Loading datasets...")
    train_set = torch.load(train_path)
    test_set = torch.load(test_path)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=0)

    # # Extracting normalization metrics (Assuming custom dataset attributes)
    # mean, std = train_set.mean[[7]], train_set.std[[7]]
    # mean_temp, std_temp = train_set.mean[[4]], train_set.std[[4]]

    # --- Multi-Trial Training Loop ---
    num_trials = 10
    num_epochs = 20
    
    for trial in range(num_trials):
        print(f"\n{'='*40}\nStarting Trial {trial + 1}/{num_trials}\n{'='*40}")
        
        # 1. Folder Management (Using the refactored GenerateFolder class)
        folder = GenerateFolder(generate_all=False)
        folder.generate_train_folder(generate=True)
        folder.generate_data_folder(generate=True, location=folder.trainfolder) 
        train_folder_path = folder.trainfolder

        # 2. Model Initialization
        transformer_encoder = TransformerEncoder(
            input_dim=480, d_model=480, nhead=4, 
            num_encoder_layers=2, dim_feedforward=480
        )
        gru_model = GRUModel(input_dim=480, hidden_dim=480, output_dim=192, num_layers=2)
        var_model = Variational(latent_dims=192, variotional_dims=192) # Typo in original: 'variotional_dims'
        
        model = nn.Sequential(transformer_encoder, gru_model, var_model).to(device)
        
        # 3. Optimizers & Trackers
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        storage = DataStorage(['Epochs', 'Batch', 'loss', 'testloss'], 
                              show=2, line=100, header=500, precision=5)
        
        # Initial baseline evaluation before training
        current_test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Initial Test Loss: {current_test_loss:.5f}")

        # 4. Epoch Loop
        batch_idx = 0
        
        for epoch in range(num_epochs):
            model.train()
            
            for _, (data, label) in enumerate(train_loader):
                optimizer.zero_grad()
                
                data = data.to(device)
                label = label.to(device)[..., 0]
                
                # Forward pass
                prediction = model(data.permute(0, 2, 1))
                loss = criterion(prediction, label)
                
                # Backward pass & optimize
                loss.backward()
                optimizer.step()
                
                batch_idx += 1
                
                # Extract loss from the Variational layer specifically for logging
                # Detach/item() is crucial here to prevent memory leaks!
                var_loss = criterion(model[-1].mu_values, label).item()
                
                storage.Store([epoch, batch_idx, var_loss, current_test_loss])
                
                # Periodic Evaluation
                if batch_idx % 100 == 0:
                    current_test_loss = evaluate_model(model, test_loader, criterion, device)
                    
            # Step the learning rate scheduler at the end of each epoch
            scheduler.step()
            
        print(f"Trial {trial + 1} completed. Final Test Loss: {current_test_loss:.5f}")

        # 5. Save Artifacts
        print(f"Saving artifacts to {train_folder_path}...")
        torch.save(model.state_dict(), os.path.join(folder.netfolder, 'model_weights.pth'))
        torch.save(model, os.path.join(folder.netfolder, 'model.pt'))
        torch.save(storage, os.path.join(folder.datafolder, 'train_storage.pt'))
        
        # 6. Generate and save plots
        fig = plot_losses(storage)
        fig.savefig(os.path.join(train_folder_path, 'loss_plot.png'), dpi=300, bbox_inches='tight')
        
    print("\nAll trials finished successfully!")


if __name__ == '__main__':
    run_training_pipeline()