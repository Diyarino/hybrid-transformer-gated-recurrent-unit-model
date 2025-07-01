# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import os
import torch

from model.transformer_GRU_model import GRUModel, TransformerEncoder

# %% config

device = 'cuda'

# %% application

path_model = os.path.join('./training/25_02_19__17_10_39', 'model', 'model_weights.pth')
transformer_encoder = TransformerEncoder(input_dim = 288, d_model = 512, nhead = 4, 
                                         num_encoder_layers = 2, dim_feedforward = 2048)
gru_model = GRUModel(input_dim = 512, hidden_dim = 128, output_dim = 96, num_layers = 2)
model = torch.nn.Sequential(transformer_encoder, gru_model).to(device)
model.load_state_dict(torch.load(path_model, map_location=device))

