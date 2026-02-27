# -*- coding: utf-8 -*-
"""
Deep Learning Sequence Architectures Module

This module defines various neural network architectures for processing 
and predicting sequential data, including Transformers, GRUs, LSTMs, 
and Variational Autoencoder (VAE) augmented models.

Created on %(date)s
@author: Diyar Altinses, M.Sc.
"""

import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """
    Encodes input sequences using a Transformer encoder architecture.
    """

    def __init__(
        self, 
        input_dim: int = 192, 
        d_model: int = 512, 
        nhead: int = 8, 
        num_encoder_layers: int = 3, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.0
    ):
        """
        Initializes the Transformer Encoder model.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the input features. Defaults to 192.
        d_model : int, optional
            Dimension of the internal model embeddings. Defaults to 512.
        nhead : int, optional
            Number of attention heads. Defaults to 8.
        num_encoder_layers : int, optional
            Number of stacked encoder layers. Defaults to 3.
        dim_feedforward : int, optional
            Dimension of the internal feedforward network. Defaults to 2048.
        dropout : float, optional
            Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, norm=None
        )
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input sequence.

        Parameters
        ----------
        src : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Encoded sequence of shape (batch_size, seq_len, d_model).
        """
        src = self.linear(src)
        output = self.transformer_encoder(src)
        return output


class GRUModel(nn.Module):
    """
    Encodes input sequences using a Gated Recurrent Unit (GRU) model.
    """

    def __init__(
        self, 
        input_dim: int = 512, 
        hidden_dim: int = 128, 
        output_dim: int = 96, 
        num_layers: int = 5
    ):
        """
        Initializes the GRU Model.

        Parameters
        ----------
        input_dim : int, optional
            Dimension of the input features. Defaults to 512.
        hidden_dim : int, optional
            Dimension of the hidden state. Defaults to 128.
        output_dim : int, optional
            Dimension of the output. Defaults to 96.
        num_layers : int, optional
            Number of GRU layers. Defaults to 5.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence through the GRU and linear decoding layer.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted output of shape (batch_size, output_dim).
        """
        # _ represents the final hidden state; we extract all outputs instead
        hidden, _ = self.gru(x)
        
        # Note: Original code passes the entire sequence through the linear layer,
        # contrary to the comment saying "Nur den letzten hidden state nutzen"
        output = self.fc(hidden[:, -1, :])
        return output


    
if __name__ == '__main__':
    # --- Example Usage and Standalone Testing ---
    
    # Hyperparameters
    batch_size = 32
    input_size = 288
    d_model = 512
    nhead = 8
    num_encoder_layers = 2
    dim_feedforward = 2048
    hidden_dim = 128
    output_size = 96
    num_layers = 2

    # Instantiate models
    transformer_encoder = TransformerEncoder(
        input_size, d_model, nhead, num_encoder_layers, dim_feedforward
    )
    gru_model = GRUModel(d_model, hidden_dim, output_size, num_layers)
    
    # Chain models together sequentially
    model = nn.Sequential(transformer_encoder, gru_model)
    
    # Create dummy input sequence
    input_data = torch.randn(batch_size, 16, input_size)

    # Execute forward pass
    output = model(input_data)

    # Print resulting tensor shape (Expected: [32, 16, 96])
    print(f"Chained model output shape: {output.shape}")