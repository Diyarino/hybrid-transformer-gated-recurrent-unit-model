# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% application

class TransformerEncoder(torch.nn.Module):
    """
    Encodes input sequences using a Transformer encoder.

    Args:
        input_dim (int): Dimension of the input features.
        d_model (int): Dimension of the model.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Returns:
        torch.Tensor: Encoded sequence.
    """
    def __init__(self, input_dim = 192, d_model = 512, nhead = 8, num_encoder_layers = 3, 
                 dim_feedforward = 2048, dropout=0.2):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.linear = torch.nn.Linear(input_dim, d_model)

    def forward(self, src):
        """
        Encodes the input sequence.

        Args:
            src (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Encoded sequence of shape (batch_size, seq_len, d_model).
        """
        src = self.linear(src)
        output = self.transformer_encoder(src)
        return output

class GRUModel(torch.nn.Module):
    """
    Encodes input sequences using a GRU model.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden state.
        output_dim (int): Dimension of the output.
        num_layers (int): Number of GRU layers.

    Returns:
        torch.Tensor: Predicted output.
    """
    def __init__(self, input_dim = 512, hidden_dim = 128, output_dim = 96, num_layers = 5):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Processes the input sequence through the GRU and linear layer.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Predicted output of shape (batch_size, output_dim).
        """
        hidden, _ = self.gru(x)
        output = self.fc(hidden[:, -1, :])  # Nur den letzten hidden state nutzen
        return output

# %% test

if __name__ == '__main__':
	# Hyperparameter (anpassen nach Bedarf)
    batch_size = 32
    input_size = 288
    d_model = 512
    nhead = 8
    num_encoder_layers = 2
    dim_feedforward = 2048
    hidden_dim = 128
    output_size = 96
    num_layers = 2

    # Modell instanziieren
    transformer_encoder = TransformerEncoder(input_size, d_model, nhead, num_encoder_layers, dim_feedforward)
    gru_model = GRUModel(d_model, hidden_dim, output_size, num_layers)
    model = torch.nn.Sequential(transformer_encoder, gru_model)
    
    # Beispielhafte Eingabe
    input_data = torch.randn(batch_size, 16, input_size)

    # Vorwärtspass
    output = model(input_data)

    print(output.shape)  # Ausgabe: torch.Size([32, 70])
