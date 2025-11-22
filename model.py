import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=4):
        super().__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None):
        if h0 is None:
            h0 = torch.zeros(
                self.layer_dim,
                x.size(0),
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )
        x, hn = self.gru(x, h0)
        out = x[:, -1, :]
        return self.fc(out), hn
