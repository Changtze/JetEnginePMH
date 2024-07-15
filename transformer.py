import torch.nn as nn


class DamagePropagationTransformer(nn.Module):
    def __init__(self, feature_count, out_dim):
        super(DamagePropagationTransformer, self).__init__()

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_count, nhead=8), num_layers=2)
        self.dense = nn.Linear(in_features=feature_count, out_features=out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.transformer(x)
        x = self.dense(x)
        x = self.activation(x)
        return x