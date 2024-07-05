import torch.nn as nn


class DamagePropagationModel(nn.Module):
    def __init__(self, seq_len, feature_count, out_dim):
        super(DamagePropagationModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=feature_count, hidden_size=128,
                             batch_first=True, return_sequences=True)

        self.batch_norm = nn.BatchNorm1d(num_features=128)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64,
                             batch_first=True, return_sequences=False)

        self.dropout2 = nn.Dropout(0.3)
        self.dense = nn.Linear(in_features=64, out_features=out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.batch_norm(x.transpose(1,2)).transpose(1,2)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.dense(x)
        x = self.activation(x)
        return x