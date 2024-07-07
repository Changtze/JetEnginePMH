import torch.nn as nn
import torch


class DamagePropagationModel(nn.Module):
    def __init__(self, feature_count, out_dim):
        super(DamagePropagationModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size=feature_count, hidden_size=128, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(num_features=128)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.dense = nn.Linear(in_features=64, out_features=out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout1(x)
        x, (hn, cn) = self.lstm2(x)
        x = self.dropout2(hn[-1])
        x = self.dense(x)
        x = self.activation(x)
        return x


# training function
def train_model(model, criterion, optimizer,
                train_loader, val_loader, num_epochs=60,
                patience=10):
    best_loss = float('inf')
    patience_counter = 0
    print("Model initialised successfully. Beginning training on {dev}...".format(dev=torch.cuda.get_device_name(0)))
    for epoch in range(num_epochs):
        loss = 0.0
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data).to(device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            loss += loss.item() * data.size(0)

        epoch_loss = loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data).to(device)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * data.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping!')
            break


# Enforcing GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
