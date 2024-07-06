from model import *
from utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# enforcing GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ', torch.cuda.get_device_name(0))


train_path = f'E:/CMAPSS/data/train_FD00X.txt'
test_path = f'E:/CMAPSS/data/test_FD00X.txt'
RUL_path = f'E:/CMAPSS/data/RUL_FD00X.txt'


"""Trajectories FD001 and FD003 both have redundant readings which can be dropped as they are constant or rarely 
change. FD004 and FD002 both have a redundant sensor 16."""
cols_to_drop = ['opMode3', 'sensor1', 'sensor5',
                'sensor6', 'sensor10', 'sensor16',
                'sensor18', 'sensor19', 'sensor17']
labels = (['unit', 'cycles', 'opMode1', 'opMode2', 'opMode3']
          + [f'sensor{i}' for i in range(1, 22)])  # for 22 sensors


# Initialising data
train_data, test_data, RUL_data = load_data(train_path), load_data(test_path), load_data(RUL_path)

# Preparing data
for i in range(len(train_data)):
    train_data[i] = prepare_data(train_data[i])
    test_data[i] = prepare_data(test_data[i])

# Scaling data
scaler = MinMaxScaler(feature_range=(-1, 1))
for j in range(len(train_data)):
    train_data[j].iloc[:, 2:] = scaler.fit_transform(train_data[j].iloc[:, 2:])
    test_data[j].iloc[:, 2:] = scaler.transform(test_data[j].iloc[:, 2:])


# Grabbing column labels
FD_columns = [[column for column in df if column != 'RUL'] for df in train_data]

sequence_length = 10
trajectory = 0  # see readme file
batch_size = 32
epochs = 20
x, y = create_training_batch(train_data[trajectory], sequence_length, FD_columns[trajectory])
feature_count = x.shape[2]
out_dim = 1

# Preparing DataLoaders and Tensors
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(x_train).float().to(device),
                              torch.tensor(y_train).float().to(device))
val_dataset = TensorDataset(torch.tensor(x_val).float().to(device),
                            torch.tensor(y_val).float().to(device))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Creating the model
Model = DamagePropagationModel(sequence_length, feature_count, out_dim).to(device)
print(Model)

# Loss function and optimisation
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(Model.parameters(), lr=0.001)

# Let's have a look at what the dataloaders and datasets look like
print("Training data size: ", x.shape)
print("---\n")
print("Training label size: ", y.shape)

print("---\n")
print("X_train shape: ", x_train.shape)
print("X_val shape: ", x_val.shape)
print("---\n")
print("Y_train shape: ", y_train.shape)
print("Y_val shape: ", y_val.shape)
# Training the model
train_model(Model, criterion, optimizer, train_loader, val_loader, num_epochs=epochs, patience=10)




