from model import *
from utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

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
batch_size = 64
epochs = 20
x, y = create_training_batch(train_data[trajectory], sequence_length, FD_columns[trajectory])
feature_count = x.shape[2]
out_dim = 1

# Preparing DataLoaders and Tensors
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(x_train).float(),
                              torch.tensor(y_train).float().unsqueeze(1))
val_dataset = TensorDataset(torch.tensor(x_val).float(),
                            torch.tensor(y_val).float().unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Creating the model
Model = DamagePropagationModel(feature_count, out_dim).to(device)
print(Model)

# Loss function and optimisation
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(Model.parameters(), lr=0.01)
print(Model.parameters())

# Training the model
#train_model(Model, criterion, optimizer, train_loader, val_loader, num_epochs=epochs, patience=10)

Model.load_state_dict(torch.load('best_model.pt'))
Model.eval()
Model.to(device)

# Predictions on training data
x_train_tensor = torch.tensor(x_train).float().to(device)

# x_train and y_train are arrays. x_train should just be opModes and sensor readings. y_train should be the
# corresponding RUL vector
print(y_train)

with torch.no_grad():
    y_train_pred = Model(x_train_tensor)

y_train_pred = y_train_pred.cpu().numpy()
y_train_tensor = torch.tensor(y_train).float().unsqueeze(1).cpu().numpy()

plt.figure(figsize=(10, 5))
plt.plot(y_train[0:200], label='Actual')
plt.plot(y_train_pred[0:200], label='Predicted')
plt.xlabel('Sample index')
plt.ylabel('Value')
plt.legend()
plt.title('Actual vs Predicted on Training Data')
plt.show()