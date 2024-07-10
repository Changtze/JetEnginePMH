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

# Data directories
train_path = 'D:/Data Science/CMAPSS Engine Simulations/Data/train_FD00X.txt'
test_path = 'D:/Data Science/CMAPSS Engine Simulations/Data/test_FD00X.txt'
RUL_path = 'D:/Data Science/CMAPSS Engine Simulations/Data/RUL_FD00X.txt'
# train_path = f'E:/CMAPSS/data/train_FD00X.txt'
# test_path = f'E:/CMAPSS/data/test_FD00X.txt'
# RUL_path = f'E:/CMAPSS/data/RUL_FD00X.txt'

"""Trajectories FD001 and FD003 both have redundant readings which can be dropped as they are constant or rarely 
change. FD004 and FD002 both have a redundant sensor 16."""
cols_to_drop = ['opMode3', 'sensor1', 'sensor5',
                'sensor6', 'sensor10', 'sensor16',
                'sensor18', 'sensor19', 'sensor17']
labels = (['unit', 'cycles', 'opMode1', 'opMode2', 'opMode3']
          + [f'sensor{i}' for i in range(1, 22)])  # for 22 sensors

# Initialising data
train_data, test_data, RUL_data = load_data(train_path), load_data(test_path), load_data(RUL_path, True)

# Preparing data
for i in range(len(train_data)):
    train_data[i] = prepare_data(train_data[i])
    test_data[i] = prepare_data(test_data[i])

# Scaling data
scaler = MinMaxScaler(feature_range=(-1, 1))
for j in range(len(train_data)):
    train_data[j].iloc[:, 2:-1] = scaler.fit_transform(train_data[j].iloc[:, 2:-1])
    test_data[j].iloc[:, 2:-1] = scaler.transform(test_data[j].iloc[:, 2:-1])


# Grabbing column labels
FD_columns = [[column for column in df if column != 'RUL'] for df in train_data]

sequence_length = 20
trajectory = 0  # see readme file
batch_size = 32
epochs = 100
x, y = create_training_batch(train_data[trajectory], sequence_length, FD_columns[trajectory])
x_test, y_test = create_test_batch(test_data[trajectory], sequence_length, FD_columns[trajectory])
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
Model = DamagePropagationLSTM(feature_count, out_dim).to(device)
print(Model)

# Loss function and optimisation
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(Model.parameters(), lr=0.0001)

# Training the model (note that model1 is a list of MSE values per epoch)
model_filename = 'LSTM_model1.pt'
model1 = train_model(Model, criterion, optimizer,
                     train_loader, val_loader, num_epochs=epochs,
                     patience=10, filename=model_filename)
epoch_id = np.arange(1, epochs+1, 1)

# Loading trained model
model_path = f'D:/Data Science/CMAPSS Engine Simulations/code/tft_new_311/{model_filename}'
Model.load_state_dict(torch.load(model_path))
Model.to(device)
Model.eval()

# Predictions on training and test data
x_train_tensor = torch.tensor(x_train).float().to(device)
x_test_tensor = torch.tensor(x_test).float().to(device)

with torch.no_grad():
    y_train_pred = Model(x_train_tensor)
    y_test_pred = Model(x_test_tensor)

y_train_pred = y_train_pred.cpu().numpy()
y_train_tensor = torch.tensor(y_train).float().unsqueeze(1).cpu().numpy()

y_test_pred = y_test_pred.cpu().numpy()
y_test_tensor = torch.tensor(y_test_pred).float().unsqueeze(1).cpu().numpy()

# Creating appropriate RUL vectors for plotting
elapsed_life = pd.DataFrame(test_data[trajectory].groupby('unit')['cycles'].max()).reset_index()
elapsed_life = elapsed_life['cycles'].values
RUL_truth = RUL_data[trajectory].to_numpy().squeeze(1)
max_life = RUL_truth + elapsed_life
RUL_vec = np.arange(int(max_life[0]), 0, -1)
for i in max_life[1:]:
    RUL_vec = np.concatenate((RUL_vec, np.arange(int(i), 1, -1)))

print(y_train[0:200])
# Plotting predictions on training data
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
axs[0].plot(y_train[0:100], label='Actual')
axs[0].plot(y_train_pred[0:100], label='Predicted')
axs[0].set_title('Model performance on training data')
axs[0].set_xlabel('Sample index')
axs[0].set_ylabel('RUL')
axs[0].legend()

# Plotting predictions on test data
axs[1].plot(RUL_vec[0:400], label='Actual')
axs[1].plot(y_test_pred[0:400], label='Predicted')
axs[1].set_title('Model performance on test data')
axs[1].set_xlabel('Sample index')
axs[1].set_ylabel('RUL')
axs[1].legend()

plt.show()
