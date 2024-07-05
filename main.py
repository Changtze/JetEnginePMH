from model import DamagePropagationModel
from utilsimport *
import torch

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
for j in range(len(train_data)):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data[j].iloc[:, 2:] = scaler.fit_transform(train_data[j].iloc[:, 2:])
    test_data[j].iloc[:, 2:] = scaler.transform(test_data[j].iloc[:, 2:])
