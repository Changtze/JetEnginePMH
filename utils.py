import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter as svg

# change file path as appropriate
train_path = f'E:/CMAPSS/data/train_FD00X.txt'
test_path = f'E:/CMAPSS/data/test_FD00X.txt'
RUL_path = f'E:/CMAPSS/data/'


"""Trajectories FD001 and FD003 both have redundant readings which can be dropped as they are constant or rarely 
change. FD004 and FD002 both have a redundant sensor 16."""
cols_to_drop = ['opMode3', 'sensor1', 'sensor5',
                'sensor6', 'sensor10', 'sensor16',
                'sensor18', 'sensor19', 'sensor17']
labels = (['unit', 'cycles', 'opMode1', 'opMode2', 'opMode3']
          + [f'sensor{i}' for i in range(1, 22)])  # for 22 sensors


def denoise(df, window_length, polyorder):
    u = df.groupby('unit')['cycles'].max().astype(int)
    for max_life in u:
        for j in range(2, df.shape[1]):
            df.iloc[1:max_life+1, j] = svg(df.iloc[1:max_life+1, j], window_length, polyorder)
    return None


    for unit in range(1, df['unit'].max()+1):
        for j in range(2, df.shape[1]):
            df.iloc[:, j] = svg(df.iloc[:, j], window_length, polyorder)
    return None


def load_data(filepath, rul=False):
    data = []
    # Creating lists which hold dataframes for each test trajectory
    if not rul:
        for i in range(1, 5):
            data.append(pd.read_csv(filepath.replace('X', str(i)),
                                    names=labels, delimiter='\s+',
                                    dtype=np.float32))
            if i == 1 or i == 3:
                data[i - 1].drop(cols_to_drop, axis=1, inplace=True)
            if i == 4 or i == 2:
                data[i - 1].drop('sensor16', axis=1, inplace=True)
        return data
    else:
        for i in range(1, 5):
            data.append(pd.read_csv(filepath.replace('X', str(i)),
                                    delimiter='\s+', header=None,
                                    dtype=np.float32))

        return data  # a list


def prepare_data(data: pd.DataFrame):
    # Drops redundant columns and adds an RUL column for training
    rul = pd.DataFrame(data.groupby('unit')['cycles'].max()).reset_index()
    rul.columns = ['unit', 'max']

    data = data.merge(rul, on=['unit'], how='left')

    data['RUL'] = data['max'] - data['cycles']
    data.drop('max', axis=1, inplace=True)
    return data


def create_training_sequence(df, seq_length, seq_cols):
    """function to prepare training data into (samples, time steps, features)
    df = training dataframe
    seq_length = look-back period
    seq_cols = feature columns"""

    data_array = df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []

    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        lstm_array.append(data_array[start:stop, :])

    return np.array(lstm_array)


def create_target_sequence(df, seq_length, label):
    data_array = df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length-1:num_elements+1]


def create_training_batch(df, seq_length, columns):
    x = np.concatenate(list(list(create_training_sequence(df[df['unit'] == i], seq_length, columns)) for i in df['unit'].unique()))
    y = np.concatenate(list(list(create_target_sequence(df[df['unit'] == i], seq_length, 'RUL')) for i in df['unit'].unique()))
    return x, y


def create_test_batch(df, seq_length, columns):
    x = np.concatenate(list(list(create_training_sequence(df[df['unit'] == i], seq_length, columns)) for i in df['unit'].unique()))
    y = np.concatenate(list(list(create_target_sequence(df[df['unit'] == i], seq_length, 'RUL')) for i in df['unit'].unique()))
    return x, y