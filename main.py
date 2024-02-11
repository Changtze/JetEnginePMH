import pandas as pd
import torch
import filterpy as fp
import numpy as np
import math

from torch import nn
from scipy.signal import savgol_filter as svg
from matplotlib import pyplot as plt


train_path = "D:/SimulatedData/"

test_path = "D:/SimulatedData/"


def create_headers():
    new_labels = ['Unit', 'Cycles', 'M1', 'M2', 'M3']
    for j in range(1, 22):  # 21 sensors exist
        new_labels.append(f'S{j}')
    return new_labels


def create_predictors_and_targets(filtered_dataframe: pd.DataFrame | None = None):
    """
    Extracts predictor and target variables from a dataframe passed though denoise()
    :param filtered_dataframe: dataframe containing predictor variables and target variables
    :return:
    """
    if filtered_dataframe is None:
        raise Exception("No data provided.")
    targets = filtered_dataframe['Cycles']
    predictors = filtered_dataframe.iloc[:, 2:26]
    # to be completed
    # predictor will be cycles in some sense.
    # training data runs till failure.
    # test data cycles end some time before failure
    return predictors, targets


def create_data_set(path: str | None = None, labels: list | None = None):
    """
    Creates a prepared dataset, eliminating null columns and adding necessary user-defined headers
    :param path: file path
    :param labels: user-provided column headers
    :return: returns the prepared df
    """
    if path is None:
        raise Exception("No path provided.")

    if labels:  # user-provided headers
        data = pd.read_table(f'{path}', sep='\s+', na_values=['NaN'], names=labels)  # space delimited txt files
    else:
        data = pd.read_table(f'{path}', sep='\s+', na_values=['Nan'])

    return data


def denoise(noisy_data: pd.DataFrame | None = None, denoiser: int = 1):
    """
    Takes a dataframe of a complete set of trajectories and returns a list of
    separate dataframes of filtered sensor data for each specific unit
    Implements a noise filter based on user-provided id for a specific engine unit
    :param noisy_data: sensor data to be filtered
    :param denoiser: id selection for filter,
    1 for Savitzky-Golay...
    :return filtered_dataframes: list of filtered dataframes
    """
    if noisy_data is None:
        raise Exception("No data provided")

    poly_order = 3  # Savitzky-Golay parameter
    cycles_to_failure = noisy_data['Cycles'].max()  # window length for Savitzky-Golay

    filtered_data = noisy_data.copy(deep=True)

    for noisy_sensor in range(1, 22):  # iterating through 21 sensors
        noise = noisy_data[f'S{noisy_sensor}']
        filtered_data[f'S{noisy_sensor}'] = svg(noise.to_numpy(), cycles_to_failure, poly_order)

    return filtered_data


def data_overview(dataframe: pd.DataFrame | None = None):
    print(f'Information: ' + dataframe.info() + "\n")
    print(f'First 5 rows: ' + dataframe.head() + "\n")
    return None


def compare_data(filtered_data: pd.DataFrame | None = None, noisy_data: pd.DataFrame | None = None):
    """
    Compares filtered data to noisy
    :param filtered_data: filtered_data pd.Dataframe containing filtered data
    :param noisy_data: noisy_data pd.DataFrame containing noisy sensor data
    :return None: returns None, this function just provides plots
    """
    if filtered_data is None or noisy_data is None:
        raise Exception("Not enough datasets provided")

    unit = filtered_data['Unit'].max()
    for s in range(1, 22):
        noisy_sensor_data = noisy_data[f'S{s}']
        filtered_sensor_data = filtered_data[f'S{s}']
        cycles_to_failure = np.arange(1, noisy_data['Cycles'].shape[0]+1, 1)
        plt.plot(cycles_to_failure, noisy_sensor_data, color="g")
        plt.plot(cycles_to_failure, filtered_sensor_data, color="r")
        plt.xlabel("Cycle number")
        plt.ylabel("Reading")
        plt.title(f"Unit {unit} Sensor {s} time history")
        plt.legend(["Unfiltered", "Savitzky-Golay"])
        plt.show()

    return None


def plot_sensor_data(unit_data: list | None = None, c: str = "red"):
    """
    plots sensor data for each engine unit
    :param unit_data: list of dataframes containing sensor data for each engine unit
    :param c: user-specified colour, default is red
    :return:
    """
    if unit_data is None:
        raise Exception("No list provided")
    number_of_units = len(unit_data)  # number of units
    for i in range(0, 1):  # for each engine unit (limited to first engine unit for now)
        cycles = np.arange(1, len(unit_data[i]) + 1)  # number of cycles each engine lasts for
        for sensor in range(1, len(unit_data[i].axes[1])-4):  # iterating through all sensors
            plt.plot(cycles, unit_data[i][f'S{sensor}'], color=c, label=f"Sensor {i} reading ")
            plt.xlabel("Cycle number")
            plt.ylabel("Reading")
            plt.title(f"Sensor {sensor} readings till failure")
            plt.show()

    return None


def get_sensor_data(dataframe: pd.DataFrame | None = None):
    """
    creates dataframes for each unit with their cycles, o. modes and sensor readings
    :param dataframe : contains all engine units, cycles, operational modes and all sensors readings
    :return sensor_data : returns a list containing a dataframe of sensor readings for each engine unit
    """
    if dataframe is None:
        raise Exception("No data provided")

    no_of_sensors = 26
    sensor_data = []
    engine_units = dataframe['Unit'].max()

    for unit in range(1, engine_units+1):  # iterating through each engine unit
        unit_data = dataframe['Unit'] == unit
        unit_sensor_data = dataframe[unit_data]
        sensor_data.append(unit_sensor_data)
    return sensor_data


def normalise(unscaled_data: pd.DataFrame | None = None):
    """
    takes a dataframe of a single unit's mode and sensor data and normalises all the sensor readings using
    min-max normalisation
    :param unscaled_data:
    :return:
    """
    scaled_data = unscaled_data.copy(deep=True)

    if unscaled_data is None:
        raise Exception("No data provided")

    for sensor in range(1, 22):
        s_min = unscaled_data[f'S{sensor}'].min()
        s_max = unscaled_data[F'S{sensor}'].max()
        s = unscaled_data[f'S{sensor}']
        scaled_data[f'S{sensor}'] = (s - s_min) / (s_max - s_min)

    return scaled_data


def prepare_data():
    # raw data has no headers
    new_columns = create_headers()

    # list of pd.DataFrames for each training and test trajectories
    training_data = []
    test_data = []

    # loading data
    for i in range(1, 5):  # change range to (1,5) for all 4 test cases
        train_dataset = create_data_set(f'{train_path}train_FD00{i}.txt', labels=new_columns)
        train_dataset.columns = new_columns
        training_data.append(train_dataset)

        test_dataset = create_data_set(f'{test_path}test_FD00{i}.txt', labels=new_columns)
        test_dataset.columns = new_columns
        test_data.append(test_dataset)

    return training_data, test_data


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def convert_to_tensor(df: pd.DataFrame | None = None):
    if df is None:
        raise Exception("Dataframe not provided.")
    return torch.from_numpy(df.values).to(get_device())


def optimiser(learning_rate: float = None):
    if learning_rate is None:
        raise Exception("Learning rate provided)")
    return torch.optim.Adam(params, lr=learning_rate)


def main():

    # grabbing training and test sets
    training_data, test_data = prepare_data()

    # dataframes of unit sensor readings for FD0001
    s_train_unfiltered = get_sensor_data(training_data[0])
    s_train_filtered = [denoise(test_set) for test_set in s_train_unfiltered]

    # normalising data due to large scale difference
    s_train_filtered_normalised = normalise(s_train_filtered[0])

    # predictors: mode + sensor readings, target: cycles
    ms, c = create_predictors_and_targets(s_train_filtered_normalised)

    # PyTorch compatibility
    ms_numpy = ms.to_numpy()
    ms_tensor = torch.from_numpy(ms_numpy)

    dtype = torch.float
    device = get_device()

    training_data = torch.tensor(ms_tensor, dtype=dtype, device=device)

    # training parameters
    input_size = 24  # 3 operational modes & 21 sensors
    hidden_layers = 32
    recurrent_layers = 2

    # hyperparameters
    learning_rate = 0.005  # starting point


if __name__ == "__main__":
    main()
