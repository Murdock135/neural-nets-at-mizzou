import os
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from PIL import Image
import torch

# the following function are specific to the Online retail dataset
# used for the RNN project and cannot be used for any other project
# ---------------------------------------------------------------------
def create_sequences(df, sequence_length, prediction_window = 1):
    """
    Generates input and target sequences from a DataFrame to prepare data for training RNN models.
    
    This function creates overlapping sequences of a specified length from a DataFrame,
    where each sequence is used as input for RNN predictions. The targets are determined
    by the prediction_window, allowing for both many-to-one and many-to-many predictions.

    Parameters:
    df (pd.DataFrame): DataFrame containing sequential data. Each row is expected to be 
                       a time step in the sequence.
    sequence_length (int): The number of time steps in each input sequence.
    prediction_window (int, optional): The number of future time steps to predict. A value of
                                       1 indicates a many-to-one prediction, while a value greater
                                       than 1 indicates a many-to-many prediction.

    Returns:
    np.ndarray: An array of input sequences.
    np.ndarray: An array of targets corresponding to the sequences. Targets can be a single
                time step or a sequence of time steps, depending on the prediction_window.
    """
    
    sequences = []
    targets = []
    data = df.values

    for i in range(len(data) - sequence_length):
        # create input sequence
        sequence = data[i:(i + sequence_length)]

        # make the last time step the target (many to one)
        target = data[i+sequence_length : + i+sequence_length+prediction_window]

        # store sequence and target
        sequences.append(sequence)
        targets.append(target)

    return np.array(sequences), np.array(targets)# -----------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, X, y, loss_func):
        self.loss_func = loss_func
        self.X = X
        self.y = y


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):  
            # apply transforms
            x_i, y_i = self.transforms(self.X[i], self.y[i])

            return x_i, y_i
    
    def transforms(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float)
        
        # Convert y to tensor based on specified loss function
        if self.loss_func == str.lower("cross entropy"):
            # For cross entropy loss, y should be of type torch.long (class labels)
            y_tensor = torch.tensor(y, dtype=torch.long)
        elif self.loss_func == str.lower("mse"):
            # For MSE loss, y should be a floating point tensor (continuous values)
            y_tensor = torch.tensor(y, dtype=torch.float)
        else:
            raise ValueError("Unsupported loss function specified.")
        
        return X_tensor, y_tensor

from torch.utils.data import DataLoader, random_split

def CustomDataLoader(dataset, training_portion, batch_size=32, shuffle=True):
    size = len(dataset)
    train_size = int(training_portion * size)
    val_size = size - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size, shuffle)
    val_loader = DataLoader(val_set, batch_size, shuffle)

    return train_loader, val_loader
