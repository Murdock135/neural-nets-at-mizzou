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
import numpy as np
import pandas as pd

def create_sequences(df, sequence_length, prediction_window=1, future_strategy='many_to_one'):
    """
    Generates input and target sequences from a DataFrame for training RNN models, 
    adjusted for different future strategies.

    Parameters:
    df (pd.DataFrame): DataFrame containing sequential data.
    sequence_length (int): Number of time steps in each input sequence.
    prediction_window (int, optional): Number of future time steps to predict.
    future_strategy (str, optional): Strategy for future prediction ('many_to_one', 'fixed_window', 'sequential').

    Returns:
    np.ndarray: An array of input sequences.
    np.ndarray: An array of targets corresponding to the sequences.
    """
    data = df.values
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length - prediction_window + 1):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)

        if future_strategy == 'many_to_one':
            # Many-to-One: The target is the next single timestep after the sequence
            target = data[i + sequence_length]
        elif future_strategy == 'fixed_window':
            # Fixed Window: The targets are the next 'prediction_window' timesteps
            target = data[i + sequence_length:i + sequence_length + prediction_window]
        elif future_strategy == 'sequential':
            # Sequential: Every timestep in the input predicts the next timestep
            target = data[i + 1:i + 1 + sequence_length]
        else:
            raise ValueError("Unsupported future strategy specified.")

        targets.append(target)

    return np.array(sequences), np.array(targets)
#----------------------------------------------------------------------------------------------------------------

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
