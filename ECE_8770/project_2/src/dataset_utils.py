import os
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.utils.data.dataloader import default_collate
from PIL import Image
import torch

# the following function are specific to the Online retail dataset
# used for the RNN project and cannot be used for any other project
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd

def create_sequences(df, sequence_length, column=None, prediction_window=1, future_strategy='many_to_one'):
    data = df.values if column is None else df[column].to_numpy().reshape(-1,1)
    indices = df.index.to_numpy().reshape(-1,1)

    sequences = []
    targets = []
    target_indices = []

    for i in range(len(data) - sequence_length - prediction_window + 1):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)

        if future_strategy == 'many_to_one':
            # Many-to-One: The target is the next single timestep after the sequence
            target = data[i + sequence_length]
            target_index = indices[i + sequence_length] # Index of the target

        elif future_strategy == 'fixed_window':
            # Fixed Window: The targets are the next 'prediction_window' timesteps
            target = data[i + sequence_length:i + sequence_length + prediction_window]
            target_index = indices[i + sequence_length:i + sequence_length + prediction_window]  # Indices for all targets

        elif future_strategy == 'sequential':
            # Sequential: Every timestep in the input predicts the next timestep
            target = data[i + 1:i + 1 + sequence_length]
            target_index = indices[i + 1:i + 1 + sequence_length]  # Indices for all sequence predictions

        else:
            raise ValueError("Unsupported future strategy specified.")
        
        targets.append(target)
        target_indices.append(target_index)

    return np.array(sequences), np.array(targets), np.array(target_indices)
#----------------------------------------------------------------------------------------------------------------

def CustomDataLoader(dataset, training_portion, batch_size=32, shuffle=True, collate_fn=default_collate):
    size = len(dataset)
    train_size = int(training_portion * size)
    val_size = size - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size, shuffle, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size, shuffle, collate_fn=collate_fn)

    return train_loader, val_loader

# Dataset class to handle tabular data
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

def temporal_collate_fn(batch):
    # Extract datetime indices and store them separately
    datetime_indices = np.array([item[0] for item in batch])
    # Collate the rest of the data normally
    batch_data = [item[1:] for item in batch]  # This creates a new batch excluding datetime indices
    collated_data = default_collate(batch_data)  # Let default_collate handle the usual collation

    # collated_data will be a list of tensors (inputs and targets)
    inputs, targets = collated_data
    return datetime_indices, inputs, targets

class TemporalCustomDataset(CustomDataset):
    def __init__(self, X, y, datetime_indices, loss_func):
        super().__init__(X, y, loss_func)
        self.datetime_indices = np.array(datetime_indices)  # This should be an array or tensor of datetime indices

    def __getitem__(self, i):
        # Tranform inputs and outputs
        x_i, y_i = self.transforms(self.X[i], self.y[i])

        # Return the datetime index along with the usual items
        return self.datetime_indices[i], x_i, y_i
    
