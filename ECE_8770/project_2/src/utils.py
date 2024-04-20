from box import box
import pandas as pd
import os
import numpy as np
import tomllib
import matplotlib.pyplot as plt
from .models import FlexibleRNN
import torch.optim as optim
import torch.nn as nn


def load_config(path_to_config) -> dict:
    with open(path_to_config, "rb") as f:
        config: dict = tomllib.load(f)
        return config
    
def get_model(config, no_features, output_size):
    model = FlexibleRNN(
        input_size=no_features,
        hidden_size=config["model"]["hidden_size"],
        output_size=output_size,
        num_layers=config["model"]["num_layers"],
        rnn_type=config["model"]["type"],
        prediction_window=config["model"]["prediction_window"],
        future_strategy=config["model"]["future_strategy"]
    )

    return model

def get_optimizer(model, config):
    optimizer_config = config['training']['optimizer']
    
    optimizer_type = optimizer_config['type']
    learning_rate = optimizer_config['learning_rate']

    if optimizer_type.lower() == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate)
    
def get_criterion(config: str):
    criterion = config['training']['criterion']

    if criterion.lower() == "cross entropy":
        return nn.CrossEntropyLoss()
    elif criterion.lower() == "mse":
        return nn.MSELoss()


class ResultsPlotter:
    def __init__(self, exp_dir, results, fold_idx=None):
        '''Args:
            exp_dir(string)- directory to export visualizations to
            results(dictionary)- a dict of results where keys are metrics and values are lists
            '''
        self.exp_dir = exp_dir
        self.results: dict = results
        self.fold_idx = fold_idx

    def plot_classification_results(self):

        # plot training loss + validation loss on same graph
        plt.figure()
        plt.plot(self.results['training loss'], label='training loss')
        plt.plot(self.results['validation loss'], label='validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.exp_dir, f"{self.fold_idx+1 if self.fold_idx is not None else ''}_loss.png"))

        # Plot accuracy and F1 score on the same graph
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['accuracy'], label='Accuracy', color='blue')
        plt.plot(self.results['f1 score'], label='F1 Score', color='green')
        plt.plot(self.results['precision'], label='precision', color='purple')
        plt.plot(self.results['recall'], label='recall', color='cyan')

        plt.title('Performance Metrics: Accuracy, F1, Precision, and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.ylim(0, 1)  # Set y-axis limits for F1 score
        plt.grid(True)
        plt.legend(loc='lower right')

        # Create a secondary y-axis for accuracy (scaled to percentage)
        ax = plt.gca().twinx()
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)  # Set y-axis limits for accuracy

        plt.savefig(os.path.join(self.exp_dir,f'{self.fold_idx if self.fold_idx is not None else ""}_accuracy, f1, precision, recall.png'))

    def plot_regression_results(self):
            # Plot Training Loss + Validation Loss on the same graph specifically for regression tasks
            plt.figure()
            plt.plot(self.results['training loss'], label='training loss')
            plt.plot(self.results['validation loss'], label='validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.exp_dir, f"{self.fold_idx if self.fold_idx is not None else ''}_regression_loss.png"))

    def plot_from_csv(self):
         pass
