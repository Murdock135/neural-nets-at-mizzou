from box import box
import pandas as pd
import os
import numpy as np
import tomllib
import matplotlib.pyplot as plt
from .models import FlexibleRNN
import torch.optim as optim
import torch.nn as nn
from datetime import datetime


def load_config(path_to_config) -> dict:
    with open(path_to_config, "rb") as f:
        config: dict = tomllib.load(f)
        return config
    
def get_model(config, no_features, output_size, rnn_type=None, future_strategy=None):
    model = FlexibleRNN(
        input_size=no_features,
        hidden_size=config["model"]["hidden_size"],
        output_size=output_size,
        num_layers=config["model"]["num_layers"],
        rnn_type=rnn_type if rnn_type is not None else config["model"]["type"],
        prediction_window=config["model"]["prediction_window"],
        future_strategy=future_strategy if future_strategy is not None else config["model"]["future_strategy"]
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
    def __init__(self, exp_dir, results, fold_idx=None, config=None, forecast_results = None, model_type=None, sequence_len=None, future_strategy=None, user_optimizer=None, learning_rate=None):
        '''Args:
            exp_dir(string)- directory to export visualizations to
            results(dictionary)- a dict of results where keys are metrics and values are lists
            '''
        self.exp_dir = exp_dir
        self.results: dict = results
        self.fold_idx = fold_idx

        if forecast_results is not None:
            self.forecast_results = forecast_results # dict{'predictions':[], 'truths':[], 'datetime indices':[]}

        # unpack config
        if config is not None:
            self.config = config

            self.optimizer_type = user_optimizer if user_optimizer is not None else config['training']['optimizer']['type']
            self.lr = learning_rate if learning_rate is not None else config['training']['optimizer']['learning_rate']
            self.batch_size = config['training']['batch_size']
            self.criterion = config['training']['criterion']
            self.rnn_type = model_type if model_type is not None else config['model']['type']
            self.seq_len = sequence_len if sequence_len is not None else config['model']['sequence_length']
            self.future_strategy = future_strategy if future_strategy is not None else config['model']['future_strategy']

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

        # Adding configuration details below the plot
        plt.figtext(0.5, 0.01, self.format_config_details(), ha="center", fontsize=9, wrap=True)
        # plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make room for text
        plt.subplots_adjust(bottom=0.3)

        # save fig
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        fold_part = f"fold_{self.fold_idx}_" if self.fold_idx is not None else ""
        plt.savefig(os.path.join(self.exp_dir, f"f{timestamp}_{fold_part}_classification_results"))

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

            # Adding configuration details below the plot
            plt.figtext(0.5, 0.01, self.format_config_details(), ha="center", fontsize=9, wrap=True)
            plt.subplots_adjust(bottom=0.2)
            # plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make room for text

            # save fig
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            fold_part = f"fold_{self.fold_idx}_" if self.fold_idx is not None else ""
            config_part = f"{self.rnn_type}_{self.seq_len}_{self.future_strategy}_{self.optimizer_type}_{self.lr}"
            plt.savefig(os.path.join(self.exp_dir, f"{timestamp}_{fold_part}_{config_part}_regression_loss.jpg"))

    def plot_forecast(self):
        # convert forecast results to dataframe
        forecast_df = pd.DataFrame(self.forecast_results)

        # convert datetime indices to pd.Datetime
        forecast_df['datetime indices'] = pd.to_datetime(forecast_df['datetime indices'])
        
        # sort dataframe by date
        forecast_df = forecast_df.sort_values('datetime indices')

        plt.figure()

        plt.plot(forecast_df['datetime indices'], forecast_df['predictions'], label='prediction')
        plt.plot(forecast_df['datetime indices'], forecast_df['truths'], label='truth')

        # Format the x-axis to display dates nicely
        plt.gcf().autofmt_xdate()  # Auto formats the x-axis labels to fit better
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d-%Y'))  # Optional: sets date format

        plt.legend()
        plt.title("Predictions vs Truths")
        plt.xlabel("Date")

        # Adding configuration details below the plot
        plt.figtext(0.5, 0.01, self.format_config_details(), ha="center", fontsize=9, wrap=True)
        plt.subplots_adjust(bottom=0.3)
        # plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make room for text

        # save fig
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        fold_part = f"fold_{self.fold_idx}_" if self.fold_idx is not None else ""
        config_part = f"{self.rnn_type}_{self.seq_len}_{self.future_strategy}_{self.optimizer_type}_{self.lr}"
        plt.savefig(os.path.join(self.exp_dir, f"{timestamp}_{fold_part}_{config_part}_forecast.jpg"))
        
    def plot_from_csv(self):
         pass
    
    def format_config_details(self):
        details = "Configuration Details:\n"
        if self.config:
            details += f"Model type: {self.rnn_type}, "
            details += f"Optimizer: {self.optimizer_type}, LR: {self.lr}, "
            details += f"Criterion: {self.criterion}, Seq Len: {self.seq_len}, "
            details += f"Batch Size: {self.batch_size}, "
            details += f"Future strategy: {self.future_strategy}"

        return details

