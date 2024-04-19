from box import box
import pandas as pd
import os
import numpy as np
import tomllib
import matplotlib.pyplot as plt
from .model_trainer import ClassifierTrainer, RegressorTrainer

def load_config(path_to_config) -> dict:
    with open(path_to_config, "rb") as f:
        config: dict = tomllib.load(f)
        return config



class ResultsPlotter:
    def __init__(self, exp_dir, results):
        '''Args:
            exp_dir(string)- directory to export visualizations to
            results(dictionary)- a dict of results where keys are metrics and values are lists
            '''
        self.exp_dir = exp_dir
        self.results = results

    def plot_results(self):

        # plot Training Loss + Validation Loss on same graph
        plt.figure()
        plt.plot(self.results['Training Loss'], label='Training Loss')
        plt.plot(self.results['Validation Loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.exp_dir, "loss.png"))

        # Plot accuracy and F1 score on the same graph
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['Validation Accuracy'], label='Accuracy', color='blue')
        plt.plot(self.results['f1 score'], label='F1 Score', color='green')
        plt.title('Accuracy and F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.ylim(0, 1)  # Set y-axis limits for F1 score
        plt.grid(True)
        plt.legend(loc='lower right')

        # Create a secondary y-axis for accuracy (scaled to percentage)
        ax = plt.gca().twinx()
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)  # Set y-axis limits for accuracy

        plt.savefig(os.path.join(self.exp_dir,'accuracy_and_f1.png'))


