import torch
import os
from src.utils import load_config, ResultsPlotter
from src.dataset_utils import CustomDataset, create_sequences, separate_features_and_target
from src.model_trainer import ClassifierTrainer, RegressorTrainer, reset_model_weights
from src.models import myRNN
import torch.optim as optim
import torch.nn as nn
import pandas as pd

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

# run program
if __name__ == "__main__":
    # load configuration
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.toml")
    config = load_config(config_path)

    # set the path to save results later on
    results_path = os.path.join(os.path.dirname(__file__), "results")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data_path = "c:/Users/Zayan/Documents/code/personal_repos/neural_nets/ECE_8770/project_2/data/online+retail/scaled_and_encoded_dataset.csv"
    data = pd.read_csv(data_path)

    # create sequences
    sequence_len = config['model']['sequence_length']
    prediction_window = config['model']['prediction_window']
    X, y = create_sequences(data, sequence_len, prediction_window)

    # create dataset
    dataset = CustomDataset(X, y, 'mse')

    # training parameters
    model = myRNN(input_dim=sequence_len, hidden_dim=config['model']['hidden_size'], layer_dim=config['model']['num_layers']).to(device)
    optimizer = get_optimizer(model, config)
    criterion = get_criterion(config)
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    is_kfold = config['training']['k_fold']['status']

    if is_kfold:
        n_folds = config['training']['k_fold']['n_folds']
    else:
        n_folds = 0

    trainer = RegressorTrainer(model=model, 
                                device=device, 
                                dataset=dataset, 
                                criterion=criterion, 
                                optimizer=optimizer,
                                epochs=epochs,
                                batch_size=batch_size,
                                kfold=is_kfold,
                                folds=n_folds)
    
    # save results
    trainer.save_model_results(results_path)

    # plot results
    plotter = ResultsPlotter(results_path)
    plotter.plot_results()