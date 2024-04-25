import torch
import os
from src.utils import load_config, get_criterion, get_model, get_optimizer, ResultsPlotter
from src.dataset_utils import CustomDataset, TemporalCustomDataset, create_sequences
import src.model_trainer as trainers
import pandas as pd
import torch.optim as optim
from itertools import product


# experiment runner
def run():
    # load configuration
    config_path = "c:/Users/Zayan/Documents/code/personal_repos/neural_nets/ECE_8770/project_2/configs/config.toml"
    config = load_config(config_path)

    # set the path to save results later on
    exp_dir_for_results = "c:/Users/Zayan/Documents/code/personal_repos/neural_nets/ECE_8770/project_2/results"

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data_name = 'data_1' # in the TOML file, there are several datasets.
    data_path = config[data_name]['data_path']
    data = pd.read_csv(data_path, index_col=0)

    # get column to make sequences out of
    column = config[data_name]['column'] if 'column' in config[data_name] else None

    # create sequences
    sequence_len = config['model']['sequence_length']

    future_strategy = config['model']['future_strategy']
    if future_strategy == "fixed_window":
        prediction_window = config['model']['prediction_window']
        X, y, t = create_sequences(data, sequence_len, future_strategy=future_strategy, prediction_window=prediction_window, column=column)
    else:
        X, y, t = create_sequences(data, sequence_len, future_strategy=future_strategy, column=column)

    input_size = X.shape[-1]
    output_size = y.shape[-1]

    # create dataset
    # dataset = CustomDataset(X, y, 'mse')
    dataset = TemporalCustomDataset(X, y, t, 'mse')

    # training parameters
    model = get_model(config, no_features=input_size, output_size=output_size).to(device)
    optimizer = get_optimizer(model, config)
    criterion = get_criterion(config)
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    is_kfold = config['training']['k_fold']['status']
    training_portion = config['training']['training_portion']

    if is_kfold:
        n_folds = config['training']['k_fold']['n_folds']
    else:
        n_folds = 0

    # print experiment config
    print(f"dataset: {data_path}")
    print(f"optimizer: {optimizer}")
    print(f"epochs: {epochs}")
    print(f"batch size: {batch_size}")
    print(f"is_kfold: {is_kfold}")
    print(f"training portion: {training_portion}")

    trainer = trainers.SequentialRegressorTrainer(model=model, 
                                device=device, 
                                dataset=dataset, 
                                criterion=criterion, 
                                optimizer=optimizer,
                                epochs=epochs,
                                training_portion=training_portion,
                                batch_size=batch_size,
                                kfold=is_kfold,
                                folds=n_folds)
    
    trainer.train()
    
    # save results
    trainer.save_model_results(exp_dir_for_results)

    # plot results
    if is_kfold:
        results_per_fold = trainer.kf_results

        for fold_idx, fold_results in enumerate(results_per_fold):
            plotter = ResultsPlotter(exp_dir_for_results, fold_results, fold_idx, config=config)
            plotter.plot_regression_results() # plot performance
            plotter.plot_forecast() # plot forecast
    else:
        plotter = ResultsPlotter(exp_dir_for_results, trainer.results, config=config, forecast_results=trainer.forecast)
        plotter.plot_regression_results() # plot performance
        plotter.plot_forecast() # plot forecast

def alt_run(rnn_type, sequence_len, user_optimizer, learning_rate, future_strategy):
    '''This runner will not use sequence_length, rnn_type, optimizer and 
    future_strategy from config.toml'''

    # load configuration
    config_path = "c:/Users/Zayan/Documents/code/personal_repos/neural_nets/ECE_8770/project_2/configs/config.toml"
    config = load_config(config_path)

    # set the path to save results later on
    exp_dir_for_results = "c:/Users/Zayan/Documents/code/personal_repos/neural_nets/ECE_8770/project_2/results"

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data_name = 'data_1' # in the TOML file, there are several datasets.
    data_path = config[data_name]['data_path']
    data = pd.read_csv(data_path, index_col=0)

    # get column to make sequences out of
    column = config[data_name]['column'] if 'column' in config[data_name] else None

    # create sequences
 
    if future_strategy == "fixed_window":
        prediction_window = config['model']['prediction_window']
        X, y, t = create_sequences(data, sequence_len, future_strategy=future_strategy, prediction_window=prediction_window, column=column)
    else:
        X, y, t = create_sequences(data, sequence_len, future_strategy=future_strategy, column=column)

    input_size = X.shape[-1]
    output_size = y.shape[-1]

    # create dataset
    # dataset = CustomDataset(X, y, 'mse')
    dataset = TemporalCustomDataset(X, y, t, 'mse')

    # training parameters
    model = get_model(config, no_features=input_size, output_size=output_size, rnn_type=rnn_type, future_strategy=future_strategy).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if user_optimizer.lower()=="adam" else optim.SGD(model.parameters(), lr=learning_rate)
    criterion = get_criterion(config)
    epochs = 100 if user_optimizer.lower() == "sgd" else 10
    batch_size = config['training']['batch_size']
    is_kfold = config['training']['k_fold']['status']
    training_portion = config['training']['training_portion']

    if is_kfold:
        n_folds = config['training']['k_fold']['n_folds']
    else:
        n_folds = 0

    # print experiment config
    print(f"dataset: {data_path}")
    print(f"optimizer: {user_optimizer}")
    print(f"epochs: {epochs}")
    print(f"batch size: {batch_size}")
    print(f"is_kfold: {is_kfold}")
    print(f"training portion: {training_portion}")

    trainer = trainers.SequentialRegressorTrainer(model=model,
                                                  model_type = rnn_type,
                                                  sequence_length=sequence_len, 
                                device=device, 
                                dataset=dataset, 
                                criterion=criterion, 
                                optimizer=optimizer,
                                optimizer_name=user_optimizer,
                                epochs=epochs,
                                training_portion=training_portion,
                                batch_size=batch_size,
                                kfold=is_kfold,
                                folds=n_folds,
                                learning_rate=learning_rate,
                                future_strategy=future_strategy)
    
    trainer.train()
    
    # save results
    trainer.save_model_results(exp_dir_for_results)

    # plot results
    if is_kfold:
        results_per_fold = trainer.kf_results

        for fold_idx, fold_results in enumerate(results_per_fold):
            plotter = ResultsPlotter(exp_dir_for_results, fold_results, fold_idx, config=config)
            plotter.plot_regression_results() # plot performance
            plotter.plot_forecast() # plot forecast
    else:
        plotter = ResultsPlotter(exp_dir_for_results, trainer.results, config=config, forecast_results=trainer.forecast, model_type=rnn_type, sequence_len=sequence_len, future_strategy=future_strategy, user_optimizer=user_optimizer, learning_rate=lr)
        plotter.plot_regression_results() # plot performance
        plotter.plot_forecast() # plot forecast



# run experiment
if __name__=="__main__":
    # run()
    rnn_type = ['lstm']
    seq_len = [3, 7, 50, 100]
    optimizer = ["adam", "sgd"]
    lr = [0.001]
    future_strategy = ['sequential', 'many_to_one']

    combos = list(product(rnn_type, seq_len, optimizer, lr, future_strategy))

    for combo in combos:
        print(combo)
        combo_rnn = combo[0]
        combo_seq_len = combo[1]
        combo_optimizer = combo[2]
        combo_lr = combo[3]
        combo_future_strategy = combo[4]

        alt_run(rnn_type=combo_rnn, 
                sequence_len=combo_seq_len,
                user_optimizer=combo_optimizer,
                learning_rate=combo_lr,
                future_strategy=combo_future_strategy)
