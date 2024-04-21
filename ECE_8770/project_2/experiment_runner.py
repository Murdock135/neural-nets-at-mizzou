import torch
import os
from src.utils import load_config, get_criterion, get_model, get_optimizer, ResultsPlotter
from src.dataset_utils import CustomDataset, TemporalCustomDataset, create_sequences
import src.model_trainer as trainers
import pandas as pd


# run program
if __name__ == "__main__":
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
            plotter.plot_regression_results()
    else:
        plotter = ResultsPlotter(exp_dir_for_results, trainer.results, config=config)
        plotter.plot_regression_results()

    # plot forecast
    
    # show plots
