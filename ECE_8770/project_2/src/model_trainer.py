import torch
import numpy as np
import pandas as pd
import os
import csv
from sklearn import metrics
from sklearn.model_selection import KFold
from tqdm import tqdm
from datetime import datetime
from .dataset_utils import CustomDataLoader, temporal_collate_fn
from abc import ABC, abstractmethod


def reset_model_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# --------------------------------------------------------------------------------------------------------------------------------------
# The base class for trainers
class BaseTrainer(ABC):
    def __init__(self, model, device, dataset, criterion, optimizer, epochs, training_portion, batch_size, kfold=False, folds=None):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.training_portion = training_portion
        self.batch_size = batch_size
        self.kfold = kfold
        self.folds = folds

        if self.kfold:
            self.kf = KFold(n_splits= self.folds, shuffle=True)
            self.kf_results = []
        else:
            self.results = None


    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            epoch_loss += loss.item()

            # backpropagation
            loss.backward()

            # update weights
            self.optimizer.step()
        
        avg_loss = epoch_loss/len(train_loader)

        return avg_loss
    
    def create_data_loaders(self):
        return CustomDataLoader(self.dataset, self.training_portion, self.batch_size, shuffle=True)
    
    @abstractmethod
    def validate(self, val_loader):
        pass

    @abstractmethod
    def define_metrics(self):
        pass

    @abstractmethod
    def log_epoch_results(self, epoch, train_loss, val_results):
        pass

    @abstractmethod
    def log_fold_results(self, epoch, train_loss, val_results, fold_results):
        pass

    def create_results_dict(self):
        metrics = self.define_metrics()
        self.results = {metric: [] for metric in metrics}

    @abstractmethod
    def display_results(self, epoch, train_loss, val_results):
        pass  

    def train_and_validate(self):
        train_loader, val_loader = self.create_data_loaders()
        self.create_results_dict()

        for epoch in tqdm(range(1, self.epochs+1)):
            print("Epoch ", epoch)

            # train
            train_loss = self.train_one_epoch(train_loader)

            # validate
            val_results = self.validate(val_loader)

            # log results
            self.log_epoch_results(train_loss, val_results)

            # print out results every 10th epoch
            if (epoch) % 10 == 0:
                self.display_results(epoch, train_loss, val_results)

    def train_and_validate_kfold(self):
        # Sample elements randomly from a given list of ids, no replacement.
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(np.arange(len(self.dataset)))):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_subsampler)
            val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=val_subsampler)

            # create a dictionary to store fold results
            metrics: list = self.define_metrics()
            fold_results = {metric:[] for metric in metrics}
            
            for epoch in tqdm(range(self.epochs), desc=f"Fold {fold+1} Training"):
                train_loss = self.train_one_epoch(train_loader=train_loader)
                val_results = self.validate(val_loader=val_loader)

                # log epoch results
                self.log_fold_results(epoch, train_loss, val_results, fold_results)

                # display results if a decade has passed
                if (epoch+1) % 10 == 0:
                    self.display_results(epoch, train_loss, val_results)

            # add results of fold to master list
            self.kf_results.append(fold_results)
                     
    def train(self):
        if self.kfold:
            self.train_and_validate_kfold()
        else:
            self.train_and_validate()

    def save_model_results(self, exp_dir):
        # get current date and time
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        if self.kfold:
            # iterate over results of each fold and log results to a csv
            for fold_idx, fold_results in enumerate(self.kf_results):
                results_file_name = os.path.join(exp_dir, f"fold_{fold_idx}_results_{date_time_str}.csv")
                results_df = pd.DataFrame(fold_results)
                results_df.to_csv(results_file_name, index=False)
        else:
            results_file_name = os.path.join(exp_dir, f"results_{date_time_str}.csv")
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(results_file_name, index=False)

# -------------------------------------------------------------------------------------------------------------------------------------
# General purpose trainers
class ClassifierTrainer(BaseTrainer):
    
    def validate(self, val_loader):
        self.model.eval()

        total_loss = 0.0
        correct = 0.0
        total_labels = 0
        y_pred, y_true = [], []

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_labels += labels.size(0)

                predicted = predicted.to('cpu')
                labels = labels.to('cpu')

                y_pred.extend(predicted.numpy())
                y_true.extend(labels.numpy())
                correct += (predicted == labels).sum().item()

        # compute metrics
        avg_loss = total_loss/len(val_loader)
        accuracy = 100 * correct/total_labels
        f1 = metrics.f1_score(y_true, y_pred, average='weighted')
        precision = metrics.precision_score(y_true, y_pred, average='weighted')
        recall = metrics.recall_score(y_true, y_pred, average='weighted')

        return avg_loss, accuracy, f1, precision, recall
    
    def define_metrics(self):
        return ['training loss', 'validation loss', 'accuracy', 'f1 score', 'precision', 'recall']
    
    def log_epoch_results(self, train_loss, val_results):
        val_loss, val_accuracy, f1, precision, recall = val_results

        self.results['training loss'].append(train_loss)
        self.results['validation loss'].append(val_loss)
        self.results['accuracy'].append(val_accuracy)
        self.results['f1 score'].append(f1)
        self.results['precision'].append(precision)
        self.results['recall'].append(recall)

    def log_fold_results(self, train_loss, val_results, fold_results):
        val_loss, val_accuracy, f1, precision, recall = val_results

        fold_results['training loss'].append(train_loss)
        fold_results['validation loss'].append(val_loss)
        fold_results['accuracy'].append(val_accuracy)
        fold_results['f1 score'].append(f1)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)

    def display_results(self, epoch, train_loss, val_results, epoch_to_display = 10):
        val_loss, val_accuracy, f1, precision, recall = val_results

        print("-" * 10)
        print(f"Epoch {epoch+1}:\n"
        f"Training loss = {train_loss}\n"
        f"Validation loss = {val_loss}\n"
        f"Validation accuracy = {val_accuracy}\n"
        f"f1 score = {f1}\n"
        f"precision = {precision}\n"
        f"recall = {recall}\n")


class RegressorTrainer(BaseTrainer):
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                inputs, targets = data
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss/len(val_loader)

        return avg_loss
    
    def define_metrics(self):
        return ['training loss', 'validation loss']
    
    def log_epoch_results(self, epoch, train_loss, val_results):
        val_loss = val_results

        self.results['training loss'].append(train_loss)
        self.results['validation loss'].append(val_loss)

    def log_fold_results(self, epoch, train_loss, val_results, fold_results):
        val_loss = val_results

        fold_results['training loss'].append(train_loss)
        fold_results['validation loss'].append(val_loss)

    def display_results(self, epoch, train_loss, val_results):
        val_loss = val_results

        print("-" * 10)
        print(f"Epoch {epoch+1}:\n"
        f"Training loss = {train_loss}\n"
        f"Validation loss = {val_loss}\n")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Trainers for sequential data

class SequentialClassifierTrainer(ClassifierTrainer):
    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            time_idx, inputs, labels = data  # Unpack the tuple with datetime_idx

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            epoch_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss
      
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_labels = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                time_idx, inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Depending on the output shape, adjust the accuracy calculation
                if self.model.future_strategy in ['many_to_one', 'sequential']:
                    _, predicted = torch.max(outputs, -1)
                    correct += (predicted == labels).sum().item()
                    total_labels += labels.numel()

        accuracy = 100 * correct / total_labels
        return total_loss / len(val_loader), accuracy
    
    def create_data_loaders(self):
        return CustomDataLoader(self.dataset, self.training_portion, self.batch_size, shuffle=True, collate_fn=temporal_collate_fn)

    def define_metrics(self):
        return ['training loss', 'validation loss', 'accuracy']

    def log_epoch_results(self, train_loss, val_results):
        val_loss, accuracy = val_results
        self.results['training loss'].append(train_loss)
        self.results['validation loss'].append(val_loss)
        self.results['accuracy'].append(accuracy)

    def display_results(self, epoch, train_loss, val_results):
        val_loss, accuracy = val_results
        print(f"Epoch {epoch+1}: Training loss = {train_loss}, Validation loss = {val_loss}, Accuracy = {accuracy}%")
                
class SequentialRegressorTrainer(RegressorTrainer):
    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            time_idx, inputs, targets = data  # Adjust to unpack the datetime_idx

            # move inputs and outputs to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # clear gradients
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # accumulate loss
            epoch_loss += loss.item()

            # backward pass (optimize)
            loss.backward()
            self.optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()

        total_loss = 0.0
        predictions = []
        truths = []
        datetime_indices = []

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                time_idx, inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                # Adjust the loss computation based on the output strategy
                if self.model.future_strategy == 'many_to_one':
                    # For many-to-one, targets are typically a single value per sequence
                    loss = self.criterion(outputs, targets)
                elif self.model.future_strategy in ['sequential', 'fixed_window']:
                    # For sequential or fixed_window, reshape outputs to match targets if necessary
                    # This assumes that targets are appropriately shaped for these strategies
                    outputs = outputs.view(-1, outputs.shape[-1])  # Flatten output for batch processing
                    targets = targets.view(-1, targets.shape[-1])  # Flatten targets to match output
                    time_idx = time_idx.flatten() # Flatten time indices to match output
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()

                # collect predictions, truths and datetime_indices for this batch
                predictions.extend(outputs.cpu().numpy())
                truths.extend(targets.cpu().numpy())

                # reshape datetime_indices to match the shape of predictions and truths
                datetime_indices.extend(time_idx)

        avg_loss = total_loss / len(val_loader)
        return avg_loss, predictions, truths, datetime_indices
    
    def create_data_loaders(self):
        return CustomDataLoader(self.dataset, self.training_portion, self.batch_size, shuffle=True, collate_fn=temporal_collate_fn)

    def define_metrics(self):
        return ['training loss', 'validation loss', 'predictions', 'truths', 'datetime indices']

    def log_epoch_results(self, train_loss, val_results):
        val_loss, predictions, truths, datetime_indices = val_results
        self.results['training loss'].append(train_loss)
        self.results['validation loss'].append(val_loss)
        self.results['predictions'].extend(predictions)
        self.results['truths'].extend(truths)
        self.results['datetime indices'].extend(datetime_indices)

    def display_results(self, epoch, train_loss, val_loss):
        print(f"Epoch {epoch+1}: Training loss = {train_loss}, Validation loss = {val_loss}")

    def save_model_results(self, exp_dir):
    # get current date and time
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        if self.kfold:
            # iterate over results of each fold and log results to a csv
            for fold_idx, fold_results in enumerate(self.kf_results):
                # store forecast values in separate dataframe
                forecast = dict()

                for key, values in fold_results.items():
                    if key in ["predictions", "truths", "datetime indices"]:
                        forecast[key] = values

                # delete predictions, truths and datetime indices from fold_results
                del fold_results['predictions']
                del fold_results['truths']
                del fold_results['datetime indices']

                results_file_name = os.path.join(exp_dir, f"fold_{fold_idx}_results_{date_time_str}.csv")
                forecast_file_name = os.path.join(exp_dir, f"fold_{fold_idx}_forecast_{date_time_str}.csv")

                results_df = pd.DataFrame(fold_results)
                forecast_df = pd.DataFrame(forecast)

                results_df.to_csv(results_file_name, index=False)
                forecast_df.to_csv(forecast_file_name, index=False)
        else:
            # store forecast values in separate dataframe
            forecast = dict()

            for key, values in self.results.items():
                if key in ["predictions", "truths", "datetime indices"]:
                    forecast[key] = values

            del self.results['predictions']
            del self.results['truths']
            del self.results['datetime indices']

            results_file_name = os.path.join(exp_dir, f"results_{date_time_str}.csv")
            forecast_file_name = os.path.join(exp_dir, f"forecast_{date_time_str}.csv")

            results_df = pd.DataFrame(self.results)
            forecast_df = pd.DataFrame(forecast)

            results_df.to_csv(results_file_name, index=False)
            forecast_df.to_csv(forecast_file_name, index=False)

