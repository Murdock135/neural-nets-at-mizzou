import torch
import numpy as np
import pandas as pd
import os
import csv
from sklearn import metrics
from sklearn.model_selection import KFold
from tqdm import tqdm
from datetime import datetime
from .dataset_utils import CustomDataLoader
from abc import ABC, abstractmethod


def reset_model_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
class Trainer(ABC):
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

        for epoch in tqdm(range(self.epochs)):
            train_loss = self.train_one_epoch(train_loader)
            val_results = self.validate(val_loader)
            self.log_epoch_results(epoch, train_loss, val_results)

            if (epoch+1) % 10 == 0:
                self.display_results(epoch, train_loss, val_results)

    def train_and_validate_kfold(self):
        # Sample elements randomly from a given list of ids, no replacement.
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(np.arange(len(self.dataset)))):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_loader = torch.utils.data.Dataloader(self.dataset, batch_size=self.batch_size, sampler=train_subsampler)
            val_loader = torch.utils.data.Dataloader(self.dataset, batch_size=self.batch_size, sampler=val_subsampler)

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
                filename = os.path.join(exp_dir, f"fold_{fold_idx}_results_{date_time_str}.csv")
                results_df = pd.DataFrame(fold_results)
                results_df.to_csv(filename, index=False)
        else:
            filename = os.path.join(exp_dir, f"results_{date_time_str}.csv")
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(filename, index=False)

class ClassifierTrainer(Trainer):
    
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


class RegressorTrainer(Trainer):
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

    def display_results(self, epoch, train_loss, val_results):
        val_loss = val_results

        print("-" * 10)
        print(f"Epoch {epoch+1}:\n"
        f"Training loss = {train_loss}\n"
        f"Validation loss = {val_loss}\n")


#     def train_and_validate(self):
#         train_loader, val_loader = CustomDataLoader(self.dataset, self.training_portion, self.batch_size, shuffle=True)

#         for epoch in tqdm(range(self.epochs)):
#             train_loss = self.train_one_epoch(train_loader=train_loader)
#             val_loss, val_accuracy, f1 = self.validate(val_loader=val_loader)

#             self.results['training loss'].append(train_loss)
#             self.results['validation loss'].append(val_loss)
#             self.results['accuracy'].append(val_accuracy)
#             self.results['f1 score'].append(f1)

#             if (epoch+1) % 10 == 0:
#                     print("-" * 10)
#                     print(f"Epoch {epoch+1}:\n"
#                         f"Training loss = {train_loss}\n"
#                         f"Validation loss = {val_loss}\n"
#                         f"Validation accuracy = {val_accuracy}\n"
#                         f"f1 score = {f1}")

#     def train_and_validate_kfold(self):
#         # Sample elements randomly from a given list of ids, no replacement.
#         for fold, (train_idx, val_idx) in enumerate(self.kf.split(np.arange(len(self.dataset)))):
#             train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
#             val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

#             train_loader = torch.utils.data.Dataloader(self.dataset, batch_size=self.batch_size, sampler=train_subsampler)
#             val_loader = torch.utils.data.Dataloader(self.dataset, batch_size=self.batch_size, sampler=val_subsampler)

#             fold_results = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'f1': []}
#             for epoch in tqdm(range(self.epochs), desc=f"Fold {fold+1} Training"):
#                 train_loss = self.train_one_epoch(train_loader=train_loader)
#                 val_loss, val_accuracy, f1 = self.validate(val_loader=val_loader)

                
#                 fold_results['train_loss'].append(train_loss)
#                 fold_results['val_loss'].append(val_loss)
#                 fold_results['val_accuracy'].append(val_accuracy)
#                 fold_results['f1'].append(f1)

#                 if (epoch+1) % 10 == 0:
#                      print("-" * 10)
#                      print(f"Epoch {epoch+1}:\n"
#                         f"Training loss = {train_loss}\n"
#                         f"Validation loss = {val_loss}\n"
#                         f"Validation accuracy = {val_accuracy}\n"
#                         f"f1 score = {f1}")
            
#             self.kf_results.append(fold_results)
                     
#     def train(self):
#         if self.kfold:
#             self.train_and_validate_kfold()
#         else:
#             self.train_and_validate()

#     def save_model_results(self, exp_dir):
#         # date and time
#         now = datetime.now()
#         date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

#         if KFold:
#             # iterate over results of each fold and log results to a csv
#             for fold_idx, fold_results in enumerate(self.kf_results):
#                 filename = os.path.join(exp_dir, f"fold_{fold_idx}_results_{date_time_str}.csv")
#                 results_df = pd.DataFrame(fold_results)
#                 results_df.to_csv(filename, index=False)
#         else:
#             filename = os.path.join(exp_dir, f"results_{date_time_str}.csv")
#             results_df = pd.DataFrame(self.results)
#             results_df.to_csv(filename, index=False)


# class RegressorTrainer:
#     pass
                
