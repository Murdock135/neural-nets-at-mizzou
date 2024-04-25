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
    def __init__(self, model, model_type, device, dataset, criterion, optimizer, optimizer_name, learning_rate, epochs, training_portion, batch_size, kfold=False, folds=None):
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
        self.model_type = model_type
        self.optimizer_type = optimizer_name
        self.learning_rate = learning_rate

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
            self.log_epoch_results(epoch, train_loss, val_results)

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
        config_part = f"{self.rnn_type}_{self.seq_len}_{self.future_strategy}_{self.optimizer_type}_{self.lr}"

        if self.kfold:
            # iterate over results of each fold and log results to a csv
            for fold_idx, fold_results in enumerate(self.kf_results):
                results_file_name = os.path.join(exp_dir, f"fold_{fold_idx}_results_{date_time_str}.csv")
                results_df = pd.DataFrame(fold_results)
                results_df.to_csv(results_file_name, index=False)
        else:
            results_file_name = os.path.join(exp_dir, f"results_{date_time_str}_{config_part}.csv")
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
    
    def log_epoch_results(self, epoch, train_loss, val_results):
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
# Trainers for sequential data (N.B. READ NOTE AT THE END)

class SequentialClassifierTrainer(ClassifierTrainer):
    def __init__(self, model, device, dataset, criterion, optimizer, epochs, training_portion, batch_size, kfold=False, folds=None):
        super().__init__(model, device, dataset, criterion, optimizer, epochs, training_portion, batch_size, kfold, folds)
        self.forecast = {'predictions':[], 'truths':[], 'datetime indices':[]}

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

        predictions = []
        truths = []
        datetime_indices = []

        correct = 0
        total_labels = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                time_idx, inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                if outputs.dim() > 2 and outputs.shape[-1] == labels.shape[-1]:
                    # flatten if (sequential prediction)
                    outputs_flat = outputs.view(-1, outputs.shape[-1])
                    labels_flat = labels.view(-1, outputs.shape[-1])
                else:
                    outputs_flat = outputs
                    labels_flat = labels
                
                loss = self.criterion(outputs_flat, labels_flat)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs_flat, 1)
                correct += (predicted == labels_flat).sum().item()
                total_labels += labels_flat.size(0)

                # Store predictions and truths for detailed analysis
                predictions.extend(predicted.cpu().numpy())
                truths.extend(labels_flat.cpu().numpy())
                datetime_indices.extend(time_idx.flatten().cpu().numpy())

        accuracy = 100 * correct / total_labels
        avg_loss = total_loss / len(val_loader)

        return avg_loss, accuracy, predictions, truths, datetime_indices
    
    def create_data_loaders(self):
        return CustomDataLoader(self.dataset, self.training_portion, self.batch_size, shuffle=True, collate_fn=temporal_collate_fn)

    def define_metrics(self):
        return ['training loss', 'validation loss', 'accuracy', 'predictions', 'truths', 'datetime indices']

    def log_epoch_results(self, epoch, train_loss, val_results):
        val_loss, accuracy, predictions, truths, datetime_indices = val_results

        # store preformance metrics results
        self.results['training loss'].append(train_loss)
        self.results['validation loss'].append(val_loss)
        self.results['accuracy'].append(accuracy)

        if epoch == self.epochs:
            # store predictions and truths if on last epoch
            self.forecast['predictions'].extend(predictions)
            self.forecast['truths'].extend(truths)
            self.forecast['datetime indices'].extend(datetime_indices)

    def display_results(self, epoch, train_loss, val_results):
        val_loss, accuracy = val_results
        print(f"Epoch {epoch+1}: Training loss = {train_loss}, Validation loss = {val_loss}, Accuracy = {accuracy}%")
                
class SequentialRegressorTrainer(RegressorTrainer):
    def __init__(self, model, model_type, device, dataset, criterion, optimizer, optimizer_name, learning_rate, epochs, training_portion, batch_size, sequence_length, future_strategy, kfold=False, folds=None):
        super().__init__(model, model_type, device, dataset, criterion, optimizer, optimizer_name, learning_rate, epochs, training_portion, batch_size, kfold, folds)
        self.forecast = {'predictions':[], 'truths':[], 'datetime indices':[]}
        self.seq_len = sequence_length
        self.future_strategy = future_strategy

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
                    time_idx = time_idx.flatten()
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

    def log_epoch_results(self, epoch, train_loss, val_results):
        val_loss, predictions, truths, datetime_indices = val_results

        # store preformance metrics results
        self.results['training loss'].append(train_loss)
        self.results['validation loss'].append(val_loss)

        if epoch == self.epochs:
            # store predictions and truths if on last epoch
            self.forecast['predictions'].extend(predictions)
            self.forecast['truths'].extend(truths)
            self.forecast['datetime indices'].extend(datetime_indices)

    def display_results(self, epoch, train_loss, val_results):
        val_loss, predictions, truths, datetime_indices = val_results
        print(f"Epoch {epoch+1}: Training loss = {train_loss}, Validation loss = {val_loss}")

    def save_model_results(self, exp_dir):
    # get current date and time
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # if self.kfold:
        #     # iterate over results of each fold and log results to a csv
        #     for fold_idx, fold_results in enumerate(self.kf_results):

        #         results_file_name = os.path.join(exp_dir, f"fold_{fold_idx}_results_{date_time_str}.csv")
        #         forecast_file_name = os.path.join(exp_dir, f"fold_{fold_idx}_forecast_{date_time_str}.csv")

        #         results_df = pd.DataFrame(fold_results)
        #         forecast_df = pd.DataFrame(forecast)

        #         results_df.to_csv(results_file_name, index=False)
        #         forecast_df.to_csv(forecast_file_name, index=False)
        # else:
            # store forecast values in separate dataframe
        del self.results['predictions']
        del self.results['truths']
        del self.results['datetime indices']

        config_part = f"{self.model_type}_{self.seq_len}_{self.future_strategy}_{self.optimizer_type}_{self.learning_rate}"

        results_file_name = os.path.join(exp_dir, f"results_{date_time_str}_{config_part}.csv")
        forecast_file_name = os.path.join(exp_dir, f"forecast_{date_time_str}_{config_part}.csv")

        results_df = pd.DataFrame(self.results)
        forecast_df = pd.DataFrame(self.forecast)

        results_df.to_csv(results_file_name, index=False)
        forecast_df.to_csv(forecast_file_name, index=False)

#------------------------------------------------------------------------
# NOTE FOR HANDLING SEQUENTIAL DATA 

# Sequential (Many-to-Many):
# For a "sequential" or "many-to-many" approach where the input and output sequences are of equal length:

# Inputs: Shape of (n, sequence_len, num_features) where n is the batch size, sequence_len is the number of timesteps per sample, and num_features is the number of features per timestep.
# Labels/Outputs: Shape of (n, sequence_len, num_features) if predicting the same structure as the input. Often, the output might just predict a single feature, in which case it could be (n, sequence_len, 1).
# Time Indices: Shape of (n, sequence_len) representing the time indices for each timestep in each sequence in the batch.

# Many-to-One:
# In a "many-to-one" setting:
# Inputs: Shape of (n, s, k) aligns perfectly with your description where s is the sequence length and k is the number of features per timestep.
# Outputs: Typically, the output for "many-to-one" would be (n, 1) or (n,) if predicting a single feature per sequence. If predicting multiple features (still one output per sequence), then (n, 1, k) or (n, k) might apply.
# Time Indices for Inputs: Will be (n, s) as each input sequence will have its own time index.
# Time Indices for Outputs: Typically, the time index will be (n,) since there's one output per sequence at a specific time point.

# Fixed Window:
# In "fixed window" forecasting:
# Inputs: Shape (n, s, k) is correct.
# Outputs: Shape (n, w, k) where w is the number of timesteps you want to predict into the future, and k is the number of features predicted per timestep.
# Time Indices for Inputs: (n, s) since each input timestep has an associated time index.
# Time Indices for Outputs: Ideally, (n, w) as each output timestep in the window will have its own time index.
# Additional Points
# Time Indices Alignment: It's crucial that the time_idx aligns correctly with both inputs and outputs. For training, particularly with neural networks, it's common to keep track of these indices separately and not input them into the neural network unless you're using them explicitly for conditional computations within the network architecture.
# Consistency Across Batch: Ensure that all sequences within a batch are of uniform length unless using techniques like padding or masking to handle variable lengths, which is common in sequence models like RNNs, LSTMs, etc.
# Shape Handling in PyTorch: When specifying shapes, remember that PyTorch typically does not need explicit single dimensions (e.g., (n, 1, k) could often just be (n, k)), unless the dimensionality is essential for operations like broadcasting.