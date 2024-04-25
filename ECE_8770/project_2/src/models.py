import torch
from torch.autograd import Variable
import torch.nn as nn

import torch
import torch.nn as nn

class FlexibleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, rnn_type='rnn', dropout=0.0, prediction_window=1, future_strategy='sequential'):
        super(FlexibleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_window = prediction_window
        self.future_strategy = future_strategy  # 'sequential', 'fixed_window', or 'many_to_one'
        self.rnn_type = rnn_type.lower()

        # Initialize the appropriate RNN layer
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Define the output layer based on the strategy
        if self.future_strategy == 'fixed_window':
            # Multiply output_size by prediction_window for fixed window predictions
            self.fc = nn.Linear(hidden_size, output_size * prediction_window)
        elif self.future_strategy == 'many_to_one':
            # Standard output size for many-to-one
            self.fc = nn.Linear(hidden_size, output_size)
        else:
            # Sequential predictions where each timestep predicts one output
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initial hidden state
        hn = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        cn = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) if self.rnn_type == 'lstm' else None

        # Forward pass
        if isinstance(self.rnn, nn.LSTM):
            out, (hn, cn) = self.rnn(x, (hn, cn))
        else:
            out, hn = self.rnn(x, hn)

        # Apply the fully connected layer
        out = self.fc(out)

        # Post-processing based on the strategy
        if self.future_strategy == 'fixed_window':
            # Reshape for fixed window predictions
            out = out.view(x.size(0), -1, self.prediction_window, out.size(-1) // self.prediction_window)
        elif self.future_strategy == 'many_to_one':
            # Select the last timestep for many-to-one prediction
            out = out[:, -1, :]  # Take the last timestep output only

        return out
