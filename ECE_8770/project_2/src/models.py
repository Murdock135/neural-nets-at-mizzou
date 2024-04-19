import torch
from torch.autograd import Variable
import torch.nn as nn

class BaseRNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, model_type='rnn') -> None:
        super(BaseRNNLSTM).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        
        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize H_t's  and C_t's (if lstm) to 0 vector
        # --------------------------------------------------
        # Size of H_t -> #stacked RNNs x (batch size x #hidden units)
        # or, mathematically
        # Size of H_t = Nx(nxh)
        # [N.B. in torch & np, to create K mxn matrices, we use A=torch.tensor((K,m,n))]

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # x.size(0) = batch size
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) if self.model_type.lower() == "lstm" else None

        # One time step
        # ------------
        # O_t: 
        # Size of o_t -> (#batch_size x #sequence length x (2 if bidirectional=True else, 1) #hidden_units)
        # Or, mathematically
        # Size of O_t = N x (L x c x h) 
        # [if c=1, (Lxcxh)=(Lxh)-> square | Intuitively, output should be equal to number of hidden units times sequence length
        #  if c=2, (Lxcxh)-> Cube | [TO BE WRITTEN LATER]]
        
        if isinstance(self.rnn, nn.LSTM):
            out, (hn, cn) = self.rnn(x, (h0, c0))
        else:
            out, hn = self.rnn(x, h0)

        return out


class ManyToOneRNNLSTM(BaseRNNLSTM):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, model_type='rnn') -> None:
        super().__init__(input_size, hidden_size, output_size, num_layers, model_type)

    def forward(self, x):
        out = super().forward(x)

        # use output of last time step
        # The slicing operation out[:, -1, :] selects the output corresponding to the last time step for each sequence in the batch. The dimensions of the output after this operation are [batch_size, hidden_size]
        out = self.fc(out[:, -1, :])

        return out
    
class ManyToManyRNNLSTM(BaseRNNLSTM):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, model_type='rnn') -> None:
        super().__init__(input_size, hidden_size, output_size, num_layers, model_type)

    def forward(self, x):
        out = super().forward(x)
        out = self.fc(out)

        return out