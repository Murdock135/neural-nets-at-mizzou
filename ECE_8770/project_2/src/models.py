import torch
from torch.autograd import Variable
import torch.nn as nn

class myRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim) -> None:
        super(myRNN).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize H_t's to 0
        # ---------------------
        # Size of H_t -> #stacked RNNs x (batch size x #hidden units)
        # or, mathematically
        # Size of H_t = Nx(nxh)
        # [N.B. in torch & np, to create K mxn matrices, we use A=torch.tensor((K,m,n))]
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)) # x.size(0) = batch size

        # One time step
        # ------------
        # O_t: 
        # Size of o_t -> (#batch_size x #sequence length x (2 if bidirectional=True else, 1) #hidden_units)
        # Or, mathematically
        # Size of O_t = N x (L x c x h) 
        # [if c=1, (Lxcxh)=(Lxh)-> square | Intuitively, output should be equal to number of hidden units times sequence length
        #  if c=2, (Lxcxh)-> Cube | [TO BE WRITTEN LATER]]
        
        out, hn = self.rnn(x, h0)

        # The slicing operation out[:, -1, :] selects the output corresponding to the last time step for each sequence in the batch. The dimensions of the output after this operation are [batch_size, hidden_dim]
        out = self.fc(out[:, -1, :])

        return out
