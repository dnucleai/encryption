import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self, in_size, out_size, hidden_size, activation=nn.Tanh):
        
        super(Net, self).__init__()
        self.activation = activation()
        self.hidden = nn.Linear(in_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.hidden2(x)
        return x