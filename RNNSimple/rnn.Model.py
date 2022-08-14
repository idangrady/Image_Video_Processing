import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpy')

print(torch.cuda.is_available())

input_size = 28
seq_length = 28
num_layes = 2
h_size = 256
num_classes = 10
lr =.001
batch_size = 64
num_epoch = 2

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        self.input_size= input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.rnn =nn.RNN(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers= self.num_layers,
                         bias =False,
                         batch_first=True) # output : (sequence length , hiddensize)
        self.fc = nn.Linear(self.hiddensize*seq_length, self.num_classes) #TODO: why here multiple by seqLength

    def forward(self, x):
        h_0 = torch.zeros(self.input_size, x.size(0), self.hidden_size).to(device) #TODO: check why no x.shape[1]
        seq_length, hidden_size = self.rnn(x,h_0)
        seq_length = seq_length.reshape(-1, seq_length)
        out = self.fc(seq_length)
        
        return out
        

