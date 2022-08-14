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

class BRNN(nn.Module):
    def __init__(self,input_Size, hidden_size , num_layers, num_classes ):
        super(BRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_Size
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self,x):
        h = torch.zero_(self.num_layers*2, x.size(0), self.hidden_size).to(device) #x.size(0) is the number of examples we send to the batch size
        c = torch.zero_(self.num_layers*2, x.size(0), self.hidden_size).to(device) #x.size(0) is the number of examples we send to the batch size

        out, (hidden_state, cellState) = self.lstm(x, (h,c))

        return self.fc(out.resize(-1,seq_length))

