#website for stuff:
#https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html



from __future__ import unicode_literals, print_function, division
from io import open
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join

from get_coords import returnCoordList
from make_data_dict import getDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size,batch_first=True)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden

    
    def reinit(self):
        '''Reinitialize weights'''
        def weights_init(l):
            if hasattr(l,'weight') and isinstance(l.weight, torch.Tensor):
                nn.init.xavier_uniform_(l.weight.data)
            if hasattr(l,'bias') and isinstance(l.bias, torch.Tensor):
                nn.init.uniform_(l.bias)
        self.apply(weights_init)

    
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DecoderRNN, self).__init__()
	self.hidden_size = hidden_size
	self.fc1 = nn.Linear(hidden_size,9)
        self.out = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
	output = F.relu(self.fc1(input))
        #output = F.softmax(output,dim=-1)

	print("output")
	print(output)

        return output

    def reinit(self):
        '''Reinitialize weights'''
        def weights_init(l):
            if hasattr(l,'weight') and isinstance(l.weight, torch.Tensor):
                nn.init.xavier_uniform_(l.weight.data)
            if hasattr(l,'bias') and isinstance(l.bias, torch.Tensor):
                nn.init.uniform_(l.bias)
        self.apply(weights_init)


