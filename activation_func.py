import torch
import torch.nn as nn
import numpy as np

''' 
we can either directly use the activation functions in the
forward method of a model class or we can save it first 
in the init method and then use the functions
'''
# eg of using it by saving

class NeuralNet(nn.Module):
    def __init__(self, ip_size, hidden_size) -> None:
        super(NeuralNet, self).__init__()
        self.lin_1 = nn.Linear(ip_size, hidden_size)
        self.relu = nn.ReLU()
        # other avaliable activations : nn.Sigmoid , nn.Softmax(), nn.TanH(), nn.LeakyReLU()
        self.lin_2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # torch.relu(), torch.sigmoid() for using it directly
        out = self.relu(self.lin_1(x))
        out = self.sigmoid(self.lin_2(out))
        return out