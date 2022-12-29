import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, ip_size, hidden_dim, op_size) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.lin_1 = nn.Linear(ip_size, hidden_dim)
        self.relu = nn.ReLU()
        self.lin_2 = nn.Linear(hidden_dim, op_size)

    def forward(self, x):
        out = self.relu(self.lin_1(x))
        out = self.lin_2(out)
        # softmax is not required as crossenropyloss function implements it
        return out

model = FeedForwardNetwork(ip_size=28*28, hidden_dim=5, op_size=3)
criterion = nn.CrossEntropyLoss() # applies softmax

'''
if its a biary classification problem, then we need to apply a sigmoid layer
at the end as the output linear layer has only 1 neuron

then use nn.BCEloss()
'''