# data loaader is important becasue if we have a large dataset, we dont want to load everything into the memory at once
# it loads the data in batches

'''
epoch = 1 forward and backward pass for all the sample of data

batch_size = number of training samples in one forward and backward pass

num_iterations = no of passes, each pass using [batch_size] number of samples

eg: 100 samples, batch_size 20, --> 100/20 = 5 iterations for each epoch

'''

import torch
import torchvision
from torch.utils.data import Dataset , DataLoader
import numpy as np
import math

class WineDataSet(Dataset):

    def __init__(self) -> None:
        super(WineDataSet, self).__init__()
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataSet()
# first_data = dataset[0]

# features, labels = first_data

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

# dataiter = iter(dataloader)
# data = next(dataiter)
# features, labels = data
# print(features, labels)

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)

print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        print(f'epoch : {epoch + 1}/ {num_epochs}, step : {i+1}/ {n_iterations}, inputs : {inputs.shape}')