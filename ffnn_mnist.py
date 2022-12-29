import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


'''
MNIST
Dataloader, Transformation
Multilayer neural net, activation function
Loss and optimizer
Training loop
model evaluation 
GPU support
'''

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

#MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', transform=transforms.ToTensor(), train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', transform=transforms.ToTensor(), train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#loading and plotting some samples of data
examples = iter(train_loader)
samples, labels = next(examples)
# print(samples.shape)
# print(labels.shape)

for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()

#create the model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet, self).__init__()
        self.lin_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin_2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.relu(self.lin_1(x))
        out = self.lin_2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimzer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
#  returns a tensor of [100, 1 , 28, 28] but we need as input shape (batch_size, input_dim)
    for i,  (images, labels) in enumerate(train_loader): 
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        #forward
        # output is a 100 * 10 tensor
        output = model(images)
        loss = criterion(output, labels)
        
        #backward
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch : {epoch + 1}/ {num_epochs} , step : {i+1} / {n_total_steps}, loss : {loss.item():.4f}')

#test
with torch.no_grad(): 
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        #values, indexs 
        _, predictions = torch.max(outputs, dim=1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    
    acc = 100 * n_correct / n_samples
    print(f'accuracy : {acc}')