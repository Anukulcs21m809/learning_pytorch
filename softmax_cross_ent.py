import torch
import numpy as np
import torch.nn as nn

# softmax with numpy

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('Softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

# cross entropy with numpy

def cross_entropy(actual, predicted):
    loss = -1 * np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

Y = np.array([1, 0, 0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)

print(l1)
print(l2)

# cross entropy with torch
print('cross entropy with torch')
'''
cross entropy applies softmax and negative log likelihood loss 
so no need to apply the softmax layer as the last layer

Y should have the class labels as a tensor not one hot encoded 

Y should have raw scores, that is softmax is not applied   
'''


loss = nn.CrossEntropyLoss()

Y = torch.tensor([2, 0, 1])

# n samples * m classes, let 1*3
# the Y_pred values are raw values , softmax has not been done
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [2.0, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, predictions_1 = torch.max(Y_pred_good, dim=1)
_, predictions_2 = torch.max(Y_pred_bad, dim=1)

print(predictions_1)
print(predictions_2)