'''
On images
---------

Centercrop, Grayscale, Pad, RandoAffine, RandomCrop, RandomHorizontalFlip, RandomRotation, 
Resize, Scale

On Tensors
----------

Linearransformation, Normalize, RandomErasing

Conversion
----------

ToPILImage :  from tensor to ndarray
ToTensor : from numpy.ndarray to PILImage

Generic
-----------

Use lambda

Custom 
-----------
Write your own class

Compose multiple transforms
-----------

composed = transforms.Compose([Rescale(256), RandomCrop(224)])


torchvision.transforms.Rescale(256)
torchvision.transforms. ToTensor()
'''

import torch
import torchvision
import numpy as np
import torchvision
from torch.utils.data import Dataset , DataLoader


class WineDataSet(Dataset):

    def __init__(self, transform=None) -> None:
        super(WineDataSet, self).__init__()
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1
        self.n_samples = xy.shape[0]
        self.transform= transform

    def __getitem__(self, index):
        sample =  self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs , targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor) -> None:
        super(MulTransform, self).__init__()
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

# dataset = torchvision.datasets.MNIST(
#     root='./data', transform=torchvision.transforms.ToTensor()
# )

dataset = WineDataSet(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features, labels)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataSet(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features, labels)