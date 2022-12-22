import torch

# x = torch.empty(2,2,2,3)
# # torch.ones , torch.zeros

# x = torch.ones(2,2, dtype=torch.float16)

# print(x)

# x = torch.rand(2,2)
# y = torch.rand(2,2)

# print(x)
# print(y)

# z = x + y

# # can do + , - , * , /
# print(z)
# z = torch.add(x,y)

# print(z)
# y.add_(x)
# print(y)

# # slicing can be done same as lists


# print(x[1,1].item())

# view function can be used to resize the tensor

# x = torch.rand(4,4)
# # we can either specify the dimension of the 1d vector
# y = x.view(16)

# print(y)

# # or we can say how many elements in a row, -1 is important
# y = x.view(-1, 4)

# print(y)
# print(y.size())


import numpy as np

# # changing to numpy still points at the same memory location
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
# # print(type(b)) 
# a.add_(1)
# print(a)
# print(b)

# tells pytorch that the tensor's gradients need to be computed
x = torch.ones(5, requires_grad=True)

