import torch
import numpy as np

# f = w * x,  let w = 2

X = np.array([1,2,3,4,5], dtype=np.float32)
Y = np.array([2,4,6,8,10], dtype=np.float32)

w = 0.0

#model pred
def forward(x):
    return w * x

# loss
def loss(y, y_hat):
    return ((y_hat - y) ** 2).mean()

# grad computation
# MSE = 1/N * (w*x - y) ** 2
# dJ/dw = 1/N * 2(w*x - y) * x

def gradient(x, y, y_hat):
    pass