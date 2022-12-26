import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
Y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

w = 0.0


def forward(x):
    return w * x

def loss(y, y_hat):
    return ((y_hat - y)** 2).mean()

# dJ/dW = 1/N 2x (w*x - y)

def gradient(x, y, y_hat):
    return np.dot(2*x, y_hat - y).mean()

print("Prediction before training : f(5) = {}".format(str(forward(5))))

#training

lr = 0.01
n_iters = 10

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    dw = gradient(X, Y, y_pred)

    #update
    w -= (lr * dw)
    
    if epoch % 1 == 0:
        print('epoch {} : w = {}, loss = {}'.format(str(epoch + 1), str(w), str(l)))


