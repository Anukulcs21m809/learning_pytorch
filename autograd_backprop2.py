import torch

X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

def loss(y, y_hat):
    return ((y_hat - y)**2).mean()

# dJ/dW = 1/N 2x (w*x - y)

print("Prediction before training : f(5) = {}".format(str(forward(5))))

#training

lr = 0.01
n_iters = 100

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    l.backward()

    #update
    with torch.no_grad():
        w -= (lr * w.grad)
    
    #zero grad
    w.grad.zero_()
    
    if epoch % 10 == 0:
        print('epoch {} : w = {}, loss = {}'.format(str(epoch + 1), str(w), str(l)))

print("Prediction after training : f(5) = {}".format(str(forward(5))))