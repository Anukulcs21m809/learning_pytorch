# 1) Design the model (input, output, forward_pass)
# 2) Construct loss and optimizer
# 3) Training loop
    # - forward pass : compute the predictions
    # - backward pass : gradients
    # - update weights

import torch
import torch.nn as nn

# requires a 2D tensor so that the feature and the samples are clearly known
X = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)


x_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)

ip_size = n_features
op_size = n_features

# model = nn.Linear(ip_size, op_size)

#OR

class LinearRegression(nn.Module):
    def __init__(self, ip_dim, op_dim) -> None:
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(ip_dim, op_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(ip_size, op_size)

print("Prediction before training : f(5) = {}".format(str(model(x_test).item())))

# training
lr = 0.01
n_iters = 200

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_iters):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()

    #update
    optimizer.step()
    
    #zero grad
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print('epoch {} : w = {}, loss = {}'.format(str(epoch + 1), w[0][0].item(), str(l.item())))

print("Prediction after training : f(5) = {}".format(str(model(x_test).item())))