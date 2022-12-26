import torch

x = torch.randn(3, requires_grad=True)
print(x)

y =  x + 2
print(y)

z = y * y * 2

# z = z.mean()
print(z)


# since mean is a scalar we dont need to specify a vector for the gradient
# but if z was a vector we need to pass a vector (this is not very clear)

v = torch.tensor([0.1, 1, 0.001], dtype=torch.float32)
z.backward(v)
print(x.grad)

## preventing the gradient history

# trailing _ means that the function is inplace
x.requires_grad_(False)
print(x)

# detaching
y = x.detach()
print(y)

# with torch.no_grad()
with torch.no_grad():
    y = x + 2
    print(y)


# gradients get accumulated (summed) for each iteration. So we must clear it 

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_op = (weights * 3).sum()

    model_op.backward()
    print(weights.grad)

    # clear the grads
    weights.grad.zero_()

    # how it is used with optimizers
    #optimizer = torch.optim.SGD(weights, lr=0.01)
    #optimizer.step()
    #optimizer.zero_grad()