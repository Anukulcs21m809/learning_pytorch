import torch
import torch.nn as nn

'''
## lazy method (bound to serialized data and the exact path that is used to save the model)
torch.save(model, PATH)

model = torch.load(PATH)
model.eval()

## better saving technique
PATH = 'model.pth'
torch.save(model.state_dict(), PATH)

# create the same model and replace the paramters with the one from the loaded
model = Model(*args, **kwargs)  --> e.g. Model(input_features = 6)
model.load_state_dict(torch.load(PATH))
model.eval()

'''

'''
#creating checkpoints

optimizer = torch.optim.SGD(model.parameters, lr=learning_rate)

checkpoint = {
    "epoch" : 90,
    "model_state" : model.state_dict(),
    "optim_state" : optimizer.state_dict()
}

torch.save(checkpoint, "checkpoint.pth")

#loading a checkpoint
loaded_checkpoint = torch.load('checkpoint.pth')
epoch = loaded_checkpoint['epoch']

model = Model(input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint['model_state'])
optimizer.load_state_dict(loaded_checkpoint['optim_state'])

'''

class Model(nn.Module):
    def __init__(self, n_input_features) -> None:
        super(Model, self).__init__()
        self.lin = nn.Linear(n_input_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_pred = self.sigmoid(self.lin(x))
        return y_pred

model = Model(n_input_features=6)

FILE = 'model.pth'
torch.save(model, FILE)