import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(in_features=5, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

for epoch in range(10):
    # Simulate training: update weights based on some example data
    data = torch.randn(10, 5)
    target = torch.randn(10)
    output = model(data)
    loss = loss_fn(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Model trained!')
import  os

torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '../build/my_model_state.pt'))
print('Model state saved!')

imodel = MyModel()
imodel.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../build/my_model_state.pt')))
imodel.eval()  # Set model to evaluation mode

# Predict on new data
new_data = torch.randn(1, 5)
output = model(new_data)
print('Predicted output from orig:', output)

ioutput = imodel(new_data)
print('Predicted output from saved:', ioutput)