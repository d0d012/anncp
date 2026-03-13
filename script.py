import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = SimpleNet()

with open("weights.txt", "w") as f:
    for name, param in model.named_parameters():
        flat_weights = param.detach().flatten().numpy()
        
        for val in flat_weights:
            f.write(f"{val} ")
        f.write("\n") 

print("Weights exported to weights.txt successfully!")