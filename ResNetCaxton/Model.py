import torch
import torch.nn as nn
import torch.optim as optim

class CustomResNet(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Identity()  # Remove the original fully connected layer

        #output layer for each of the 4 criteria - three outputs for each criterion: 'high', 'low', 'good'
        self.fc1 = nn.Linear(2048, 3)
        self.fc2 = nn.Linear(2048, 3)
        self.fc3 = nn.Linear(2048, 3)
        self.fc4 = nn.Linear(2048, 3)

    def forward(self, x):
        print("beginning forward pass")
        x = self.base_model(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        out4 = self.fc4(x)
        print("forward pass finished")
        return out1, out2, out3, out4






