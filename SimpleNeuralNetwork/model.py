import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        self.mode = mode
        if mode == 1 or mode == 2:

            self.fc1 = nn.Linear(28*28, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, 10)
        elif mode == 3:

            self.fc1 = nn.Linear(28*28, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, 256)
            self.fc5 = nn.Linear(256, 10)
        else:
            print("Invalid mode", mode, "selected. Please select between 1-3")
            exit(0)

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3

    def model_1(self, X):
        X = torch.flatten(X, start_dim=1)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X

    def model_2(self, X):
        X = torch.flatten(X, start_dim=1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)
        return X

    def model_3(self, X):
        X = torch.flatten(X, start_dim=1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = F.relu(X)
        X = self.fc5(X)
        return X