import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(12, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.out(x))  # outputs ∈ (-1,1)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(769, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
