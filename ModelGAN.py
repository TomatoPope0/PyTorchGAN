import torch
import torch.nn as nn
from Maxout import Maxout

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin1 = nn.Sequential(
            nn.Linear(100, 1200),
            nn.ReLU()
        )
        self.lin2 = nn.Sequential(
            nn.Linear(1200, 1200),
            nn.ReLU(),
        )
        self.lin3 = nn.Sequential(
            nn.Linear(1200, 784), # 28 * 28
            nn.Sigmoid()
        )

    def forward(self, x):
        l1 = self.lin1(x)
        l2 = self.lin2(l1)
        ge = self.lin3(l2)
        return ge

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.max1 = Maxout(784, 240, 5)
        self.max2 = Maxout(240, 240, 5)
        self.lin1 = nn.Sequential(
            nn.Linear(240, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        m1 = self.max1(x)
        m2 = self.max2(m1)
        pred = self.lin1(m2)
        return pred
