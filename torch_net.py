# -*- coding: utf-8 -*-
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_unit=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(8, n_unit)
        self.fc2 = nn.Linear(n_unit, n_unit)
        self.fc3 = nn.Linear(n_unit, n_unit)
        self.fc4 = nn.Linear(n_unit, n_unit)
        self.fc5 = nn.Linear(n_unit, 2)

        self.bn1 = nn.BatchNorm1d(n_unit)
        self.bn2 = nn.BatchNorm1d(n_unit)
        self.bn3 = nn.BatchNorm1d(n_unit)
        self.bn4 = nn.BatchNorm1d(n_unit)

        self.initialization()

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.relu(self.bn3(self.fc3(h)))
        h = F.relu(self.bn4(self.fc4(h)))
        return self.fc5(h)

    def initialization(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc4.weight)
