import torch
import torch.nn as nn
import yaml
import pandas as pd
import sklearn

class MLP(nn.Module):
    def __init__(self, input, hidden, output):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x