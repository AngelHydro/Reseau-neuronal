"""Définition d'un perceptron multicouche simple.

Ce module implémente un réseau de neurones feedforward composé
principalement de couches linéaires suivies d'une activation ReLU.
"""

import torch.nn as nn


class MLP(nn.Module):
    """Classe représentant un réseau de neurones multicouche simple."""

    def __init__(self, input, hidden, output):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output),
        )

    def forward(self, x):
        """Effectue la propagation avant du modèle."""
        return self.model(x)