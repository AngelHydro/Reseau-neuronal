"""Définition d'un perceptron multicouche simple.

Ce module implémente un réseau de neurones feedforward composé
principalement de couches linéaires suivies d'une activation ReLU.
"""

# Importations nécessaires pour la définition du modèle
import torch.nn as nn


class MLP(nn.Module):
    # Classe représentant un réseau de neurones multicouche simple
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__() # Initialisation de la classe parente nn.Module
        self.model = nn.Sequential( # Définition de la séquence de couches du modèle
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # Effectue la propagation avant du modèle
        return self.model(x)
    