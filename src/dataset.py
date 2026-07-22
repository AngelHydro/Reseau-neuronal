"""Gestion des jeux de données et des transformations.

Ce fichier contient les fonctions et classes nécessaires pour
charger et préparer les données avant l'entraînement.
"""

# Importations nécessaires pour la gestion des jeux de données
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class XCarreDataset(Dataset):
    # Classe représentant un jeu de données personnalisé pour les données y = x²
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.x = torch.linspace(-10, 10, n_samples).unsqueeze(1)  # Génère des valeurs x uniformément espacées

    def __len__(self):
        return self.n_samples  # Retourne le nombre d'échantillons dans le jeu de données

    def __getitem__(self, idx):
        return self.x[idx], self.x[idx]

    def loaders(self, batch_size=32, split_ratio=0.8):
        # Crée des DataLoaders pour l'entraînement et la validation
        train_size = int(split_ratio * len(self))
        val_size = len(self) - train_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader