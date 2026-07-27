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
        self.x_raw = torch.linspace(-1000, 1000, n_samples).unsqueeze(1)  # Génère n_samples points uniformément espacés entre -1000 et 1000
        self.y_raw = self.x_raw ** 2

        # Normalisation (centrer-réduire)
        self.x_mean, self.x_std = self.x_raw.mean(), self.x_raw.std()
        self.y_mean, self.y_std = self.y_raw.mean(), self.y_raw.std()

        self.x = (self.x_raw - self.x_mean) / self.x_std
        self.y = (self.y_raw - self.y_mean) / self.y_std

    def __len__(self):
        return self.n_samples  # Retourne le nombre d'échantillons dans le jeu de données

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]   # Retourne l'échantillon à l'index idx sous forme de tuple (x, y)

    def get_loaders(self, batch_size=32, train_ratio=0.8):
        # Crée des DataLoaders pour l'entraînement et la validation
        train_size = int(train_ratio * len(self)) # Calcule la taille du jeu de données d'entraînement en fonction du ratio spécifié
        val_size = len(self) - train_size # Calcule la taille du jeu de données de validation comme le reste des échantillons
        train_dataset, val_dataset = random_split(self, [train_size, val_size]) # Divise le jeu de données en ensembles d'entraînement et de validation

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Crée un DataLoader pour l'ensemble d'entraînement avec mélange des données
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Crée un DataLoader pour l'ensemble de validation sans mélange des données
        return train_loader, val_loader