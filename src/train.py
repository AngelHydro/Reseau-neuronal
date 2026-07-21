# Fichier train.py
# Coordonne l'ensemble du processus d'entraînement : charge la configuration, les données et le modèle,
# entraîne le réseau, évalue ses performances et sauvegarde les poids appris.

"""
Les importations :

"""
import torch
import yaml

from model import MLP
from dataset import Xcarre_dataset

def train(model, train_loader, criterion, optimizer):
    """
    Fonction pour faire un boucle d'entrainement du modèle
    """
    model.train()
    running_loss = 0

    for batch_x, batch_y in train_loader:
        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def evaluate():
    """
    
    """
    pass


def save_checkpoint():
    """
    Fonction qui va enregistrer les entrainements dans le dossier 'checkpoints'
    sous l'extension '.pt'
    """
    pass


def main():
    """
    
    """
    model = MLP(input=1, hidden=16, output=1)

    dataset = Xcarre_dataset(n_samples=1000) # 'n_samples' : ça correspond au nombre de données ; Donc soit on enlève soit on garde
    train_loader, val_loader = dataset.get_loaders(batch_size=32)

    # Choisir la Loss

    # Choisir l'optimiseur

    # Boucle d'entraînement

    # Évaluation

    # Sauvegarde




if __name__ == "__main__":
    main()
