"""
Fichier train.py
Coordonne l'ensemble du processus d'entraînement : charge la configuration, les données et le modèle,
entraîne le réseau, évalue ses performances et sauvegarde les poids appris.
"""

# Les importations
import os
import torch
import yaml

from model import MLP
from dataset import XCarreDataset


def train(model, train_loader, criterion, optimizer):
    """
    Fonction pour faire une boucle d'entrainement du modèle, prend en paramètres :
        - model : le modèle du réseau de neuronne
        - train_loader : les données d'entraînement sous forme de batchs.
        - criterion : fonction qui calcule le coût (loss)
        - optimizer : l'optimisateur de poids

    Elle retourne :
        - Le loss moyenne par batch
    """
    model.train()
    running_loss = 0

    for batch_x, batch_y in train_loader:
        prediction = model(batch_x)
        loss = criterion(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    """
    Fonction pour évaluer la performance de l'entrainement, prend en paramètres :
        - model : le modèle du réseau de neuronne
        - val_loader : les données d'entraînement sous forme de batchs.
        - criterion : fonction qui calcule le coût (loss)
        
    Elle retourne :
        - Le loss moyenne par batch
    """
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            running_loss += loss.item()

    return running_loss / len(val_loader)


def saveCheckpoint(model, optimizer, epoch, save_path):
    """
    Fonction qui va enregistrer les entrainements dans le dossier 'checkpoints'
    sous l'extension '.pt'
    """

    """ ?? Qu'est-ce qu'on enregistre finalement ?? """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, save_path)

def importSaveCheckpoint(save_path):
    """
    Fonction qui importe un entrainement du dossier 'checkpoints'
    qui retroune les paramètres enregistrés du save
    """
    checkpoint = torch.load(save_path)

    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    startEpoch = checkpoint["epoch"] + 1

    return startEpoch, model_state_dict, optimizer_state_dict

def main():
    """
    Fonction principale du programme.
    Initialise le modèle, les données, la fonction de coût et l'optimiseur,
    puis lance l'entraînement, l'évaluation et la sauvegarde du modèle.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, os.pardir, "config.yaml"))

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["seed"])

    (n_samples, batch_size, train_ratio) = (config["data"]["n_samples"],
                                            config["data"]["batch_size"],
                                            config["data"]["train_ratio"])
    
    (input_dim, hidden_dim, output_dim) = (config["model"]["input_dim"],
                                           config["model"]["hidden_dim"],
                                           config["model"]["output_dim"])

    (epochs, learning_rate, optimizer_name) = (config["training"]["epochs"],
                                               config["training"]["learning_rate"],
                                               config["training"]["optimizer"])
    save_path = config["checkpoint"]["save_path"]
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(os.path.join(script_dir, save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Chargement du modèle
    model = MLP(input_dim, hidden_dim, output_dim)

    dataset = XCarreDataset(n_samples) # 'n_samples' : nombre de données
    train_loader, val_loader = dataset.get_loaders(batch_size, train_ratio) # 'batch_size' : nombre de données par lot

    # Fonction de coût
    criterion = torch.nn.MSELoss()

    # L'optimisateur chargé de mettre à jour les poids
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)


    # On recharge la dernière save_checkpoint, si elle exite
    startEpoch = 1
    if os.path.exists(save_path):
        startEpoch, model_state_dict, optimizer_state_dict = importSaveCheckpoint(save_path)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)    
    endEpoch = startEpoch + epochs - 1

    print("\nL'entrainement commence à Epoch", startEpoch, "\net se termine à Epoch", endEpoch) # à supprimer plus tard

    # L'entrainement et l'évaluation en 'epochs' boucles
    for epoch in range(startEpoch, endEpoch + 1):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)


        # ===== afficher résultat des valeurs de l'entrainement dans la console ===== #
        print(f"\nEpoch {epoch}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")

        print("\nExemples de prédictions :")

        # Récupère le premier batch du jeu de validation
        batch_x, batch_y = next(iter(val_loader))
        model.eval()
        with torch.no_grad():
            predictions = model(batch_x)

        # Dénormalisation pour affichage lisible
        x_reel = batch_x * dataset.x_std + dataset.x_mean
        y_reel = batch_y * dataset.y_std + dataset.y_mean
        pred_reel = predictions * dataset.y_std + dataset.y_mean

        for x, pred, y in zip(x_reel[:5], pred_reel[:5], y_reel[:5]):
            print(f"x={x.item():.2f} -> prédiction={pred.item():.2f} (attendu={y.item():.2f})")

        print("-" * 50)

    # Sauvegarde
    saveCheckpoint(model, optimizer, endEpoch, save_path)
    """ il faudrait revoir où on veut faire le checkpoint
        - soit à chaque tour de boucle d'entrainement
        - soit à la fin de l'entrainement
         ?
    """



if __name__ == "__main__":
    main()
