Salut !

Context: Nous allons faire du developpement en tant qu'expert python pour le developpement d'un AI.

Le but: Le but de cette AI est de faire un service qui grâce à des API permettra de créer des modèles, de les entrainer puis de lancer des recherches.

Consignes: Merci de respecter le code existant, c'est à dire à ne pas retirer des élements utilient présent si ce n'est pas nécéssaire aux corrections et amélioration, de ne pas retirer les commentaires, j'aime garder un haut niveau de commentaire dans le code, le projet à vocation a être pédagogique, le but est aussi de réaliser une AI à partir de rien.

Le projet : Je vais te partager à travers plusieurs prompts l'utilisation des APIS avec des appels REST pour te permettre de comprendre la structure des données
 puis dans une second prompt le code des APIS, des usescases (vide pour l'instant), un machine learning exemple, fait en une analyse approfondi, pour une compréhension profonde des logiques et algorithme, attends d'avoir tout pour commencer à réfléchir

Les dépendances : fastapi
uvicorn
pydantic
torch
httpx
numpy
scikit-learn

N'oublie pas aussi de mettre à jour tes connaissances sur internet sur les sujets et technologies identifiés.

Problèmatique : Nous allons implémenter un modèle GRU pour puppet-o2. 
 puppet-o2 sert à catégoriser des descriptions de ticket au support, les descriptions arrivent déjà tokenifier.
 Elle qui utilisera PyTorch, dans la couche des usecases nous pourrons faire les traitements des données (transformation)
 et nous créerons app\machine_learning\neural_network_gru.py

*appels REST*
###
POST {{host}}/create_model
Content-Type: application/json

{
  "name": "puppet-o2",
  "neural_network_type": "GRU"
}

###
POST {{host}}/train_model
Content-Type: application/json

{
  "name": "puppet-o2",
  "neural_network_type": "GRU",
  "training_data": [
    {
      "category": "Problème de connexion",
      "tokens": [
        "avoir",
        "problème",
        "connexion",
        "<PRODUCT>",
        "rappeler",
        "<PHONE>",
        "<NAME>",
        "oriente"
      ]
    },
    {
      "category": "Problème de connexion",
      "tokens": [
        "arriver",
        "plus",
        "connecter",
        "<PRODUCT>",
        "rentre",
        "codes",
        "mot de passe",
        "da",
        "met",
        "connexion",
        "expirer",
        "tester",
        "depuis",
        "extranet",
        "directement",
        "nouveau",
        "lien",
        "meme",
        "résultat",
        "aide",
        "cb",
        "appel",
        "retour",
        "mail",
        "recontacter",
        "<NAME>",
        "mail",
        "contacte",
        "suite",
        "ticket",
        "connexion",
        "expirer",
        "<PRODUCT>",
        "quel",
        "navigateur",
        "conntecter",
        "constater",
        "problème",
        "firefox",
        "safarir",
        "incident",
        "cours",
        "résolution",
        "côter",
        "savoir",
        "suite",
        "demande",
        "groupe",
        "garantir",
        "fonctionnement",
        "optimal",
        "chrome",
        "edge"
      ]
    },
    {
      "category": "Accès utilisateur",
      "tokens": [
        "<PRODUCT>",
        "souhaite",
        "créer",
        "autre",
        "catégorie",
        "exploitation",
        "catégorie",
        "gestion",
        "pouvoir",
        "être",
        "autre",
        "commerce",
        "puis",
        "être",
        "capable",
        "gérer",
        "droit",
        "utilisateur",
        "catégorie",
        "être",
        "possible",
        "coffre",
        "créer",
        "utilisateur",
        "connecté",
        "<PRODUCT>",
        "devoir",
        "contacter",
        "être",
        "ajouter",
        "nouveaux",
        "coffre"
      ]
    },
    ...
  ]
}


###
POST {{host}}/search
Content-Type: application/json

{
  "name": "puppet-o2",
  "neural_network_type": "GRU",
  "vector": [
      "ticket",
      "créer",
      "suite",
      "échange",
      "mails",
      "lucile",
      "delbarre",
      "pouvoir",
      "rajouterr",
      "emma",
      "georges",
      "coffre",
      "citer",
      "objet",
      "<ENTREPRISE>",
      "lucile",
      "emma",
      "emma",
      "bien",
      "ajouté",
      "coffre",
      "<COMPANY>",
      "bonne",
      "matiné",
      "<NAME>",
      "<ENTREPRISE>"
    ]
}

###
POST {{host}}/test
Content-Type: application/json

{
  "name": "puppet-o2",
  "neural_network_type": "GRU",
  "test_data":[
    {
      "category": "Création de client",
      "tokens": [
        "comme",
        "convenue",
        "mail",
        "création",
        "client",
        "sasre",
        "ecovitres",
        "résolu"
      ]
    },
    {
      "category": "Accès utilisateur",
      "tokens": [
        "client",
        "couvivet",
        "pouvoir",
        "modifier",
        "ladresse",
        "mail",
        "arnord",
        "dozot",
        "arnolddozotcouvivetcom",
        "nair",
        "sre",
        "faire",
        "car",
        "voir",
        "utilisateur",
        "<PRODUCT>",
        "couvivet",
        "support",
        "<PRODUCT>",
        "comme",
        "convenue",
        "téléphone",
        "problème",
        "mail",
        "résolu",
        "si",
        "venir",
        "arriver",
        "modifier",
        "utilisateur",
        "être",
        "être",
        "tout",
        "simplement",
        "rattacher",
        "banque",
        "hésite",
        "contacter",
        "cas",
        "soucis",
        "<NAME>",
        "<NAME>"
      ]
    },
    ...
  ]
}

*app\usecases\gru\create_model_gru.py*
from app.services.logger import logger
from fastapi import HTTPException  # type: ignore
from app.repositories.memory import model_exists, save_model

def create_model_gru(name: str):
    logger.info(f"Type de machine learning utilisé pour la création du model 'GRU'")

    if model_exists(name):
        raise HTTPException(status_code=400, detail="Model already exists")

    # Enregistrer le modèle avec le glossaire et le dictionnaire d'indices
    model_data = {
        "neural_network_type": "GRU" # Enregistrement du type de modèle
    }
    save_model(name, model_data)

    return {"status": "model created", "model_name": name}

*train_model_gru.py*
import joblib
from typing import List
from app.services.logger import logger
from fastapi import HTTPException  # type: ignore
from app.repositories.memory import get_model, update_model
from app.apis.models.gru_training_data import GRUTrainingData

def train_model_gru(name: str, training_data: List[GRUTrainingData]):
    """
    Entraîne le modèle avec des données d'entraînement fournies.
    :param name: Nom du modèle
    :param training_data: Liste des données d'entraînement
    """
    model = get_model(name)
    
    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if training_data is None or len(training_data) == 0:
        raise HTTPException(status_code=400, detail="No training data provided or training data is empty")
    
    logger.info(f"Type de machine learning utilisé pour l'entraînement GRU")
    
    # Transformation des données
    #... = transform_data(training_data)
    
    # Entraîner le réseau de neurones
    #nn_model, losses = train_gru(...)
    
    # Enregistrer le modèle de réseau de neurones entraîné
    #update_model(name, {
    #    "nn_model": nn_model
    #    ...
    #})
    
    return {"status": "training completed", "model_name": name}


*app\usecases\gru\search_model_gru.py*
import joblib
from typing import List
from app.repositories.memory import get_model
from fastapi import HTTPException # type: ignore

def search_model_gru(name: str, vector: List[str]):
    model = get_model(name)
    
    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    nn_model = model.get("nn_model", None)
    if nn_model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    # Transformer les données d'entrée
    #input_processed = process_input_data(...)
    
    # Prédiction
    #predicted = predict(...)
    
    return {"category": "..."}


*app\usecases\gru\mesure_gru.py*
import joblib
from app.services.logger import logger
from app.repositories.memory import get_model
from app.machine_learning.neural_network_simple import predict
from app.usecases.simple_nn.simple_nn_commons import process_input_data

def mesure_gru(name, test_data):
    try:
        total_error = 0
        correct_predictions = 0
        total_tests = len(test_data)
        model = get_model(name)
        nn_model = model.get("nn_model", None)
        if nn_model is None:
            raise Exception("Model not trained yet")
        
        for data in test_data:
            # Préparer les données d'entrée
            input_data = [
                data.tokens
            ]
            expected = data.category
            
            # Transformer les données d'entrée
            #... = process_input_data(input_data)
            
            # Prédiction
            #predicted = predict(nn_model)
            predicted = expected #todo
            
            # Calcul de l'erreur
            if(predicted == expected) :
                correct_predictions += 1
            else :
                total_error += 1
            
            # Utiliser le logger pour les sorties
            logger.info(f"Requête: {input_data}")
            logger.info(f"Category attendu: {expected}, Category donné par le modèle: {predicted}")
        
        # Afficher le nombre de prédictions correctes sur le nombre total d'essais
        logger.info(f"Nombre de prédictions correctes: {correct_predictions}/{total_tests}")
    except Exception as e:
        logger.error(f"Une erreur s'est produite pendant la mesure : {str(e)}")
        raise Exception(f"Une erreur s'est produite pendant la mesure : {str(e)}")

*Machine learning exemple*
import time
import torch
import torch.nn as nn
import torch.optim as optim
from app.services.logger import logger

# Définition du réseau de neurones SimpleNN amélioré avec plusieurs couches et du Dropout pour la régularisation
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialisation du réseau SimpleNN avec :
        - input_size : taille du vecteur d'entrée
        - hidden_size : nombre de neurones dans la couche cachée
        - output_size : taille du vecteur de sortie
        """
        super(SimpleNN, self).__init__()
        # Première couche fully connected
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Fonction d'activation ReLU après la première couche
        self.relu1 = nn.ReLU()
        # Deuxième couche fully connected pour ajouter de la profondeur au réseau
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Fonction d'activation ReLU après la deuxième couche
        self.relu2 = nn.ReLU()
        # Troisième couche fully connected (couche de sortie)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # Dropout pour réduire le surapprentissage (50% des neurones désactivés aléatoirement)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Fonction de passage avant (forward) :
        - Prend un vecteur d'entrée x et le passe à travers les couches du réseau.
        - Applique des fonctions d'activation ReLU et un Dropout pour la régularisation.
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout(out)  # Applique Dropout après la première couche cachée
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)  # Pas de fonction d'activation après la dernière couche
        return out

# Fonction pour entraîner le modèle SimpleNN avec Early Stopping
def train_model_nn(features_processed, targets_standardized, input_size, epochs=3000, learning_rate=0.001, patience=10, improvement_threshold=0.00001):
    """
    Entraîne le modèle de réseau de neurones avec les features et les targets fournis.
    """
    try:
        # Initialisation des tailles
        hidden_size = 128  # Taille des couches cachées
        output_size = 1  # La taille de sortie doit être 1 (pour prédire un prix unique)

        # Création du modèle SimpleNN
        model = SimpleNN(input_size, hidden_size, output_size)

        # Utilisation de la fonction de perte
        criterion = nn.MSELoss()
        # Optimiseur Adam avec un taux d'apprentissage initial
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Conversion en tenseurs
        inputs_tensor = torch.tensor(features_processed, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_standardized, dtype=torch.float32).unsqueeze(1)

        # Liste pour stocker la perte à chaque époque
        losses = []
        start_time = time.time()

        best_loss = float('inf')  # Initialisation de la meilleure perte à une grande valeur
        epochs_without_improvement = 0  # Compteur pour l'arrêt anticipé

        # Boucle d'entraînement
        for epoch in range(epochs):
            optimizer.zero_grad()  # Réinitialiser les gradients
            outputs = model(inputs_tensor)  # Passer les entrées dans le modèle
            loss = criterion(outputs, targets_tensor)  # Calculer la perte
            loss.backward()  # Calculer les gradients
            optimizer.step()  # Mettre à jour les poids

            current_loss = loss.item()  # Récupérer la perte actuelle
            losses.append(current_loss)

            # Afficher la perte toutes les 10 époques
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Époque {epoch + 1}/{epochs}, Perte: {current_loss}")

            # Vérification de l'amélioration de la perte
            if current_loss < best_loss - improvement_threshold:
                best_loss = current_loss
                epochs_without_improvement = 0  # Réinitialiser si la perte s'améliore
            else:
                epochs_without_improvement += 1  # Incrémenter si aucune amélioration

            # Arrêter l'entraînement si la perte n'améliore plus
            if epochs_without_improvement >= patience:
                logger.info(f"Arrêt anticipé à l'époque {epoch + 1}. Perte optimale atteinte : {best_loss:.6f}")
                break

        # Calcul du temps total d'entraînement
        total_training_time = time.time() - start_time
        total_parameters = sum(p.numel() for p in model.parameters())

        # Statistiques sur les pertes
        avg_loss = sum(losses) / len(losses)
        min_loss = min(losses)
        max_loss = max(losses)

        # Log des statistiques d'entraînement
        logger.info(f"Temps total d'entraînement: {total_training_time:.2f} secondes")
        logger.info(f"Nombre total de paramètres: {total_parameters}")
        logger.info(f"Perte moyenne: {avg_loss}")
        logger.info(f"Perte minimale: {min_loss}")
        logger.info(f"Perte maximale: {max_loss}")
        logger.info(f"Perte finale après {len(losses)} epochs : {losses[-1]}")

        return model, losses

    except RuntimeError as e:
        raise RuntimeError(f"Erreur lors de l'entraînement du modèle : {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Erreur inattendue lors de l'entraînement : {str(e)}") from e

# Fonction pour faire une prédiction avec le modèle SimpleNN
def predict(model, input_processed, targets_mean, targets_std):
    try:
        # Conversion en tenseur
        input_tensor = torch.tensor(input_processed, dtype=torch.float32)
        
        # Prédiction
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Conversion du tenseur de sortie en valeur scalaire
        predicted_standardized = output_tensor.item()
        
        # Déstandardisation de la prédiction
        predicted_price = predicted_standardized * targets_std + targets_mean
        
        return float(predicted_price)
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise
