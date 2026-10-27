import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from preprocessing import load_dataset
from feature_extraction import extract_features

def train_classifier(csv_path="data/annotations.csv", model_path="models/classifier_model.pkl"):
    # Charger les données
    images, labels = load_dataset(csv_path)
    if not images or not labels:
        raise ValueError("Aucune donnée chargée à partir du CSV. Vérifiez data/annotations.csv.")
    
    # Extraire les caractéristiques pour chaque image
    print("Extraction des caractéristiques...")
    X = [extract_features(img_hsv, img_gray) for img_hsv, img_gray in images]
    y = labels
    
    # Diviser les données en ensembles d'entraînement et de test
    print("Division des données...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le classificateur SVM
    print("Entraînement du modèle SVM...")
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    
    # Évaluer le modèle
    y_pred = clf.predict(X_test)
    print(f"Précision : {accuracy_score(y_test, y_pred):.2f}")
    
    # Sauvegarder le modèle
    if not os.path.exists("models"):
        os.makedirs("models")
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Modèle sauvegardé dans {model_path}")
    
    return clf

def predict_coin(img_hsv, img_gray, model_path="models/classifier_model.pkl"):
    # Extraire les caractéristiques
    features = extract_features(img_hsv, img_gray)
    
    # Charger le modèle
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    # Prédire la classe
    return clf.predict([features])[0]

if __name__ == "__main__":
    train_classifier()