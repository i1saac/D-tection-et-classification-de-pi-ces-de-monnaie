import cv2
import numpy as np
import pandas as pd

def preprocess_image(image_path, size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_resized, img_hsv, img_gray

def load_dataset(csv_path="data/annotations.csv"):
    try:
        # Charger le fichier CSV
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"Le fichier {csv_path} est vide.")
        
        images = []
        labels = []
        
        # Parcourir chaque ligne du CSV
        for _, row in df.iterrows():
            img_path = row["image_path"]
            label = row["value"]
            try:
                _, img_hsv, img_gray = preprocess_image(img_path)
                images.append((img_hsv, img_gray))
                labels.append(label)
            except Exception as e:
                print(f"Erreur avec l'image {img_path} : {e}")
                continue
        
        if not images or not labels:
            raise ValueError("Aucune image valide chargée à partir du CSV.")
        
        return images, labels
    
    except FileNotFoundError:
        print(f"Le fichier {csv_path} n'a pas été trouvé. Vérifiez le chemin.")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None