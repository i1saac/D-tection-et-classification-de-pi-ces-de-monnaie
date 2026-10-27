import os
import pandas as pd

def generate_csv(data_dir="data/train/", output_csv="data/annotations.csv"):
    # Liste pour stocker les données
    data = []
    
    # Parcourir les sous-dossiers (1_DA, 5_DA, etc.)
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            # Extraire la valeur de la pièce à partir du nom du dossier
            try:
                # Remplacer "_DA" pour obtenir la valeur (ex: "1_DA" -> "1")
                coin_value = folder.replace("_DA", "")
                coin_value = float(coin_value)  # Convertir en float (ex: "1" -> 1.0)
            except ValueError:
                print(f"Ignorer le dossier {folder} : nom invalide")
                continue
            
            # Parcourir les images dans le dossier
            for img_file in os.listdir(folder_path):
                if img_file.endswith((".jpg", ".jpeg", ".png")):
                    # Construire le chemin de l'image
                    img_path = os.path.join(folder_path, img_file)
                    # Ajouter les données au tableau
                    data.append({
                        "image_path": img_path,
                        "value": coin_value
                    })
    
    # Créer un DataFrame et sauvegarder en CSV
    df = pd.DataFrame(data)
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv(output_csv, index=False)
    print(f"Fichier CSV généré : {output_csv}")

if __name__ == "__main__":
    generate_csv()