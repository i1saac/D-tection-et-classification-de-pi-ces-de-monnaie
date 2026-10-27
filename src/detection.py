import cv2
import numpy as np
from preprocessing import preprocess_image
from classification import predict_coin

def detect_coins(image_path, model_path="models/classifier_model.pkl"):
    # Charger et prétraiter l'image
    img, img_hsv, img_gray = preprocess_image(image_path, size=(512, 512))
    
    # Détection des contours
    edges = cv2.Canny(img_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coins = []
    total_value = 0
    
    for contour in contours:
        # Ignorer les petits contours (bruit)
        if cv2.contourArea(contour) < 500:
            continue
        
        # Extraire la région d'intérêt (ROI)
        x, y, w, h = cv2.boundingRect(contour)
        coin_img = img[y:y+h, x:x+w]
        coin_hsv = img_hsv[y:y+h, x:x+w]
        coin_gray = img_gray[y:y+h, x:x+w]
        
        # Prédire la valeur de la pièce
        coin_value = predict_coin(coin_hsv, coin_gray, model_path)
        coins.append((coin_value, (x, y, w, h)))
        total_value += float(coin_value)
    
    return coins, total_value