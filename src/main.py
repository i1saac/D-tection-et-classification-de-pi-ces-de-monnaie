from detection import detect_coins
import cv2

def main(image_path):
       # Détecter les pièces et calculer la somme
       coins, total_value = detect_coins(image_path)
       
       # Charger l'image pour affichage
       img = cv2.imread(image_path)
       
       # Afficher les résultats
       for coin_value, (x, y, w, h) in coins:
           cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
           cv2.putText(img, f"{coin_value} DA", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
       
       print(f"Nombre de pièces : {len(coins)}")
       print(f"Somme totale : {total_value} DA")
       
       # Afficher l'image avec les annotations
       cv2.imshow("Résultat", img)
       cv2.waitKey(0)
       cv2.destroyAllWindows()

if __name__ == "__main__":
       image_path = "data/test/test_image.jpg"
       main(image_path)