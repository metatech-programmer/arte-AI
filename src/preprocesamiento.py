import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

def organizar_imagenes(directorio_origen, directorio_destino):
    categorias = [d for d in os.listdir(directorio_origen) if os.path.isdir(os.path.join(directorio_origen, d))]
    for categoria in categorias:
        os.makedirs(os.path.join(directorio_destino, 'train', categoria), exist_ok=True)
        os.makedirs(os.path.join(directorio_destino, 'val', categoria), exist_ok=True)
        
        imagenes = [f for f in os.listdir(os.path.join(directorio_origen, categoria)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        train_imgs, val_imgs = train_test_split(imagenes, test_size=0.2, random_state=42)
        
        for img in train_imgs:
            shutil.copy2(os.path.join(directorio_origen, categoria, img), os.path.join(directorio_destino, 'train', categoria))
        
        for img in val_imgs:
            shutil.copy2(os.path.join(directorio_origen, categoria, img), os.path.join(directorio_destino, 'val', categoria))

def redimensionar_imagenes(directorio, tamaño=(224, 224)):
    for root, _, files in os.walk(directorio):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                ruta_imagen = os.path.join(root, file)
                with Image.open(ruta_imagen) as img:
                    img_redimensionada = img.resize(tamaño, Image.LANCZOS)
                    img_redimensionada.save(ruta_imagen)

if __name__ == "__main__":
    directorio_origen = "datos/imagenes_originales"
    directorio_destino = "datos/imagenes_procesadas"
    organizar_imagenes(directorio_origen, directorio_destino)
    redimensionar_imagenes(os.path.join(directorio_destino, 'train'))
    redimensionar_imagenes(os.path.join(directorio_destino, 'val'))
    print("Preprocesamiento completado.")