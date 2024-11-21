# Clasificación de Arte en Tiempo Real

Bienvenido a nuestra aplicación de clasificación de arte en tiempo real. Esta aplicación utiliza técnicas de aprendizaje automático para clasificar imágenes de arte en diferentes categorías.

## Requisitos

* Python 3.x instalado en tu sistema
* Dependencias necesarias instaladas utilizando el archivo `requirements.txt`
* Un conjunto de imágenes de arte para entrenar y probar el modelo

## Preparación del entorno

1. Instala las dependencias necesarias utilizando el archivo `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

2. Asegúrate de tener Python 3.x instalado en tu sistema.

## Preprocesamiento de imágenes

1. Ejecuta el script `src/preprocesamiento.py` para organizar y redimensionar las imágenes:
   ```
   python src/preprocesamiento.py
   ```

   Este script organizará tus imágenes en carpetas de entrenamiento y validación, y las redimensionará al tamaño adecuado.

## Entrenamiento del modelo

1. Ejecuta el script `src/entrenamiento.py` para entrenar el modelo de clasificación:
   ```
   python src/entrenamiento.py
   ```

   Este proceso puede llevar tiempo dependiendo de la cantidad de imágenes y la potencia de tu computadora.

## Clasificación en tiempo real

1. Ejecuta el script `src/clasificacion_tiempo_real.py` para clasificar imágenes en tiempo real:
   ```
   python src/clasificacion_tiempo_real.py
   ```

   Asegúrate de tener una cámara web conectada a tu computadora.

## Consideraciones adicionales

* Asegúrate de que la estructura de directorios sea correcta:
  ```
  proyecto/
  ├── datos/
  │   ├── imagenes_originales/
  │   └── imagenes_procesadas/
  ├── modelos/
  ├── resultados/
  ├── src/
  │   ├── preprocesamiento.py
  │   ├── entrenamiento.py
  │   └── clasificacion_tiempo_real.py
  ├── requirements.txt
  ├── yolov3.cfg
  ├── yolov3.weights
  └── coco.names
  ```

  * Si encuentras problemas con la detección de objetos, verifica que los archivos `yolov3.cfg`, `yolov3.weights` y `coco.names` estén en el directorio correcto y sean accesibles por el script.

## Modelo utilizado

El modelo utilizado es un ResNet50 pre-entrenado en ImageNet, con una capa adicional de clasificación para las categorías de arte.

## Categorías de arte

Las categorías de arte utilizadas en este proyecto son:
+ Pintura
+ Escultura
+ Fotografía
+ Grabado
+ Dibujo

## Ejemplo de uso

Para clasificar una imagen en tiempo real, ejecuta el script `src/clasificacion_tiempo_real.py` y apunta la cámara web a la imagen que deseas clasificar.