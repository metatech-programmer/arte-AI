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

## Explicacion del codigo

Aquí tienes la explicación detallada de cada uno de los archivos de tu proyecto, en formato Markdown para que lo uses directamente en un archivo `README.md` de GitHub.

---

# Proyecto: Clasificación de Arte con IA

Este repositorio contiene el código para un sistema de clasificación de imágenes de arte utilizando aprendizaje profundo. El proyecto incluye tres componentes principales:

1. **Preprocesamiento de datos** (`preprocesamiento.py`).
2. **Entrenamiento del modelo** (`entrenamiento.py`).
3. **Clasificación en tiempo real** (`clasificacion_tiempo_real.py`).

A continuación, se describe el propósito y funcionamiento de cada script.

---

## **1. Preprocesamiento de datos (`preprocesamiento.py`)**

### Propósito
Prepara las imágenes originales para el entrenamiento del modelo, organizándolas en carpetas de entrenamiento y validación, y ajustando su tamaño.

### Explicación del código
- **`organizar_imagenes(directorio_origen, directorio_destino)`**  
  - Toma imágenes desde `directorio_origen`.
  - Crea subcarpetas para entrenamiento (`train`) y validación (`val`) dentro de `directorio_destino`.
  - Divide las imágenes en un 80% para entrenamiento y 20% para validación.
  - Copia las imágenes a las carpetas correspondientes.

- **`redimensionar_imagenes(directorio, tamaño=(224, 224))`**  
  - Ajusta el tamaño de las imágenes en el directorio al especificado, utilizando una interpolación de alta calidad.

- **Ejecución del script**  
  El script ejecuta las funciones anteriores en las rutas de origen y destino configuradas:
  ```python
  directorio_origen = "datos/imagenes_originales"
  directorio_destino = "datos/imagenes_procesadas"
  ```

### Resultado
Genera dos carpetas en `datos/imagenes_procesadas`: `train` y `val`, con imágenes redimensionadas y listas para el entrenamiento.

---

## **2. Entrenamiento del modelo (`entrenamiento.py`)**

### Propósito
Entrena un modelo de aprendizaje profundo basado en ResNet50 para clasificar imágenes de arte en diferentes categorías.

### Explicación del código
- **Preprocesamiento de datos**  
  - Utiliza `ImageDataGenerator` para aplicar aumentos de datos (rotación, desplazamiento, espejado) y escalar valores de píxeles.
  - Configura generadores para carpetas de entrenamiento (`train`) y validación (`val`).

- **Definición del modelo**  
  - Utiliza ResNet50 preentrenado como base (pesos de `imagenet`).
  - Agrega capas densas, de agrupamiento global y de regularización (`Dropout`) para mejorar el rendimiento.

- **Entrenamiento del modelo**  
  - Configura una tasa de aprendizaje inicial (`0.0001`) y entrena el modelo por 20 épocas.
  - Guarda el modelo entrenado como `modelos/modelo_clasificacion_arte.h5`.

- **Visualización de resultados**  
  - Genera gráficos de precisión y pérdida del entrenamiento y validación.
  - Guarda los gráficos en `resultados/resultados_entrenamiento.png`.

### Resultado
Obtendrás un modelo entrenado y gráficos que muestran el rendimiento durante el entrenamiento.

---

## **3. Clasificación en tiempo real (`clasificacion_tiempo_real.py`)**

### Propósito
Detecta y clasifica objetos en tiempo real utilizando la cámara, combinando YOLO para la detección y un modelo de clasificación para la predicción.

### Explicación del código
- **Cargar recursos**  
  - Carga el modelo entrenado desde `modelos/modelo_clasificacion_arte.h5`.
  - Obtiene las clases desde la carpeta `datos/imagenes_procesadas/train`.
  - Configura YOLO para detección de objetos.

- **Captura de video**  
  - Inicializa la cámara para procesar fotogramas en tiempo real.

- **Detección y clasificación**  
  - Utiliza YOLO para detectar objetos y extraer las regiones correspondientes.
  - Clasifica cada objeto detectado utilizando el modelo entrenado.
  - Muestra en pantalla el nombre de la clase predicha y la confianza.

- **Interfaz en tiempo real**  
  - Muestra la imagen procesada en una ventana.
  - Cierra la ventana si se presiona la tecla `q`.

### Resultado
Una ventana que muestra el video en tiempo real con los objetos detectados y clasificados.

---
