---

# 🎨 Clasificación de Arte en Tiempo Real con IA

> **Demo en vivo:** [huggingface.co/spaces/MetatechLabs/clasificador-arte](https://huggingface.co/spaces/MetatechLabs/clasificador-arte), el modelo corre en tu navegador con ONNX Runtime Web.

Este proyecto combina técnicas avanzadas de aprendizaje profundo con un sistema en tiempo real para clasificar imágenes de arte en categorías como pintura, escultura, fotografía, grabado y dibujo.

---

## 📚 **Contenido**

- [📚 **Contenido**](#-contenido)
- [📋 **Requisitos**](#-requisitos)
- [📂 **Estructura del Proyecto**](#-estructura-del-proyecto)
- [🚀 **Guía Rápida**](#-guía-rápida)
- [🔍 **Detalles del Código**](#-detalles-del-código)
  - [**1. Preprocesamiento (`src/preprocesamiento.py`)**](#1-preprocesamiento-srcpreprocesamientopy)
    - [**Funcionamiento**:](#funcionamiento)
  - [**2. Entrenamiento (`src/entrenamiento.py`)**](#2-entrenamiento-srcentrenamientopy)
    - [**Funcionamiento**:](#funcionamiento-1)
  - [**3. Clasificación en Tiempo Real (`src/clasificacion_tiempo_real.py`)**](#3-clasificación-en-tiempo-real-srcclasificacion_tiempo_realpy)
    - [**Funcionamiento**:](#funcionamiento-2)
- [📊 **Modelo y Técnicas Utilizadas**](#-modelo-y-técnicas-utilizadas)
  - [**1. ResNet50**](#1-resnet50)
  - [**2. YOLOv3**](#2-yolov3)
  - [**3. Aumentos de Datos**](#3-aumentos-de-datos)
  - [**4. Entrenamiento Transferido**](#4-entrenamiento-transferido)
- [Con esta combinación, el sistema logra una detección y clasificación eficiente, robusta y precisa, ideal para aplicaciones en tiempo real. 🎨](#con-esta-combinación-el-sistema-logra-una-detección-y-clasificación-eficiente-robusta-y-precisa-ideal-para-aplicaciones-en-tiempo-real-)
- [🤝 **Contribuciones**](#-contribuciones)
- [📜 **Licencia**](#-licencia)
  - [¡Explora, aprende y crea con este proyecto de clasificación de arte en tiempo real! 🎨](#explora-aprende-y-crea-con-este-proyecto-de-clasificación-de-arte-en-tiempo-real-)

---

## 📋 **Requisitos**

- **Python 3.x**
- Instalar dependencias:
  ```bash
  pip install -r requirements.txt
  ```
- Archivos necesarios:
  - `yolov3.cfg`, `yolov3.weights`, `coco.names`  
    (estos se utilizan para la detección de objetos en tiempo real).

---

## 📂 **Estructura del Proyecto**

```plaintext
proyecto/
├── datos/
│   ├── imagenes_originales/       # Imágenes sin procesar
│   └── imagenes_procesadas/       # Imágenes listas para entrenar
├── modelos/                       # Modelos entrenados
├── resultados/                    # Gráficos y reportes
├── src/                           # Código fuente
│   ├── preprocesamiento.py        # Preprocesamiento de imágenes
│   ├── entrenamiento.py           # Entrenamiento del modelo
│   └── clasificacion_tiempo_real.py # Clasificación en tiempo real
├── requirements.txt               # Dependencias
├── yolov3.cfg                     # Configuración de YOLOv3
├── yolov3.weights                 # Pesos preentrenados de YOLOv3
└── coco.names                     # Nombres de clases YOLO
```

---

## 🚀 **Guía Rápida**

1. **Preprocesamiento**: Prepara las imágenes para el entrenamiento.
   ```bash
   python src/preprocesamiento.py
   ```

2. **Entrenamiento**: Entrena el modelo con tus datos.
   ```bash
   python src/entrenamiento.py
   ```

3. **Clasificación en Tiempo Real**: Clasifica imágenes usando tu cámara.
   ```bash
   python src/clasificacion_tiempo_real.py
   ```

---

## 🔍 **Detalles del Código**

### **1. Preprocesamiento (`src/preprocesamiento.py`)**

Este script organiza y prepara las imágenes para el modelo.  

#### **Funcionamiento**:

1. **Organización de Imágenes**:
   - Divide las imágenes originales en carpetas `train` y `val` (80%-20%).
   - Crea subcarpetas por categoría en cada división.

   ```python
   def organizar_imagenes(directorio_origen, directorio_destino):
       categorias = [d for d in os.listdir(directorio_origen) if os.path.isdir(os.path.join(directorio_origen, d))]
       for categoria in categorias:
           # Crear carpetas para entrenamiento y validación
           os.makedirs(os.path.join(directorio_destino, 'train', categoria), exist_ok=True)
           os.makedirs(os.path.join(directorio_destino, 'val', categoria), exist_ok=True)
   ```

2. **Redimensionamiento de Imágenes**:
   - Cambia el tamaño de las imágenes a `(224, 224)` para que sean compatibles con ResNet50.
   ```python
   def redimensionar_imagenes(directorio, tamaño=(224, 224)):
       for root, _, files in os.walk(directorio):
           for file in files:
               with Image.open(ruta_imagen) as img:
                   img_redimensionada = img.resize(tamaño, Image.LANCZOS)
                   img_redimensionada.save(ruta_imagen)
   ```

3. **Resultado**:
   - Estructura de directorios:
     ```plaintext
     datos/imagenes_procesadas/
     ├── train/
     │   ├── pintura/
     │   ├── escultura/
     └── val/
         ├── pintura/
         ├── escultura/
     ```

---

### **2. Entrenamiento (`src/entrenamiento.py`)**

Este script entrena un modelo basado en ResNet50.

#### **Funcionamiento**:

1. **Carga de Datos**:
   - Utiliza `ImageDataGenerator` para aplicar aumentos de datos.
   ```python
   train_datagen = ImageDataGenerator(
       rescale=1./255,
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
       horizontal_flip=True)
   ```

2. **Definición del Modelo**:
   - ResNet50 preentrenado, con capas densas para clasificación:
   ```python
   model = tf.keras.Sequential([
       ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
       tf.keras.layers.GlobalAveragePooling2D(),
       tf.keras.layers.Dense(1024, activation='relu'),
       tf.keras.layers.Dropout(0.5),
       tf.keras.layers.Dense(num_classes, activation='softmax')
   ])
   ```

3. **Entrenamiento**:
   - El modelo se entrena durante 20 épocas y se guarda como `modelo_clasificacion_arte.h5`.
   - Resultados guardados como gráficos de precisión y pérdida en `resultados/`.

4. **Visualización**:
   ```python
   plt.plot(history.history['accuracy'], label='Entrenamiento')
   plt.plot(history.history['val_accuracy'], label='Validación')
   ```

---

### **3. Clasificación en Tiempo Real (`src/clasificacion_tiempo_real.py`)**

Este script utiliza YOLO para detectar regiones en imágenes y un modelo entrenado para clasificarlas.

#### **Funcionamiento**:

1. **Carga de Recursos**:
   - Modelo preentrenado y archivos YOLO (`yolov3.cfg`, `yolov3.weights`).
   ```python
   model = tf.keras.models.load_model('modelos/modelo_clasificacion_arte.h5')
   net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
   ```

2. **Detección de Objetos**:
   - Utiliza YOLO para identificar regiones relevantes.
   ```python
   blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
   net.setInput(blob)
   outputs = net.forward(output_layers)
   ```

3. **Clasificación**:
   - Clasifica cada región detectada utilizando el modelo entrenado.
   ```python
   img = cv2.resize(roi, (224, 224))
   img = np.expand_dims(img, 0)
   prediccion = model.predict(img)
   ```

4. **Interfaz Visual**:
   - Muestra los resultados en tiempo real, incluyendo la clase y la confianza.
   ```python
   cv2.putText(frame, f"{clase_predicha} ({confianza:.2f})", (x, y - 10), ...)
   ```

---
¡Claro! Aquí tienes una explicación más detallada de la sección **Modelo y Técnicas Utilizadas**:

---

## 📊 **Modelo y Técnicas Utilizadas**

### **1. ResNet50**  
**ResNet50** (Residual Network con 50 capas) es una arquitectura de red neuronal convolucional ampliamente utilizada para tareas de clasificación de imágenes. Es conocida por abordar el problema del "degradado" en redes profundas mediante el uso de conexiones residuales (skip connections).  
- **Por qué ResNet50**:  
  - Ofrece un balance ideal entre precisión y eficiencia computacional.  
  - Al estar preentrenada en el conjunto de datos **ImageNet**, proporciona un buen punto de partida para la clasificación de imágenes artísticas.  
- **Implementación**:  
  - Se usa ResNet50 como extractor de características.  
  - Las capas finales de clasificación se reemplazan por:
    - Una capa de promedio global (`GlobalAveragePooling2D`) para reducir dimensionalidad.  
    - Una capa densa con 1024 neuronas y activación `relu`.  
    - Una capa de salida con activación `softmax` para categorizar entre las clases de arte.  
- **Ventaja**:  
  - Reduce el tiempo de entrenamiento y mejora la precisión en tareas especializadas al utilizar un modelo preentrenado.

---

### **2. YOLOv3**  
**YOLOv3** (You Only Look Once, versión 3) es un modelo de detección de objetos en tiempo real. Es rápido y preciso, diseñado para detectar múltiples objetos en una sola pasada de la imagen.  
- **Rol en el proyecto**:  
  - Identificar regiones de interés (ROIs) en las imágenes en tiempo real (por ejemplo, esculturas en una habitación o cuadros en una pared).  
  - Las regiones detectadas son enviadas al modelo de clasificación (ResNet50) para su categorización.  
- **Características principales de YOLOv3**:  
  - Divide la imagen en una cuadrícula y predice cajas delimitadoras junto con probabilidades para cada celda.  
  - Compatible con detección de objetos a múltiples escalas.  
- **Ventaja**:  
  - Detecta regiones relevantes en la imagen de forma eficiente, permitiendo clasificar solo las áreas relevantes en lugar de analizar toda la imagen.  

---

### **3. Aumentos de Datos**  
Los **aumentos de datos** son técnicas que generan versiones modificadas de las imágenes originales para incrementar el tamaño y la diversidad del conjunto de datos de entrenamiento. Esto ayuda a reducir el sobreajuste y mejora la capacidad generalizadora del modelo.  
- **Técnicas utilizadas**:  
  - **Rotación**: Rotaciones aleatorias dentro de un rango definido.  
  - **Traslación**: Cambios en la posición horizontal y vertical.  
  - **Escalado**: Ajustes de tamaño que simulan acercamientos o alejamientos.  
  - **Volteo horizontal**: Reflejar imágenes para añadir simetría.  
- **Ventaja**:  
  - Simula escenarios reales y mejora la robustez del modelo frente a variaciones en los datos.

---

### **4. Entrenamiento Transferido**  
El **entrenamiento transferido** utiliza un modelo preentrenado como base, reutilizando las características aprendidas en un conjunto de datos general (por ejemplo, **ImageNet**) para resolver un problema específico (clasificación de arte).  
- **Implementación en este proyecto**:  
  - La parte convolucional de ResNet50 (capas convolucionales y conexiones residuales) se congela inicialmente para mantener las características generales.  
  - Solo las capas superiores (capa densa y softmax) se entrenan con datos de arte.  
  - En etapas avanzadas, se ajustan las capas profundas (fine-tuning) para optimizar la clasificación de imágenes artísticas.  
- **Ventaja**:  
  - Reduce drásticamente los tiempos de entrenamiento.  
  - Aprovecha características preaprendidas, mejorando la precisión, especialmente cuando los datos disponibles son limitados.

---

Con esta combinación, el sistema logra una detección y clasificación eficiente, robusta y precisa, ideal para aplicaciones en tiempo real. 🎨
---

## 🤝 **Contribuciones**

¡Todas las contribuciones son bienvenidas! Puedes:
- Reportar problemas.
- Mejorar la documentación.
- Proponer nuevas características.

---

## 📜 **Licencia**

Este proyecto está licenciado bajo la [License](LICENSE).

--- 

### ¡Explora, aprende y crea con este proyecto de clasificación de arte en tiempo real! 🎨