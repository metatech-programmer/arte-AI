import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os
from PIL import Image, UnidentifiedImageError

# Configuración
img_height, img_width = 224, 224
batch_size = 32
epochs = 20
base_learning_rate = 0.0001

# Verificar imágenes y eliminar las corruptas
def eliminar_imagenes_corruptas(directorio):
    for root, _, files in os.walk(directorio):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                img = Image.open(filepath)
                img.verify()
            except (UnidentifiedImageError, IOError):
                print(f"Eliminando imagen corrupta: {filepath}")
                os.remove(filepath)

eliminar_imagenes_corruptas('datos/imagenes_procesadas/train')
eliminar_imagenes_corruptas('datos/imagenes_procesadas/val')

# Generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'datos/imagenes_procesadas/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'datos/imagenes_procesadas/val',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)

# Modelo base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Congelar capas base inicialmente

# Modelo completo
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilar modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento inicial
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Ajuste fino (fine-tuning)
base_model.trainable = True
for layer in base_model.layers[:100]:  # Congelar las primeras 100 capas
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10
)

# Guardar modelo
model.save('modelos/modelo_clasificacion_arte.h5')

# Graficar resultados
def graficar_resultados(histories, labels, filename):
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(histories):
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label=f'{labels[i]} - Entrenamiento')
        plt.plot(history.history['val_accuracy'], label=f'{labels[i]} - Validación')
        plt.title('Precisión del modelo')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label=f'{labels[i]} - Entrenamiento')
        plt.plot(history.history['val_loss'], label=f'{labels[i]} - Validación')
        plt.title('Pérdida del modelo')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

graficar_resultados([history, history_fine_tune], ['Base', 'Fine-tuning'], 'resultados/resultados_entrenamiento.png')

print("Entrenamiento completado. Modelo guardado como 'modelo_clasificacion_arte.h5'")
print("Gráficos de resultados guardados como 'resultados_entrenamiento.png'")
