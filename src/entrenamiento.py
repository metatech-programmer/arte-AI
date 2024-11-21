import tensorflow as tf
from keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from PIL import Image, UnidentifiedImageError
import numpy as np

# Configuración
img_height, img_width = 224, 224
batch_size = 32
epochs_base = 20
epochs_fine_tune = 10
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

# Generadores de datos con augmentación y normalización
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

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

# Modelo base: EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Congelar capas base inicialmente

# Modelo completo
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compilar modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Entrenamiento inicial
history_base = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs_base,
    callbacks=[early_stopping, reduce_lr]
)

# Ajuste fino (fine-tuning)
base_model.trainable = True
for layer in base_model.layers[:100]:  # Congelar las primeras 100 capas
    layer.trainable = False

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs_fine_tune,
    callbacks=[early_stopping, lr_scheduler]
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

graficar_resultados([history_base, history_fine_tune], ['Base', 'Fine-tuning'], 'resultados/resultados_entrenamiento.png')

# Evaluación final: Matriz de confusión
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(validation_generator.class_indices.keys()))
disp.plot(cmap='viridis')
plt.title('Matriz de confusión')
plt.show()

print("Entrenamiento completado. Modelo guardado como 'modelo_clasificacion_arte.h5'")
print("Gráficos de resultados guardados como 'resultados_entrenamiento.png'")
