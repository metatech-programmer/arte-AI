import cv2
import tensorflow as tf
import numpy as np
import os

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelos/modelo_clasificacion_arte.h5')

# Obtener nombres de clases
train_dir = 'datos/imagenes_procesadas/train'
class_names = sorted(os.listdir(train_dir))

# Cargar YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Inicializar la cámara
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Detección de objetos
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Umbral de confianza
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas del cuadro delimitador
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maxima Suppression para eliminar cuadros superpuestos
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Clasificar cada objeto detectado
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # Extraer la región de interés
            roi = frame[y:y+h, x:x+w]
            img = cv2.resize(roi, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, 0)

            # Realizar la predicción
            prediccion = model.predict(img)
            clase_predicha = class_names[np.argmax(prediccion[0])]
            confianza = np.max(prediccion[0])

            # Mostrar el resultado en la imagen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{clase_predicha} ({confianza:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar la imagen
    cv2.imshow('Clasificador de Arte en Tiempo Real', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()