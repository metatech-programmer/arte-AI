import cv2
import tensorflow as tf
import numpy as np
import os

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelos/modelo_clasificacion_arte.h5')

# Obtener nombres de clases
train_dir = 'datos/imagenes_procesadas/train'
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# Configuración de YOLO
yolo_weights = 'yolov3.weights'
yolo_config = 'yolov3.cfg'

net = cv2.dnn.readNet(yolo_weights, yolo_config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Función para procesar cuadros en tiempo real
def procesar_frame(frame):
    height, width, _ = frame.shape

    # Detección de objetos con YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Umbral de confianza
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = center_x - w // 2, center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maxima Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)  # Evitar coordenadas negativas
        roi = frame[y:y+h, x:x+w]

        # Clasificación con el modelo de arte
        try:
            roi_resized = cv2.resize(roi, (224, 224))
            roi_normalized = roi_resized / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=0)

            prediccion = model.predict(roi_expanded, verbose=0)
            clase_predicha = class_names[np.argmax(prediccion)]
            confianza = np.max(prediccion)

            # Dibujar cuadro y texto
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{clase_predicha} ({confianza:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error procesando ROI: {e}")

    return frame

# Loop principal
print("Presiona 'q' para salir...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar frame actual
    frame_procesado = procesar_frame(frame)

    # Mostrar el resultado
    cv2.imshow('Clasificador de Arte en Tiempo Real', frame_procesado)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
