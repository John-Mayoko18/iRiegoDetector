import cv2
from ultralytics import YOLO
import time
import numpy as np
from PIL import Image
import requests
from requests.auth import HTTPBasicAuth

# Load YOLOv8 pre-trained model (choose yolov8n.pt for lightweight or yolov8m.pt for better accuracy)
model = YOLO('agua_nivel_detector.pt')

# Credenciales de autenticación
username = "informatica@iriego.es"
password = "325Hjk53"

# URL del recurso
url = "https://camaras.canalparamo.com/api/CAM_REGLA_CANAL_DE_LA_MATA/latest.jpg"
url = "https://camaras.canalparamo.com/api/REGLA_CANAL_DE_LA_MATA/latest.jpg"
#url = "https://camaras.canalparamo.com/api/PARTIDOR_SANTA_MARIA_-_CAMARA_2/latest.jpg"



# Descarga la imagen
#response = requests.get(url)

# Initialize webcam capture
# cap = cv2.VideoCapture(0)  # 0 for default camera

# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# Set camera resolution (optional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
frame_skip = 5  # Process every 5th frame

while True:
    # Realizar la solicitud GET con autenticación básica
    response = requests.get(url, auth=HTTPBasicAuth(username, password))
    # Si la descarga fue exitosa
    #print("response status_code: ", response.status_code)
    if response.status_code == 200:

        # Convierte la respuesta en una imagen
        img = np.array(bytearray(response.content), dtype=np.uint8)

        # Decodifica la imagen en formato BGR
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # Perform YOLO detection
        results = model(frame, stream=True)
        time.sleep(0.5)  # Time pause before next frame

         # Loop through detections and draw bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]

                value = 0
                if label.startswith("nivel") and label !="nivel bajo de cero":  # Ensure it's the correct label
                    last_three_chars = label[-3:]  # Get last 3 characters
                    
                    try:
                        value = float(last_three_chars)  # Convert to float
                        print(f"Extracted value as float: {value}")
                        
                        # Calculation
                        result = 9.9 -  value  # To be checked with Andi
                        area = (x2 - x1) * (y2 - y1)  # Example calculation: bbox area

                        print(f"Possible cantidad de agua a enviar: {result}")
                        print(f"Detected label value: {value}, Bounding box area: {area}")

                    except ValueError:
                        print(f"Error: Could not convert '{last_three_chars}' to float")

                    print(f"Detected label: {label}, Extracted value: {last_three_chars}")

                elif label == "nivel bajo de cero":
                    value = 100
                    print(f"Calculation result: {result}")

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                # if using python v less than 3.x
                cv2.putText(frame, '{} {:.2f}'.format(label, confidence), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the frame
        cv2.imshow('YOLOv8 Real-Time Detection', frame)

        # Procesa la imagen como lo harías normalmente
        # ...
    else:
        print(f"Error al descargar la imagen: {response.status_code}")
        break

    #ret, frame = cap.read()
    #if not ret:
    #    print("Failed to grab frame.")
     #   break
    #frame_count += 1
    #if frame_count % frame_skip != 0:
    #    continue  # Skip this frame

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
#cap.release()
cv2.destroyAllWindows()