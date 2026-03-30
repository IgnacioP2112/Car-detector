from ultralytics import YOLO

class CarDetector:
    def __init__(self, vehicle_classes, confidence_threshold):
        # Carga el modelo preentrenado. 
        self.model = YOLO("yolov8n.pt")
        self.vehicle_classes = vehicle_classes
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        Corre el modelo sobre el frame.
        Devuelve una lista de dicts con bbox, confianza y clase.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)

            # Filtramos solo vehículos con suficiente confianza
            if class_id in self.vehicle_classes and confidence >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "class_id": class_id
                })

        return detections

    def draw_detections(self, frame, detections):
        import cv2

        class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{class_names[det["class_id"]]} {det["confidence"]:.2f}'

            # Rectángulo verde
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Etiqueta con fondo negro para que se lea bien
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 10, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame