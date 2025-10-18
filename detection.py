from ultralytics import YOLO
import cv2
import numpy as np

class detection:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, conf: float, iou: float, allowed_classes: list = None):
        
        obj = []
        
        results = self.model.predict(frame, conf=conf, iou=iou)
        
        # Check if any detections were found
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return results, obj
        
        class_ids = results[0].boxes.cls
        # Extract xyxy coordinates (x1, y1, x2, y2)
        x1 = results[0].boxes.xyxy[:, 0]
        y1 = results[0].boxes.xyxy[:, 1]
        x2 = results[0].boxes.xyxy[:, 2]
        y2 = results[0].boxes.xyxy[:, 3]

        for i in range(len(class_ids)):
            class_id = int(class_ids[i])
            
            # Filter by allowed classes if specified
            if allowed_classes is not None and class_id not in allowed_classes:
                continue
                
            obj.append({
                "class_id": class_id,
                "x1": float(x1[i]),
                "y1": float(y1[i]),
                "x2": float(x2[i]),
                "y2": float(y2[i])
            })

        return results, obj

    def xywh_to_xyxy(self, box: np.ndarray):
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        return [x1, y1, x2, y2]

