from ultralytics import YOLO
import numpy as np

class detection:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, conf: float, iou: float, allowed_classes: list = None):
        obj = []
        results = self.model.predict(frame, conf=conf, iou=iou, verbose=False)
        if results is None or len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return results, obj

        b = results[0].boxes
        xyxy = b.xyxy; cls = b.cls; cf = b.conf
        # Pull to CPU if tensors
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
            cls = cls.cpu().numpy().astype(int)
            cf = cf.cpu().numpy()

        for i in range(len(cls)):
            cid = int(cls[i])
            if allowed_classes is not None and cid not in allowed_classes:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i][:4])
            obj.append({
                "class_id": cid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": float(cf[i])
            })
        return results, obj

    def xywh_to_xyxy(self, box: np.ndarray):
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        return [x1, y1, x2, y2]
