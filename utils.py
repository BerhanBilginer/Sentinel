import cv2
import numpy as np

def draw_boxes(frame: np.ndarray, boxes: list):
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        cv2.rectangle(frame, (int(x1), int(y1)) , (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, str(box['class_id']), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def centroids(frame: np.ndarray, boxes: list):
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        center_x = int((x1 + x2) // 2)
        center_y = int((y1 + y2) // 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red color (BGR format)
    return frame

def draw_kalman_predictions(frame: np.ndarray, predictions: list):
    """Draw Kalman filter predictions on frame"""
    for pred in predictions:
        track_id = pred['track_id']
        cx, cy = pred['center']
        bbox = pred['bbox']  # [cx, cy, w, h]
        age = pred['age']
        
        # Draw predicted center as blue circle
        cv2.circle(frame, (int(cx), int(cy)), 8, (255, 0, 0), -1)  # Blue color
        
        # Draw predicted bounding box as blue rectangle
        w, h = bbox[2], bbox[3]
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color
        
        # Draw track ID
        cv2.putText(frame, f"ID:{track_id}", (int(cx-20), int(cy-15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw age indicator (optional)
        if age > 0:
            cv2.putText(frame, f"Age:{age}", (int(cx-20), int(cy+25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    return frame
    