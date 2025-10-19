import cv2
import numpy as np
from typing import Optional, Dict, List

def draw_boxes(frame: np.ndarray, boxes: list, color=(0, 255, 0), thickness=2, label_prefix=""):
    """Draw detection boxes on frame (default: green)."""
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = f"{label_prefix}{box.get('class_id', '')}"
        if label:
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def centroids(frame: np.ndarray, boxes: list, color=(0, 0, 255), radius=5):
    """Draw centroids of detections (default: red)."""
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        center_x = int((x1 + x2) // 2)
        center_y = int((y1 + y2) // 2)
        cv2.circle(frame, (center_x, center_y), radius, color, -1)
    return frame

def draw_kalman_predictions(frame: np.ndarray, predictions: list):
    """Draw Kalman predictions (blue/orange/gray by staleness)."""
    for pred in predictions:
        track_id = pred['track_id']
        cx, cy = pred['center']
        bbox = pred['bbox']  # [cx, cy, w, h]
        age = pred['age']
        time_since_update = pred.get('time_since_update', 0)

        # Color based on update status
        if time_since_update == 0:
            color = (255, 0, 0)        # blue - recently updated
        elif time_since_update < 5:
            color = (255, 165, 0)      # orange - not recently updated
        else:
            color = (128, 128, 128)    # gray - stale

        # Predicted center
        cv2.circle(frame, (int(cx), int(cy)), 8, color, -1)

        # Predicted bbox
        w, h = bbox[2], bbox[3]
        x1 = int(cx - w/2); y1 = int(cy - h/2)
        x2 = int(cx + w/2); y2 = int(cy + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Track ID
        cv2.putText(frame, f"ID:{track_id}", (int(cx-20), int(cy-15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if time_since_update > 0:
            cv2.putText(frame, f"Lost:{time_since_update}", (int(cx-25), int(cy+25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame

def draw_gate_connections(frame: np.ndarray, predictions: list, detections: list, gate_matrix: Optional[np.ndarray]):
    """
    Draw gating connections:
    - Green thin lines from tracks to detections where gates passed
    - Red 'X' on detections that failed for all tracks
    """
    if gate_matrix is None or len(predictions) == 0 or len(detections) == 0:
        return frame

    n_tracks_gate, n_dets_gate = gate_matrix.shape
    if n_dets_gate != len(detections):
        return frame

    det_centers = []
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
        det_centers.append((cx, cy))

    n_tracks_to_draw = min(len(predictions), n_tracks_gate)
    for i in range(n_tracks_to_draw):
        pred = predictions[i]
        track_cx, track_cy = int(pred['center'][0]), int(pred['center'][1])
        for j, (det_cx, det_cy) in enumerate(det_centers):
            if j < n_dets_gate and gate_matrix[i, j]:
                cv2.line(frame, (track_cx, track_cy), (det_cx, det_cy), (0, 255, 0), 1, cv2.LINE_AA)

    for j in range(min(len(det_centers), n_dets_gate)):
        if not gate_matrix[:, j].any():
            cx, cy = det_centers[j]
            size = 10
            cv2.line(frame, (cx-size, cy-size), (cx+size, cy+size), (0, 0, 255), 2)
            cv2.line(frame, (cx+size, cy-size), (cx-size, cy+size), (0, 0, 255), 2)
    return frame

def draw_association_results(frame: np.ndarray, predictions: list, detections: list, assoc_result: Optional[Dict]):
    """
    Association overlays:
    - Cyan thick lines: matched pairs
    - Yellow circles: unmatched detections (new tracks)
    - Magenta squares: unmatched tracks (lost)
    """
    if assoc_result is None or len(predictions) == 0 or len(detections) == 0:
        return frame

    matches = assoc_result.get('matches', [])
    unmatched_tracks = assoc_result.get('unmatched_tracks', [])
    unmatched_dets = assoc_result.get('unmatched_dets', [])

    det_centers = []
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
        det_centers.append((cx, cy))

    for (track_idx, det_idx) in matches:
        if track_idx < len(predictions) and det_idx < len(det_centers):
            track_cx, track_cy = int(predictions[track_idx]['center'][0]), int(predictions[track_idx]['center'][1])
            det_cx, det_cy = det_centers[det_idx]
            cv2.line(frame, (track_cx, track_cy), (det_cx, det_cy), (255, 255, 0), 3, cv2.LINE_AA)
            mid_x, mid_y = (track_cx + det_cx) // 2, (track_cy + det_cy) // 2
            cv2.circle(frame, (mid_x, mid_y), 5, (255, 255, 0), -1)

    for det_idx in unmatched_dets:
        if det_idx < len(det_centers):
            cx, cy = det_centers[det_idx]
            cv2.circle(frame, (cx, cy), 15, (0, 255, 255), 3)
            cv2.putText(frame, "NEW", (cx-20, cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for track_idx in unmatched_tracks:
        if track_idx < len(predictions):
            cx, cy = int(predictions[track_idx]['center'][0]), int(predictions[track_idx]['center'][1])
            size = 15
            cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (255, 0, 255), 3)
            cv2.putText(frame, "LOST", (cx-25, cy-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    return frame

def draw_legend(frame: np.ndarray, show_gating=True, show_association=True):
    """Legend box for overlay meanings."""
    height, width = frame.shape[:2]
    legend_x, legend_y, line_height = 10, 30, 25
    cv2.putText(frame, "Legend:", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y = legend_y + line_height

    cv2.rectangle(frame, (legend_x, y-15), (legend_x+20, y-5), (0, 255, 0), 2)
    cv2.putText(frame, "YOLO Detections", (legend_x+25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += line_height

    cv2.rectangle(frame, (legend_x, y-15), (legend_x+20, y-5), (255, 0, 0), 2)
    cv2.putText(frame, "Active Tracks", (legend_x+25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += line_height

    if show_gating:
        cv2.line(frame, (legend_x, y-10), (legend_x+20, y-10), (0, 255, 0), 2)
        cv2.putText(frame, "Gate Passed", (legend_x+25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += line_height

    if show_association:
        cv2.line(frame, (legend_x, y-10), (legend_x+20, y-10), (255, 255, 0), 3)
        cv2.putText(frame, "Matched Pair", (legend_x+25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += line_height

        cv2.circle(frame, (legend_x+10, y-10), 8, (0, 255, 255), 2)
        cv2.putText(frame, "New Detection", (legend_x+25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += line_height

        cv2.rectangle(frame, (legend_x, y-15), (legend_x+20, y-5), (255, 0, 255), 2)
        cv2.putText(frame, "Lost Track", (legend_x+25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame

# --- NEW: ID panel & ID-only overlays ---

def draw_id_panel(frame: np.ndarray, predictions: list):
    """Top-left translucent panel listing active track IDs."""
    if frame is None or not predictions:
        return frame
    overlay = frame.copy()
    x, y = 10, 10
    h = min(30 + 22 * len(predictions), 300)
    w = 260
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    alpha = 0.35
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, "Active Tracks", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    yy = y + 45
    for p in predictions:
        tid = p['track_id']; age = p.get('age', 0); lost = p.get('time_since_update', 0)
        line = f"ID:{tid:3d}  age:{age:3d}  lost:{lost:2d}"
        cv2.putText(frame, line, (x + 10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        yy += 20
        if yy > y + h - 5:
            break
    return frame

def draw_id_only(frame: np.ndarray, predictions: list):
    """Scene with only large ID labels on track centers."""
    if frame is None or not predictions:
        return frame
    for p in predictions:
        tid = p['track_id']
        cx, cy = int(p['center'][0]), int(p['center'][1])
        cv2.putText(frame, f"{tid}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)  # thick white
        cv2.putText(frame, f"{tid}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)        # black outline
    return frame
