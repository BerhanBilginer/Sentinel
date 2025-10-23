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

def draw_kalman_predictions(frame: np.ndarray, predictions: list, min_hits: int = 0):
    """Draw Kalman predictions with color-coded track status:
    - Green: actively tracked (time_since_update = 0)
    - Yellow: recently lost (time_since_update < 3)
    - Orange: moderately lost (time_since_update < 8)
    - Red: critically lost (time_since_update >= 8)
    Only draw tracks with hits >= min_hits.
    """
    for pred in predictions:
        if pred.get('hits', 0) < min_hits:
            continue

        track_id = pred['track_id']
        cx, cy = pred['center']
        bbox = pred['bbox']  # [cx, cy, w, h]
        time_since_update = pred.get('time_since_update', 0)

        # Color based on update status (BGR format)
        if time_since_update == 0:
            color = (0, 255, 0)        # bright green - actively tracked
        elif time_since_update < 3:
            color = (0, 255, 255)      # yellow - recently lost
        elif time_since_update < 8:
            color = (0, 165, 255)      # orange - moderately lost
        else:
            color = (0, 0, 255)        # red - critically lost

        # Predicted center
        cv2.circle(frame, (int(cx), int(cy)), 8, color, -1)

        # Predicted bbox
        w, h = bbox[2], bbox[3]
        x1 = int(cx - w/2); y1 = int(cy - h/2)
        x2 = int(cx + w/2); y2 = int(cy + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Track ID
        shown_id = pred.get('display_id', track_id)
        cv2.putText(frame, f"ID:{shown_id}", (int(cx-20), int(cy-15)),
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

    # Track status legend
    cv2.rectangle(frame, (legend_x, y-15), (legend_x+20, y-5), (0, 255, 0), 2)
    cv2.putText(frame, "Active Tracks", (legend_x+25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += line_height
    
    cv2.rectangle(frame, (legend_x, y-15), (legend_x+20, y-5), (0, 255, 255), 2)
    cv2.putText(frame, "Recently Lost", (legend_x+25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += line_height
    
    cv2.rectangle(frame, (legend_x, y-15), (legend_x+20, y-5), (0, 165, 255), 2)
    cv2.putText(frame, "Moderately Lost", (legend_x+25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += line_height
    
    cv2.rectangle(frame, (legend_x, y-15), (legend_x+20, y-5), (0, 0, 255), 2)
    cv2.putText(frame, "Critically Lost", (legend_x+25, y),
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

# --- ID panel & ID-only overlays ---
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
        tid = p.get('display_id', p['track_id']); age = p.get('age', 0); lost = p.get('time_since_update', 0); hits = p.get('hits', 0)
        line = f"ID:{tid:3d}  age:{age:3d}  lost:{lost:2d}  hits:{hits:2d}"
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
        tid = p.get('display_id', p['track_id'])
        cx, cy = int(p['center'][0]), int(p['center'][1])
        cv2.putText(frame, f"{tid}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)  # thick white
        cv2.putText(frame, f"{tid}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)        # black outline
    return frame


def draw_edge_points(frame: np.ndarray, predictions: list, min_hits: int = 0):
    """Draw corner points for tracks with edge tracking enabled."""
    if frame is None or not predictions:
        return frame
        
    for pred in predictions:
        if pred.get('hits', 0) < min_hits:
            continue
            
        corners = pred.get('corners')
        if corners is None:
            continue
            
        track_id = pred['track_id']
        shown_id = pred.get('display_id', track_id)
        time_since_update = pred.get('time_since_update', 0)
        
        # Color based on update status (same as main tracking)
        if time_since_update == 0:
            color = (0, 255, 0)        # bright green - actively tracked
        elif time_since_update < 3:
            color = (0, 255, 255)      # yellow - recently lost
        elif time_since_update < 8:
            color = (0, 165, 255)      # orange - moderately lost
        else:
            color = (0, 0, 255)        # red - critically lost
        
        # Draw corner points
        for i, (x, y) in enumerate(corners):
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
            # Label corners: TL, TR, BR, BL
            labels = ['TL', 'TR', 'BR', 'BL']
            cv2.putText(frame, labels[i], (int(x+5), int(y-5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw lines connecting corners to form quadrilateral
        for i in range(4):
            pt1 = (int(corners[i][0]), int(corners[i][1]))
            pt2 = (int(corners[(i+1)%4][0]), int(corners[(i+1)%4][1]))
            cv2.line(frame, pt1, pt2, color, 1)
        
        # Draw track ID at center
        cx = sum(c[0] for c in corners) / 4
        cy = sum(c[1] for c in corners) / 4
        cv2.putText(frame, f"ID:{shown_id}", (int(cx-15), int(cy)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame
