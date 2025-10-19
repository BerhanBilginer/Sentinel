from detection import detection
from video_source import video_source
from tracker import PersonTracker
from gating import GatingParams
from association import AssocParams
from kalman import KalmanParams
from reid_encoder import ReIDEncoder  # opsiyonel
import cv2
import json
import numpy as np

from utils import (
    draw_boxes, centroids, draw_kalman_predictions,
    draw_gate_connections, draw_association_results, draw_legend,
    draw_id_panel, draw_id_only
)

def print_controls():
    print("\n" + "="*60)
    print("SENTINEL - Enhanced Multi-Object Tracking")
    print("="*60)
    print("q: quit | d: detections | t: tracks | g: gating | a: assoc | l: legend | SPACE: pause")
    print("i: ID Panel | o: ID-Only mode")
    print("="*60 + "\n")

def main():
    with open('detection_conf.json', 'r') as f:
        config = json.load(f)

    detector = detection(config['model_path'])

    video = video_source('test_cropped.mp4')
    if not video.is_opened():
        print("Error: Could not open video source"); return

    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    dt_nominal = 1.0 / float(fps if fps and fps > 1e-3 else 30.0)

    # Very relaxed gating for stable IDs - motion gate disabled
    gating_params = GatingParams(
        dof=4,
        chi2_p=0.99,
        tau_motion_override=1000.0,  # Effectively disable motion gate
        tau_iou=0.01,     # Very low IoU threshold
        tau_log_s=2.0,    # Very high scale tolerance
        tau_ratio=1.5     # Very high ratio tolerance
    )
    assoc_params  = AssocParams(
        w_m0=1.0, w_i0=1.5, w_a0=0.5, alpha=0.3, beta=0.5,
        stage1_conf_thr=0.6, l2_norm=150.0, app_default_cost=1.0,
        gamma_dt=0.2, delta_dt=0.15
    )
    kalman_params = KalmanParams(dt=dt_nominal)

    # ReID (opsiyonel)
    try:
        # kendi ağırlık yolunu ver (yoksa None ile ImageNet pretrain ile de çalışır)
        reid_weights = "weights/osnet_ain_ms_d_c.pth.tar"  # "weights/osnet_ain_x1_0_msmt17.pth"
        reid = ReIDEncoder(model_name="osnet_ain_x1_0", weight_path=reid_weights, half=False)
        print("ReID encoder loaded.")
        reid_debug = True  # Enable to diagnose ID stability issues
    except Exception as e:
        print(f"ReID unavailable ({e}). Continuing without appearance.")
        reid = None
        reid_debug = False

    tracker = PersonTracker(
        max_age=15,  # Moderate deletion (was 30, then 5)
        gating_params=gating_params,
        assoc_params=assoc_params,
        kalman_params=kalman_params,
        reid_encoder=reid,
        reid_momentum=0.6,
        reid_debug=False,  # Disable debug for clean output
        reid_log_interval=10
    )

    # Minimal display: only tracks with IDs
    show_detections = False
    show_tracks = True
    show_gating = False
    show_association = False
    show_legend = False
    show_id_panel = False
    id_only_mode = False
    paused = False

    print_controls()

    # Get allowed classes (supports both 'class_id' and 'class_ids')
    allowed = config.get('class_ids', config.get('class_id', None))
    if isinstance(allowed, int): 
        allowed = [allowed]

    frame_count = 0
    frame = None
    obj = []
    predictions = []
    gate_matrix = None
    assoc_result = None

    last_time_ms = None

    while True:
        if not paused:
            frame = video.get_frame()
            if frame is None:
                print("No frame received, ending..."); break

            # --- Gerçek dt hesapla ---
            cur_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            if last_time_ms is None or cur_ms is None or cur_ms <= 0:
                dt_real = dt_nominal
            else:
                dt_real = max(1e-6, (cur_ms - last_time_ms) / 1000.0)
            last_time_ms = cur_ms

            # YOLO
            results, obj = detector.detect(frame, config['conf'], config['iou'], allowed)

            # Tracker (dt ver)
            tracker.update(obj, frame_bgr=frame, dt=dt_real)

            predictions = tracker.get_predictions()
            gate_matrix = tracker.get_gate_info()
            assoc_result = tracker.get_assoc_info()

            frame_count += 1

        vis_frame = frame.copy() if frame is not None else None
        if vis_frame is None:
            break

        if id_only_mode:
            vis_frame = draw_id_only(vis_frame, predictions)
        else:
            if show_detections:
                vis_frame = draw_boxes(vis_frame, obj, color=(0, 255, 0), thickness=2)
                vis_frame = centroids(vis_frame, obj, color=(0, 0, 255), radius=5)
            if show_tracks:
                vis_frame = draw_kalman_predictions(vis_frame, predictions)
            if show_gating and predictions and obj:
                vis_frame = draw_gate_connections(vis_frame, predictions, obj, gate_matrix)
            if show_association and predictions and obj:
                vis_frame = draw_association_results(vis_frame, predictions, obj, assoc_result)
            if show_legend:
                vis_frame = draw_legend(vis_frame, show_gating=show_gating, show_association=show_association)
            if show_id_panel:
                vis_frame = draw_id_panel(vis_frame, predictions)

        status_text = f"Frame:{frame_count} | Tracks:{len(predictions)} | Dets:{len(obj)} | dt:{dt_real:.2f}s"
        cv2.putText(vis_frame, status_text, (10, vis_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Sentinel - Enhanced MOT', vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('d'): show_detections = not show_detections; print(f"Detections: {'ON' if show_detections else 'OFF'}")
        elif key == ord('t'): show_tracks = not show_tracks; print(f"Tracks: {'ON' if show_tracks else 'OFF'}")
        elif key == ord('g'): show_gating = not show_gating; print(f"Gating: {'ON' if show_gating else 'OFF'}")
        elif key == ord('a'): show_association = not show_association; print(f"Association: {'ON' if show_association else 'OFF'}")
        elif key == ord('l'): show_legend = not show_legend; print(f"Legend: {'ON' if show_legend else 'OFF'}")
        elif key == ord('i'): show_id_panel = not show_id_panel; print(f"ID Panel: {'ON' if show_id_panel else 'OFF'}")
        elif key == ord('o'): id_only_mode = not id_only_mode; print(f"ID-Only: {'ON' if id_only_mode else 'OFF'}")
        elif key == ord(' '): paused = not paused; print('PAUSED' if paused else 'RESUMED')

    video.release()
    cv2.destroyAllWindows()
    print("\nSentinel shutdown complete.")

if __name__ == "__main__":
    main()
