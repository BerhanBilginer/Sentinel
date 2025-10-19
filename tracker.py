"""
PersonTracker with variable-dt Kalman, dt-aware gating and association
+ TorchReID appearance (opsiyonel)
+ ReID terminal logları
"""
import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy

from kalman import KalmanBBox, KalmanParams
from gating import GatingParams, build_gate_matrix
from association import (
    TrackPred, Detection, AssocParams, association_multistage
)

def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return (x / n).astype(np.float32)

class PersonTracker:
    def __init__(
        self,
        max_age: int = 30,
        max_distance: float = None,
        gating_params: Optional[GatingParams] = None,
        assoc_params: Optional[AssocParams] = None,
        kalman_params: Optional[KalmanParams] = None,
        reid_encoder=None,
        reid_momentum: float = 0.6,
        reid_debug: bool = True,
        reid_log_interval: int = 10
    ):
        self.kalman_params = kalman_params or KalmanParams()
        self.gating_params = gating_params or GatingParams(dof=4, chi2_p=0.95, tau_iou=0.15, tau_log_s=0.7, tau_ratio=0.5)
        self.assoc_params  = assoc_params  or AssocParams(
            w_m0=1.0, w_i0=1.5, w_a0=0.5, alpha=0.3, beta=0.5,
            stage1_conf_thr=0.6, l2_norm=150.0, app_default_cost=1.0,
            gamma_dt=0.2, delta_dt=0.15
        )

        self.tracks: Dict[int, KalmanBBox] = {}
        self.track_ages: Dict[int, int] = {}
        self.track_time_since_update: Dict[int, int] = {}
        self.track_feats: Dict[int, np.ndarray] = {}
        self.next_id = 0
        self.max_age = max_age

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.R = np.eye(4) * self.kalman_params.r_epsilon

        self.last_gate_matrix: Optional[np.ndarray] = None
        self.last_assoc_result: Optional[Dict] = None

        self.reid = reid_encoder
        self.reid_momentum = float(reid_momentum)
        self.reid_debug = bool(reid_debug)
        self.reid_log_interval = int(reid_log_interval)
        self._frame_idx = 0

    def bbox_to_measurement(self, box: Dict) -> np.ndarray:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h], dtype=float)

    def bbox_to_centroid_and_size(self, box):  # legacy alias
        return self.bbox_to_measurement(box)

    def update(self, detections: List[Dict], frame_bgr=None, dt: float = None):
        """
        dt: bu frame ile bir önceki işlenen frame arasındaki gerçek süre (saniye).
        """
        # 1) Predict (dt ile)
        self._predict_all(dt=dt)
        self._frame_idx += 1

        if not detections:
            self._age_tracks()
            self._remove_old_tracks()
            self.last_gate_matrix = None
            self.last_assoc_result = None
            return

        # 2) Tracks → TrackPred
        track_ids = list(self.tracks.keys())
        tracks_pred: List[TrackPred] = []
        for tid in track_ids:
            kalman = self.tracks[tid]
            tracks_pred.append(TrackPred(
                track_id=tid,
                x_pred=kalman.x.copy(),
                P_pred=kalman.P.copy(),
                feat=(self.track_feats.get(tid, None)),
                age=self.track_ages[tid],
                time_since_update=self.track_time_since_update[tid]
            ))

        # 3) Detections (ReID features)
        dets: List[Detection] = []
        det_feats = None
        if self.reid is not None and frame_bgr is not None and len(detections) > 0:
            det_feats = self.reid.encode_from_frame(frame_bgr, detections)
            if det_feats is not None and len(det_feats) == len(detections):
                det_feats = det_feats.astype(np.float32)

        for idx, det in enumerate(detections):
            z = self.bbox_to_measurement(det)
            conf = float(det.get('conf', 1.0))
            feat = None
            if det_feats is not None and idx < det_feats.shape[0]:
                feat = _l2norm(det_feats[idx])
            dets.append(Detection(z=z, conf=conf, feat=feat))

        # 4) Gating — dt'ye göre IoU eşiğini esnet
        if tracks_pred:
            gp = deepcopy(self.gating_params)
            if dt is not None:
                base = self.gating_params.tau_iou
                k = 0.15  # esneme katsayısı
                gp.tau_iou = max(0.01, float(base) * float(np.exp(-k * float(dt))))
            tracks_x = [tp.x_pred for tp in tracks_pred]
            tracks_P = [tp.P_pred for tp in tracks_pred]
            dets_z = [d.z for d in dets]
            gate_matrix = build_gate_matrix(tracks_x, tracks_P, dets_z, self.H, self.R, gp)
            self.last_gate_matrix = gate_matrix
        else:
            gate_matrix = None
            self.last_gate_matrix = None

        # 5) Association — dt’yi aktar
        if tracks_pred:
            assoc_result = association_multistage(
                tracks_pred, dets, self.H, self.R, gate_matrix,
                self.assoc_params, use_mahalanobis=True, dt=(0.0 if dt is None else float(dt))
            )
            self.last_assoc_result = assoc_result

            # ReID DEBUG LOGS (sadece reid varsa)
            if self.reid_debug and (self.reid is not None) and (self._frame_idx % self.reid_log_interval == 0):
                num_det = len(dets); num_det_feat = sum(1 for d in dets if d.feat is not None)
                num_trk = len(tracks_pred); num_trk_feat = sum(1 for t in tracks_pred if t.feat is not None)
                print(f"[ReID] frame={self._frame_idx} dt={dt:.3f} reid_active=True "
                      f"det_feat={num_det_feat}/{num_det} track_feat={num_trk_feat}/{num_trk}")
                for (track_idx, det_idx) in assoc_result.get('matches', []):
                    tr = tracks_pred[track_idx]; de = dets[det_idx]
                    cos_sim = None
                    if tr.feat is not None and de.feat is not None:
                        cos_sim = float(np.clip(np.dot(tr.feat, de.feat), -1.0, 1.0))
                    print(f"  match: track_id={tr.track_id} <- det#{det_idx} conf={de.conf:.2f} "
                          f"cos={('NA' if cos_sim is None else f'{cos_sim:.3f}')}")

                if assoc_result.get('unmatched_tracks'):
                    print(f"  unmatched_tracks: {[tracks_pred[i].track_id for i in assoc_result['unmatched_tracks']]}")
                if assoc_result.get('unmatched_dets'):
                    print(f"  unmatched_dets: {assoc_result['unmatched_dets']}")

            # 6) Matches → update state & feat EMA
            for (track_idx, det_idx) in assoc_result['matches']:
                tid = track_ids[track_idx]
                z = dets[det_idx].z
                self.tracks[tid].update(z)
                self.track_ages[tid] += 1
                self.track_time_since_update[tid] = 0
                df = dets[det_idx].feat
                if df is not None:
                    prev = self.track_feats.get(tid, df)
                    newf = _l2norm(self.reid_momentum * prev + (1.0 - self.reid_momentum) * df)
                    self.track_feats[tid] = newf

            # 7) Unmatched → age
            for track_idx in assoc_result['unmatched_tracks']:
                tid = track_ids[track_idx]
                self.track_time_since_update[tid] += 1
                self.track_ages[tid] += 1

            # 8) Unmatched detections → new tracks
            for det_idx in assoc_result['unmatched_dets']:
                z = dets[det_idx].z
                tid_new = self._create_new_track(z)
                df = dets[det_idx].feat
                if df is not None:
                    self.track_feats[tid_new] = _l2norm(df)
        else:
            # hiç track yoksa tüm detlerden başlat
            for det in dets:
                tid_new = self._create_new_track(det.z)
                if det.feat is not None:
                    self.track_feats[tid_new] = _l2norm(det.feat)
            self.last_assoc_result = None

        self._remove_old_tracks()

    def _predict_all(self, dt: float = None):
        for track in self.tracks.values():
            track.predict(dt=dt)

    def _create_new_track(self, z: np.ndarray) -> int:
        kalman = KalmanBBox(self.kalman_params)
        kalman.initialize(z)
        tid = self.next_id
        self.tracks[tid] = kalman
        self.track_ages[tid] = 1
        self.track_time_since_update[tid] = 0
        self.next_id += 1
        return tid

    def _age_tracks(self):
        for track_id in self.tracks:
            self.track_ages[track_id] += 1
            self.track_time_since_update[track_id] += 1

    def _remove_old_tracks(self):
        to_remove = [tid for tid, tsu in self.track_time_since_update.items() if tsu > self.max_age]
        for tid in to_remove:
            self.tracks.pop(tid, None)
            self.track_ages.pop(tid, None)
            self.track_time_since_update.pop(tid, None)
            self.track_feats.pop(tid, None)

    def get_predictions(self) -> List[Dict]:
        predictions = []
        for track_id in sorted(self.tracks.keys()):
            kalman = self.tracks[track_id]
            cx, cy = kalman.center()
            bbox = kalman.bbox()
            predictions.append({
                'track_id': track_id,
                'center': (cx, cy),
                'bbox': bbox,
                'age': self.track_ages[track_id],
                'time_since_update': self.track_time_since_update[track_id]
            })
        return predictions

    def get_gate_info(self) -> Optional[np.ndarray]:
        return self.last_gate_matrix

    def get_assoc_info(self) -> Optional[Dict]:
        return self.last_assoc_result
