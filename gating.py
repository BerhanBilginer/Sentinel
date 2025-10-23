# gating_core.py
# -*- coding: utf-8 -*-
"""
MOT Gating (zone gate yok):
  1) Motion gate (Mahalanobis): d^2 = y^T S^{-1} y ≤ τ_motion
     y = z - H x̂^-                         # innovation
     S = H P^- H^T + R                      # innovation covariance (R≈0 mümkün)
  2) IoU gate: IoU(b_pred, b_det) ≥ τ_IoU
  3) Scale/Ratio gate:
       s = w*h, r = w/h
       |log s_det - log s_pred| ≤ τ_s
       |r_det - r_pred|         ≤ τ_r

Kutu formatı: (cx, cy, w, h)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Deque
import numpy as np
import math
from collections import deque


# Yaygın χ^2 eşikleri (dof: ölçüm boyutu)
CHI2_CRITICAL: Dict[int, Dict[float, float]] = {
    2: {0.90: 4.605, 0.95: 5.991, 0.99: 9.210},
    3: {0.90: 6.251, 0.95: 7.815, 0.99: 11.345},
    4: {0.90: 7.779, 0.95: 9.488, 0.99: 13.277},
}

@dataclass
class GatingParams:
    # Motion (Mahalanobis)
    dof: int = 4               # z = [cx,cy,w,h] → 4
    chi2_p: float = 0.95       # 0.90 / 0.95 / 0.99
    tau_motion_override: Optional[float] = None
    # IoU
    tau_iou: float = 0.1
    # Scale / Ratio
    tau_log_s: float = 0.7
    tau_ratio: float = 0.5


# ---------- Geometri ----------
def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = map(float, box)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)

def iou_cxcywh(a: np.ndarray, b: np.ndarray) -> float:
    return iou_xyxy(xywh_to_xyxy(a), xywh_to_xyxy(b))


# ---------- Gate fonksiyonları ----------
def chi2_threshold(dof: int, p: float) -> float:
    if dof in CHI2_CRITICAL and p in CHI2_CRITICAL[dof]:
        return CHI2_CRITICAL[dof][p]
    return 9.488  # varsayılan: dof=4, p=0.95

def motion_gate(y: np.ndarray, S: np.ndarray, tau_motion: float) -> Tuple[bool, float]:
    """
    Motion gate (Mahalanobis):
      d^2 = y^T S^{-1} y  ≤  τ_motion
    """
    d2 = float(y.T @ np.linalg.inv(S) @ y)
    return (d2 <= tau_motion), d2

def iou_gate(b_pred: np.ndarray, b_det: np.ndarray, tau_iou: float) -> Tuple[bool, float]:
    """
    IoU gate:
      IoU(b_pred, b_det) ≥ τ_IoU
    """
    i = iou_cxcywh(b_pred, b_det)
    return (i >= tau_iou), i

def scale_ratio_gate(b_pred: np.ndarray, b_det: np.ndarray,
                     tau_log_s: float, tau_ratio: float) -> Tuple[bool, float, float]:
    """
    Scale/ratio gate:
      s = w*h, r = w/h
      |log s_det - log s_pred| ≤ τ_s
      |r_det - r_pred|         ≤ τ_r
    """
    wp, hp = float(b_pred[2]), float(b_pred[3])
    wd, hd = float(b_det[2]), float(b_det[3])
    eps = 1e-12
    sp, sd = max(eps, wp*hp), max(eps, wd*hd)
    rp, rd = wp / max(eps, hp), wd / max(eps, hd)
    d_log_s = abs(math.log(sd) - math.log(sp))
    d_ratio = abs(rd - rp)
    ok = (d_log_s <= tau_log_s) and (d_ratio <= tau_ratio)
    return ok, d_log_s, d_ratio


# ---------- Katmanlı gating (zone yok) ----------
def layered_gates_for_pair(
    x_pred: np.ndarray,   # x̂_k^-  (örn. [cx,cy,vx,vy,w,h])
    P_pred: np.ndarray,   # P_k^-   (6x6)
    H: np.ndarray,        # ölçüm matrisi (4x6)
    R: np.ndarray,        # ölçüm kovaryansı (4x4) — YOLO referanssa R≈0 al
    z_det: np.ndarray,    # detection: [cx,cy,w,h]
    params: GatingParams
) -> Tuple[bool, dict]:
    """
    Tek (track_pred, det) çifti için katmanlı gating:
      Sıra: motion → IoU → scale/ratio
    Dönüş: passed (bool), info (ara metrikler)
    """
    info = {}

    # Motion: y = z - H x̂^- ,  S = H P^- H^T + R
    y = z_det - (H @ x_pred)
    S = H @ P_pred @ H.T + R
    info["y"], info["S"] = y, S

    tau_motion = params.tau_motion_override or chi2_threshold(params.dof, params.chi2_p)
    ok_motion, d2 = motion_gate(y, S, tau_motion)
    info["d2"], info["tau_motion"] = d2, tau_motion
    if not ok_motion:
        return False, info

    # IoU
    b_pred = x_pred[[0, 1, 4, 5]]  # [cx,cy,w,h]
    ok_iou, i = iou_gate(b_pred, z_det, params.tau_iou)
    info["iou"], info["tau_iou"] = i, params.tau_iou
    if not ok_iou:
        return False, info

    # Scale/Ratio
    ok_sr, d_log_s, d_ratio = scale_ratio_gate(b_pred, z_det, params.tau_log_s, params.tau_ratio)
    info["d_log_s"], info["d_ratio"] = d_log_s, d_ratio
    info["tau_log_s"], info["tau_ratio"] = params.tau_log_s, params.tau_ratio
    if not ok_sr:
        return False, info

    return True, info


def build_gate_matrix(
    tracks_pred: List[np.ndarray],   # her track için x̂_k^- (6,)
    covs_pred:   List[np.ndarray],   # her track için P_k^- (6x6)
    detections:  List[np.ndarray],   # her detection için z (4,)
    H: np.ndarray,
    R: np.ndarray,
    params: GatingParams
) -> np.ndarray:
    """
    gate[i, j] = True  ⇔  i. track ile j. detection eşleşme için aday KALIR.
    """
    T, D = len(tracks_pred), len(detections)
    gate = np.zeros((T, D), dtype=bool)
    for i in range(T):
        x_pred, P_pred = tracks_pred[i], covs_pred[i]
        for j in range(D):
            z_det = detections[j]
            ok, _ = layered_gates_for_pair(x_pred, P_pred, H, R, z_det, params)
            gate[i, j] = ok
    return gate


@dataclass
class HistoryParams:
    K_boxes: int = 8          # kaç geçmiş kutu
    tau_hist_iou: float = 0.25
    tau_hist_cos: float = 0.70
    use_or_logic: bool = True  # True: (IoU>=thr) OR (cos>=thr); False: AND

def _cosine_max_to_bank(det_feat: Optional[np.ndarray], bank: Deque[np.ndarray]) -> float:
    if det_feat is None or bank is None or len(bank) == 0:
        return -1.0
    det = det_feat.astype(np.float32)
    best = -1.0
    for f in bank:
        if f is None:
            continue
        denom = max(1e-12, np.linalg.norm(det) * np.linalg.norm(f))
        cs = float(np.dot(det, f) / denom)
        if cs > best:
            best = cs
    return best

def _iou_max_to_bbox_hist(det_z: np.ndarray, bbox_hist: Deque[np.ndarray]) -> float:
    if bbox_hist is None or len(bbox_hist) == 0:
        return 0.0
    best = 0.0
    for b in bbox_hist:
        # b ve det_z formatı [cx,cy,w,h]
        i = iou_cxcywh(b, det_z)
        if i > best:
            best = i
    return float(best)

def build_history_mask(
    tracks_bbox_hist: List[Deque[np.ndarray]],     # her track: deque([cx,cy,w,h], ...)
    tracks_feat_bank: List[Deque[np.ndarray]],     # her track: deque(feat, ...)
    dets: List[np.ndarray],                        # her det: z = [cx,cy,w,h]
    det_feats: List[Optional[np.ndarray]],         # her det: feat (veya None)
    hp: HistoryParams,
) -> np.ndarray:
    """
    history_mask[i,j] = True  ⇔  (IoU_hist_max >= τ)  OR/AND  (cos_hist_max >= τ)
    """
    T, D = len(tracks_bbox_hist), len(dets)
    mask = np.zeros((T, D), dtype=bool)
    for i in range(T):
        bbox_hist = tracks_bbox_hist[i]
        feat_bank = tracks_feat_bank[i]
        for j in range(D):
            z = dets[j]
            f = det_feats[j] if det_feats is not None and j < len(det_feats) else None
            iou_h = _iou_max_to_bbox_hist(z, bbox_hist)
            cos_h = _cosine_max_to_bank(f, feat_bank)
            if hp.use_or_logic:
                ok = (iou_h >= hp.tau_hist_iou) or (cos_h >= hp.tau_hist_cos)
            else:
                # daha katı: ikisini de iste
                ok = (iou_h >= hp.tau_hist_iou) and (cos_h >= hp.tau_hist_cos)
            mask[i, j] = ok
    return mask