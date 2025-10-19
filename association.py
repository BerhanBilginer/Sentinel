# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = map(float, box)
    x1, y1 = cx - w/2.0, cy - h/2.0
    x2, y2 = cx + w/2.0, cy + h/2.0
    return np.array([x1, y1, x2, y2], dtype=float)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)

def iou_cxcywh(a: np.ndarray, b: np.ndarray) -> float:
    return iou_xyxy(xywh_to_xyxy(a), xywh_to_xyxy(b))

@dataclass
class TrackPred:
    track_id: int
    x_pred: np.ndarray
    P_pred: np.ndarray
    feat: Optional[np.ndarray] = None
    age: int = 0
    time_since_update: int = 0

@dataclass
class Detection:
    z: np.ndarray
    conf: float
    feat: Optional[np.ndarray] = None

@dataclass
class AssocWeights:
    w_motion: float = 1.0
    w_iou: float    = 1.0
    w_app: float    = 1.0

@dataclass
class AssocParams:
    w_m0: float = 1.0
    w_i0: float = 1.0
    w_a0: float = 1.0
    alpha: float = 0.3
    beta:  float = 0.5
    stage1_conf_thr: float = 0.6
    stage2_conf_thr: float = 0.0
    l2_norm: float = 100.0
    app_default_cost: float = 1.0
    # --- dt dinamikliği için ek parametreler ---
    gamma_dt: float = 0.2   # motion zayıflatma (dt ile böl)
    delta_dt: float = 0.15  # appearance güçlendirme (log(1+dt) ile çarp)

def motion_cost_mahalanobis(tracks: List[TrackPred], dets: List[Detection], H: np.ndarray, R: np.ndarray) -> np.ndarray:
    T, D = len(tracks), len(dets)
    C = np.zeros((T, D), dtype=float)
    for i, tr in enumerate(tracks):
        S = H @ tr.P_pred @ H.T + R
        for j, det in enumerate(dets):
            y = det.z - (H @ tr.x_pred)
            sol = np.linalg.solve(S, y)  # inv yerine solve
            C[i, j] = float(y.T @ sol)
    return C

def motion_cost_l2_center(tracks: List[TrackPred], dets: List[Detection], norm: float = 100.0) -> np.ndarray:
    T, D = len(tracks), len(dets)
    C = np.zeros((T, D), dtype=float)
    for i, tr in enumerate(tracks):
        cp = tr.x_pred[:2]
        for j, det in enumerate(dets):
            cd = det.z[:2]
            C[i, j] = float(np.linalg.norm(cp - cd)) / max(1e-12, norm)
    return C

def iou_cost(tracks: List[TrackPred], dets: List[Detection]) -> np.ndarray:
    T, D = len(tracks), len(dets)
    C = np.zeros((T, D), dtype=float)
    for i, tr in enumerate(tracks):
        b_pred = tr.x_pred[[0,1,4,5]]
        for j, det in enumerate(dets):
            C[i, j] = 1.0 - iou_cxcywh(b_pred, det.z)
    return C

def cosine_cost(tracks: List[TrackPred], dets: List[Detection], default_cost: float = 1.0) -> np.ndarray:
    T, D = len(tracks), len(dets)
    C = np.full((T, D), float(default_cost), dtype=float)
    for i, tr in enumerate(tracks):
        fi = tr.feat
        if fi is None:
            continue
        for j, det in enumerate(dets):
            fj = det.feat
            if fj is None:
                continue
            denom = max(1e-12, np.linalg.norm(fi) * np.linalg.norm(fj))
            cos_sim = float(np.dot(fi, fj) / denom)
            C[i, j] = 1.0 - cos_sim
    return C

def dynamic_weights_for_detection(det: Detection, ap: AssocParams, dt: float = 0.0) -> AssocWeights:
    """
    conf düşükse app↑ motion↓; ayrıca dt büyüdükçe motion↓ app↑
    """
    c = float(det.conf)
    wm = ap.w_m0 * (ap.alpha + (1.0 - ap.alpha) * c)
    wi = ap.w_i0
    wa = ap.w_a0 * (ap.beta + (1.0 - ap.beta) * (1.0 - c))
    # --- dt etkisi ---
    wm = wm / (1.0 + ap.gamma_dt * max(0.0, dt))
    wa = wa * (1.0 + ap.delta_dt * np.log1p(max(0.0, dt)))
    return AssocWeights(w_motion=wm, w_iou=wi, w_app=wa)

def fuse_costs_per_det(C_motion: np.ndarray, C_iou: np.ndarray, C_app: np.ndarray,
                       dets: List[Detection], ap: AssocParams, dt: float = 0.0) -> np.ndarray:
    T, D = C_motion.shape
    C_fused = np.zeros((T, D), dtype=float)
    for j in range(D):
        w = dynamic_weights_for_detection(dets[j], ap, dt=dt)
        C_fused[:, j] = w.w_motion * C_motion[:, j] + w.w_iou * C_iou[:, j] + w.w_app * C_app[:, j]
    return C_fused

def apply_gate_mask(C: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return C
    C2 = C.copy()
    C2[~mask] = np.inf
    return C2

def solve_hungarian(C: np.ndarray) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    T, D = C.shape
    if T == 0 or D == 0:
        return [], list(range(T)), list(range(D))
    if not np.isfinite(C).any():
        return [], list(range(T)), list(range(D))
    if _HAS_SCIPY:
        try:
            row_ind, col_ind = linear_sum_assignment(C)
        except ValueError:
            return [], list(range(T)), list(range(D))
    else:
        row_ind, col_ind = [], []
        C_tmp = C.copy()
        while True:
            i, j = np.unravel_index(np.argmin(C_tmp), C_tmp.shape)
            if not np.isfinite(C_tmp[i, j]):
                break
            row_ind.append(i); col_ind.append(j)
            C_tmp[i, :] = np.inf; C_tmp[:, j] = np.inf
            if not np.isfinite(C_tmp).any():
                break
    matches, used_rows, used_cols = [], set(), set()
    for i, j in zip(row_ind, col_ind):
        if np.isfinite(C[i, j]):
            matches.append((int(i), int(j)))
            used_rows.add(int(i)); used_cols.add(int(j))
    unmatched_tracks = [i for i in range(T) if i not in used_rows]
    unmatched_dets   = [j for j in range(D) if j not in used_cols]
    return matches, unmatched_tracks, unmatched_dets

def association_multistage(
    tracks: List[TrackPred], dets: List[Detection],
    H: np.ndarray, R: np.ndarray, gate_mask: Optional[np.ndarray],
    ap: AssocParams, use_mahalanobis: bool = True, dt: float = 0.0
) -> Dict[str, object]:
    """
    dt: saniye cinsinden zaman aralığı (atlamalı frame için büyük olabilir)
    """
    T, D = len(tracks), len(dets)
    result = {
        'matches': [],
        'unmatched_tracks': list(range(T)),
        'unmatched_dets': list(range(D)),
        'stage1': {}, 'stage2': {}, 'stage3': {}
    }

    def subselect(items, idxs): return [items[i] for i in idxs]

    # ---- AŞAMA 1: yüksek conf ----
    det_idx1 = [j for j in result['unmatched_dets'] if dets[j].conf >= ap.stage1_conf_thr]
    tr_idx1  = result['unmatched_tracks']
    tracks1, dets1 = subselect(tracks, tr_idx1), subselect(dets, det_idx1)
    if tracks1 and dets1:
        C_motion = (motion_cost_mahalanobis(tracks1, dets1, H, R)
                    if use_mahalanobis else
                    motion_cost_l2_center(tracks1, dets1, norm=ap.l2_norm))
        C_iou = iou_cost(tracks1, dets1)
        C_app = cosine_cost(tracks1, dets1, default_cost=ap.app_default_cost)
        C = fuse_costs_per_det(C_motion, C_iou, C_app, dets1, ap, dt=dt)
        mask1 = None
        if gate_mask is not None:
            mask1 = gate_mask[np.ix_(tr_idx1, det_idx1)]
            C = apply_gate_mask(C, mask1)
        m1, utr1, ude1 = solve_hungarian(C)
        for (ii, jj) in m1:
            result['matches'].append((tr_idx1[ii], det_idx1[jj]))
        result['unmatched_tracks'] = [tr_idx1[i] for i in utr1]
        result['unmatched_dets']   = [det_idx1[j] for j in ude1] + [j for j in result['unmatched_dets'] if j not in det_idx1]
        result['stage1'] = {'cost': C, 'matches': m1, 'mask': mask1}

    # ---- AŞAMA 2: kalanlar (appearance ağır) ----
    tr_idx2 = result['unmatched_tracks']
    det_idx2 = result['unmatched_dets']
    tracks2, dets2 = subselect(tracks, tr_idx2), subselect(dets, det_idx2)
    if tracks2 and dets2:
        C_motion2 = (motion_cost_mahalanobis(tracks2, dets2, H, R)
                     if use_mahalanobis else
                     motion_cost_l2_center(tracks2, dets2, norm=ap.l2_norm))
        C_iou2 = iou_cost(tracks2, dets2)
        C_app2 = cosine_cost(tracks2, dets2, default_cost=ap.app_default_cost)
        C2 = fuse_costs_per_det(C_motion2, C_iou2, C_app2, dets2, ap, dt=dt)
        mask2 = None
        if gate_mask is not None:
            mask2 = gate_mask[np.ix_(tr_idx2, det_idx2)]
            C2 = apply_gate_mask(C2, mask2)
        m2, utr2, ude2 = solve_hungarian(C2)
        for (ii, jj) in m2:
            result['matches'].append((tr_idx2[ii], det_idx2[jj]))
        result['unmatched_tracks'] = [tr_idx2[i] for i in utr2]
        result['unmatched_dets']   = [det_idx2[j] for j in ude2]
        result['stage2'] = {'cost': C2, 'matches': m2, 'mask': mask2}

    result['stage3'] = {'note': 'long-range ReID/gallery (opsiyonel)'}
    return result
