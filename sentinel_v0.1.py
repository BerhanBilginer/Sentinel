#!/usr/bin/env python3
"""
Offline MOT (Pass-1) — Kalman + Cascaded Matching (Byte-style) + Optional ReID (OSNet-like via ResNet+GeM)

Designed for a single, fixed-camera video. JSON/ETL etc. intentionally omitted.
Outputs a rendered MP4 with tracks and a simple CSV of frame-wise boxes.

Requirements (pip):
  - ultralytics (for YOLOv8/YOLOv11)
  - torch, torchvision, numpy, opencv-python, scipy

Example:
python offline_pass1_tracker.py \
  --video input.mp4 --out out.mp4 \
  --yolo yolov11l.pt --imgsz 1280 \
  --conf_high 0.45 --conf_low 0.05 --det_iou 0.6 \
  --max_age 45 --min_hits 3 \
  --use_reid --reid_backbone r50 --app_thr 0.35
"""

import os
import csv
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from PIL import Image

# -------------------------- Utils --------------------------

def iou_xyxy(a, b):
    """IoU between boxes in [x1,y1,x2,y2]. a: (N,4), b: (M,4) -> (N,M)"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    N, M = a.shape[0], b.shape[0]
    out = np.zeros((N, M), dtype=np.float32)
    if N == 0 or M == 0:
        return out
    for i in range(N):
        ax1, ay1, ax2, ay2 = a[i]
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for j in range(M):
            bx1, by1, bx2, by2 = b[j]
            bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            union = aa + bb - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return out


def xyxy_to_cxcywh(xyxy):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
    return np.array([cx, cy, w, h], dtype=np.float32)


def cxcywh_to_xyxy(bb):
    cx, cy, w, h = bb
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# -------------------------- Kalman (CV) --------------------------
class KalmanCV:
    """Simple constant-velocity Kalman on [cx,cy,w,h,vx,vy,vw,vh]."""
    def __init__(self, dt=1.0/25.0):
        self.dt = dt
        dt = self.dt
        # State transition F, measurement H, covariances Q, R
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i+4] = dt
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0,0] = self.H[1,1] = self.H[2,2] = self.H[3,3] = 1.0
        q_pos, q_vel = 1e-3, 1e-2
        self.Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_vel]).astype(np.float32)
        r_meas = 1e-1
        self.R = np.diag([r_meas, r_meas, r_meas, r_meas]).astype(np.float32)

    def initiate(self, meas):
        # x: [cx, cy, w, h, vx, vy, vw, vh], P: covariance
        x = np.zeros((8, 1), dtype=np.float32)
        x[0:4, 0] = meas.reshape(4)
        P = np.eye(8, dtype=np.float32)
        P[0:4, 0:4] *= 10.0
        P[4:8, 4:8] *= 100.0
        return x, P

    def predict(self, x, P):
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def update(self, x, P, meas):
        z = meas.reshape(4,1).astype(np.float32)
        y = z - (self.H @ x)
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        x = x + K @ y
        I = np.eye(8, dtype=np.float32)
        P = (I - K @ self.H) @ P
        return x, P


# -------------------------- Appearance: ResNet + GeM --------------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.mean(x, dim=(-1, -2)).pow(1.0 / self.p)

class AppearanceModel(nn.Module):
    def __init__(self, backbone="r18", use_gem=True, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        if backbone == "r50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.dim = 2048
        else:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.dim = 512
        self.trunk = nn.Sequential(*list(m.children())[:-2])
        self.pool = GeM().to(self.device) if use_gem else nn.AdaptiveAvgPool2d(1).to(self.device)
        self.bn = nn.BatchNorm1d(self.dim, affine=False)
        self.tx = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def forward(self, frame_bgr, xyxy):
        x1,y1,x2,y2 = list(map(int, xyxy))
        x1,y1 = max(0,x1), max(0,y1)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return torch.zeros(self.dim, device=self.device)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = transforms.functional.to_pil_image(rgb)
        ten = self.tx(pil).unsqueeze(0).to(self.device)
        f = self.trunk(ten)
        f = self.pool(f)
        if not isinstance(self.pool, GeM):
            f = f.squeeze(-1).squeeze(-1)
        f = f.view(-1)
        f = self.bn(f.unsqueeze(0)).view(-1)
        f = F.normalize(f, p=2, dim=0)
        return f


# -------------------------- Appearance: OSNet x1.0 --------------------------
class AppearanceModelOSNet(nn.Module):
    """OSNet x1.0 ReID (via torchreid). Returns L2-normalized 512-d feature.
    Install: pip install torchreid
    """
    def __init__(self, device="cpu", height=256, width=128, half=False):
        super().__init__()
        self.device = torch.device(device)
        self.height = int(height)
        self.width = int(width)
        self.use_half = bool(half) and (self.device.type == "cuda")
        try:
            import torchreid  # lazy import
        except ImportError as e:
            raise ImportError("torchreid is required for OSNet. Install with: pip install torchreid") from e
        self._torchreid = torchreid
        # Build OSNet x1.0 backbone
        self.model = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
        if hasattr(self.model, "classifier"):
            self.model.classifier = nn.Identity()
        self.model.to(self.device).eval()
        if self.use_half:
            self.model.half()
        # transforms (256x128, normalization inside torchreid)
        _, test_tfm = torchreid.data.transforms.build_transforms(height=self.height, width=self.width)
        self.tx = test_tfm
        self.dim = 512

    @torch.no_grad()
    def forward(self, frame_bgr, xyxy):
        x1, y1, x2, y2 = list(map(int, xyxy))
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return torch.zeros(self.dim, device=self.device)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ten = self.tx(pil).unsqueeze(0).to(self.device)
        if self.use_half:
            ten = ten.half()
        feat = self.model(ten).view(-1)
        feat = F.normalize(feat, p=2, dim=0)
        return feat


def cosine_dist(f1: torch.Tensor, f2: torch.Tensor) -> float:
    if f1 is None or f2 is None or f1.numel()==0 or f2.numel()==0:
        return 1.0
    f1 = F.normalize(f1.view(-1), p=2, dim=0)
    f2 = F.normalize(f2.view(-1), p=2, dim=0)
    sim = torch.clamp(F.cosine_similarity(f1, f2, dim=0), -1.0, 1.0).item()
    return 1.0 - sim


# -------------------------- Track --------------------------
@dataclass
class Track:
    tid: int
    x: np.ndarray   # (8,1) state
    P: np.ndarray   # (8,8) cov
    hits: int
    age: int
    time_since_update: int
    feat: torch.Tensor | None
    templates: list  # list[torch.Tensor]
    color: tuple

    def to_xyxy(self):
        bb = self.x[0:4, 0]
        return cxcywh_to_xyxy(bb)


# -------------------------- Tracker Core --------------------------
class MOTTracker:
    def __init__(self, dt, max_age=45, min_hits=3,
                 w_iou=0.5, w_app=0.35, w_motion=0.15,
                 app_thr=0.35, ema_m=0.1, use_reid=False,
                 device="cpu"):
        self.kf = KalmanCV(dt)
        self.max_age = max_age
        self.min_hits = min_hits
        self.w_iou = w_iou
        self.w_app = w_app
        self.w_motion = w_motion
        self.app_thr = app_thr
        self.ema_m = ema_m
        self.use_reid = use_reid
        self.device = device

        self.tracks: list[Track] = []
        self.next_tid = 0

    def _new_color(self):
        return tuple(np.random.randint(60, 230, size=3).tolist())

    def _init_track(self, det):
        meas = xyxy_to_cxcywh(det)
        x, P = self.kf.initiate(meas)
        t = Track(self.next_tid, x, P, hits=1, age=1, time_since_update=0,
                  feat=None, templates=[], color=self._new_color())
        self.tracks.append(t)
        self.next_tid += 1

    def _predict(self):
        for t in self.tracks:
            t.x, t.P = self.kf.predict(t.x, t.P)
            t.age += 1
            t.time_since_update += 1

    def _gating_motion(self, t: Track, dets_xyxy, img_diag):
        # simple center-distance gating in normalized units
        if len(dets_xyxy) == 0:
            return np.array([], dtype=np.int32)
        cx, cy, w, h = t.x[0,0], t.x[1,0], t.x[2,0], t.x[3,0]
        centers = np.array([[ (d[0]+d[2])/2.0, (d[1]+d[3])/2.0 ] for d in dets_xyxy], dtype=np.float32)
        d = np.sqrt((centers[:,0]-cx)**2 + (centers[:,1]-cy)**2)
        thr = 0.35 * img_diag
        keep = np.where(d <= thr)[0]
        return keep

    def _cost_matrix(self, cand_trk_idx, dets_xyxy, iou_mat, frame_bgr, app_model):
        # fuse cost = w_iou*(1-iou) + w_app*cos_dist + w_motion*center_dist_norm
        N = len(cand_trk_idx)
        M = dets_xyxy.shape[0]
        cost = np.ones((N, M), dtype=np.float32) * 1e3
        if N==0 or M==0:
            return cost
        H, W = frame_bgr.shape[:2]
        diag = math.hypot(W, H)
        for rr, ti in enumerate(cand_trk_idx):
            tr = self.tracks[ti]
            # Precompute track appearance template (use newest)
            tr_feat = tr.feat
            tr_xyxy = tr.to_xyxy()
            tr_cx = tr.x[0,0]; tr_cy = tr.x[1,0]
            for j in range(M):
                iou = iou_mat[rr, j]
                iou_c = 1.0 - iou
                # motion term
                dj = dets_xyxy[j]
                dc = math.hypot(((dj[0]+dj[2])/2.0)-tr_cx, ((dj[1]+dj[3])/2.0)-tr_cy) / (diag+1e-6)
                # appearance term
                app_c = 0.0
                if self.use_reid and (app_model is not None):
                    if tr_feat is None or tr_feat.numel()==0:
                        # bootstrap on-demand
                        tr_feat = app_model(frame_bgr, tr_xyxy)
                        tr.feat = tr_feat
                    det_feat = app_model(frame_bgr, dj)
                    app_c = cosine_dist(tr_feat, det_feat)
                w_sum = self.w_iou + self.w_app + self.w_motion
                wi = self.w_iou / w_sum
                wa = self.w_app / w_sum
                wm = self.w_motion / w_sum
                fused = wi*iou_c + wa*app_c + wm*dc
                cost[rr, j] = fused
        return cost

    def _update_matched(self, matches, dets_xyxy, frame_bgr, app_model):
        for rr, jj in matches:
            ti = rr
            det = dets_xyxy[jj]
            meas = xyxy_to_cxcywh(det)
            self.tracks[ti].x, self.tracks[ti].P = self.kf.update(self.tracks[ti].x, self.tracks[ti].P, meas)
            self.tracks[ti].hits += 1
            self.tracks[ti].time_since_update = 0
            # appearance EMA
            if self.use_reid and app_model is not None:
                newf = app_model(frame_bgr, det)
                oldf = self.tracks[ti].feat
                if oldf is None or oldf.numel()==0:
                    mixed = newf
                else:
                    mixed = F.normalize((1.0 - self.ema_m) * oldf + self.ema_m * newf, p=2, dim=0)
                self.tracks[ti].feat = mixed
                # maintain small template bank
                self.tracks[ti].templates.append(newf.detach().clone())
                if len(self.tracks[ti].templates) > 10:
                    self.tracks[ti].templates.pop(0)

    def _prune(self):
        alive = []
        for t in self.tracks:
            if (t.hits >= self.min_hits) or (t.time_since_update <= self.max_age):
                if t.time_since_update <= self.max_age:
                    alive.append(t)
        self.tracks = alive

    def step(self, frame_bgr, dets_high, dets_low, app_model=None):
        """One frame step. dets_* are arrays of [x1,y1,x2,y2]."""
        H, W = frame_bgr.shape[:2]
        diag = math.hypot(W, H)
        # 1) predict
        self._predict()
        # 2) build IoU with ALL dets (concat for convenience), but we will cascade
        all_dets = np.vstack([dets_high, dets_low]) if len(dets_low) else dets_high
        iou_all = iou_xyxy([t.to_xyxy() for t in self.tracks], all_dets)
        # 3) cascade stage-1 (high)
        # active tracks
        active_idx = [i for i,_ in enumerate(self.tracks)]
        # motion gating per track
        keep_lists = [self._gating_motion(self.tracks[i], dets_high, diag) for i in active_idx]
        # build cost only on kept columns
        if len(dets_high):
            cost_stage1 = np.ones((len(active_idx), len(dets_high)), dtype=np.float32) * 1e3
            for rr, ti in enumerate(active_idx):
                cols = keep_lists[rr]
                if cols.size == 0:
                    continue
                # build a sub cost row
                # compute fused cost on selected cols
                # reuse global iou matrix by indexing correct columns
                iou_row = np.array([iou_all[ti, j] for j in range(len(dets_high))], dtype=np.float32)
                # compute cost row
                # temporary create dets subset to compute app/motion
                cost_full = self._cost_matrix([ti], dets_high, iou_row[None, :], frame_bgr, app_model)
                cost_stage1[rr, :] = cost_full[0]
            row_ind, col_ind = linear_sum_assignment(cost_stage1)
            matched_stage1 = []
            unmatched_trk = set(active_idx)
            unmatched_det_high = set(range(len(dets_high)))
            for r, c in zip(row_ind, col_ind):
                cval = cost_stage1[r, c]
                if np.isfinite(cval) and cval < 0.90:  # match_max_cost stage-1
                    matched_stage1.append((active_idx[r], c))
                    unmatched_trk.discard(active_idx[r])
                    unmatched_det_high.discard(c)
            # update matched
            if matched_stage1:
                self._update_matched(matched_stage1, dets_high, frame_bgr, app_model)
        else:
            unmatched_trk = set(active_idx)
            unmatched_det_high = set()
            matched_stage1 = []

        # 4) cascade stage-2 (low) — only unmatched tracks vs low-score dets
        matched_stage2 = []
        if len(dets_low) and len(unmatched_trk):
            # compute IoU submatrix for remaining tracks vs dets_low
            iou_low = np.zeros((len(unmatched_trk), len(dets_low)), dtype=np.float32)
            for rr, ti in enumerate(list(unmatched_trk)):
                for j in range(len(dets_low)):
                    iou_low[rr, j] = iou_all[ti, len(dets_high)+j] if iou_all.size else 0.0
            cost_stage2 = self._cost_matrix(list(unmatched_trk), dets_low, iou_low, frame_bgr, app_model)
            row_ind, col_ind = linear_sum_assignment(cost_stage2)
            unmatched_det_low = set(range(len(dets_low)))
            for r, c in zip(row_ind, col_ind):
                cval = cost_stage2[r, c]
                if np.isfinite(cval) and cval < 0.85:  # slightly stricter
                    matched_stage2.append((list(unmatched_trk)[r], c))
                    unmatched_det_low.discard(c)
            if matched_stage2:
                self._update_matched(matched_stage2, dets_low, frame_bgr, app_model)
        else:
            unmatched_det_low = set()

        # 5) birth new tracks from remaining detections (both stages)
        used_high = {c for (_, c) in matched_stage1}
        used_low  = {c for (_, c) in matched_stage2}
        for i in range(len(dets_high)):
            if i not in used_high:
                self._init_track(dets_high[i])
        for i in range(len(dets_low)):
            if i not in used_low:
                self._init_track(dets_low[i])

        # 6) prune old
        self._prune()

        # return current active tracks
        return [t for t in self.tracks if t.time_since_update == 0]


# -------------------------- Runner --------------------------
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    # Detector
    model = YOLO(args.yolo)

    # Appearance
    app_model = None
    if args.use_reid:
        if args.reid_backbone == "osnet":
            app_model = AppearanceModelOSNet(device=device)
        else:
            app_model = AppearanceModel(backbone=args.reid_backbone, use_gem=not args.no_gem, device=device)

    # Tracker
    tracker = MOTTracker(dt=1.0/float(fps), max_age=args.max_age, min_hits=args.min_hits,
                         w_iou=args.w_iou, w_app=args.w_app, w_motion=args.w_motion,
                         app_thr=args.app_thr, ema_m=args.app_m, use_reid=args.use_reid, device=device)

    # CSV output
    csv_path = Path(args.out).with_suffix("")
    csv_path = Path(str(csv_path) + "_tracks.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame","track_id","x1","y1","x2","y2"])  # simple schema

    t0 = time.time()
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Detection per frame (we run once, then split high/low by conf)
        res = model.predict(source=frame, conf=min(args.conf_low, args.conf_high), iou=args.det_iou, imgsz=args.imgsz, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4), dtype=np.float32)
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,), dtype=np.float32)
        clses = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), dtype=int)
        # keep person class if specified
        if args.classes is not None and len(args.classes) > 0:
            keep = np.isin(clses, np.array(args.classes))
            boxes = boxes[keep]
            confs = confs[keep]
        # split high/low
        high_idx = np.where(confs >= args.conf_high)[0]
        low_idx  = np.where((confs >= args.conf_low) & (confs < args.conf_high))[0]
        dets_high = boxes[high_idx] if len(high_idx) else np.zeros((0,4), dtype=np.float32)
        dets_low  = boxes[low_idx]  if len(low_idx)  else np.zeros((0,4), dtype=np.float32)

        # Step tracker
        active_tracks = tracker.step(frame, dets_high, dets_low, app_model)

        # Draw
        vis = frame.copy()
        for t in active_tracks:
            x1,y1,x2,y2 = list(map(int, t.to_xyxy()))
            c = tuple(map(int, t.color))
            cv2.rectangle(vis, (x1,y1), (x2,y2), c, 2)
            cv2.putText(vis, f"ID {t.tid}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            csv_w.writerow([fidx, t.tid, x1,y1,x2,y2])

        writer.write(vis)
        if args.show:
            cv2.imshow("Offline MOT (Pass-1)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        fidx += 1
        if args.max_frames and fidx >= args.max_frames:
            break

    cap.release(); writer.release(); csv_f.close()
    if args.show:
        cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f"Done. {fidx} frames in {dt:.1f}s -> {fidx/max(dt,1e-6):.2f} FPS. Tracks CSV: {csv_path}")


# -------------------------- CLI --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="out_mot.mp4")
    ap.add_argument("--yolo", default="yolov11l.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--classes", nargs="*", type=int, default=[0], help="which classes to track (default: person=0)")
    # detection thresholds
    ap.add_argument("--conf_high", type=float, default=0.45)
    ap.add_argument("--conf_low", type=float, default=0.05)
    ap.add_argument("--det_iou", type=float, default=0.6)
    # tracker params
    ap.add_argument("--max_age", type=int, default=45)
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--w_iou", type=float, default=0.50)
    ap.add_argument("--w_app", type=float, default=0.35)
    ap.add_argument("--w_motion", type=float, default=0.15)
    # appearance
    ap.add_argument("--use_reid", action="store_true")
    ap.add_argument("--reid_backbone", choices=["r18","r50","osnet"], default="r50")
    ap.add_argument("--no_gem", action="store_true")
    ap.add_argument("--app_thr", type=float, default=0.35)
    ap.add_argument("--app_m", type=float, default=0.10)
    # viz/runtime
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    run(args)
