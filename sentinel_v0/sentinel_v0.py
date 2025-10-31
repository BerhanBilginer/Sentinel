"""
SAM (HF Transformers) + YOLO -> Tracking (Hungarian)
Occlusion-aware freeze + ReID (gallery memory, TTL)  [NO KALMAN]
Mavi tema: mask & bbox, Yeşil: opak overlap.

Kullanım (örnek):
python sam_hf_freeze_tracker.py --video input.mp4 --out out.mp4 \
  --yolo yolov8s.pt --sam_ckpt facebook/sam-vit-base --classes 0 --show \
  --reid_backbone r50 --gallery_ttl 180 --gallery_app_thr 0.28
"""

import cv2
import numpy as np
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from transformers import SamModel, SamProcessor

# --- Colors (BGR) ---
LIGHT_BLUE = (230, 216, 173)  # açık mavi (RGB 173,216,230 -> BGR 230,216,173)
PURE_GREEN = (0, 255, 0)      # opak yeşil
RED = (0, 0, 255)             # kırmızı - behind/occluded
ORANGE = (0, 165, 255)        # turuncu - in front/occluding
MASK_ALPHA = 0.35             # mavi mask saydamlığı

# ============ Yardımcılar ============
def iou_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.dtype != np.bool_:
        mask_a = mask_a.astype(bool)
    if mask_b.dtype != np.bool_:
        mask_b = mask_b.astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def occlusion_scores(det_masks, occ_thr_rel=0.35, occ_min_percent=0.05,
                     occ_min_iou=0.02, occ_asym_margin=0.02):
    """
    Detect occlusion relationships between detection masks.
    
    Args:
        occ_thr_rel: Main threshold - if object A covers >= this % of object B, A is in front
        occ_min_percent: Minimum occlusion % to consider (noise filter)
        occ_min_iou: Minimum IoU between masks to consider overlap
        occ_asym_margin: Required difference between directional occlusions for asymmetry
    
    Returns:
        front_of: List of sets - front_of[i] contains indices that i occludes
        behind: List of sets - behind[i] contains indices that occlude i
    """
    n = len(det_masks)
    front_of = [set() for _ in range(n)]
    behind   = [set() for _ in range(n)]
    if n <= 1:
        return front_of, behind
    masks = [m for (m, _) in det_masks]
    areas = [float(m.sum()) for m in masks]

    def mask_iou(a, b):
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        return (inter / union) if union > 0 else 0.0, inter

    for i in range(n):
        for j in range(i + 1, n):
            iou_ij, inter = mask_iou(masks[i], masks[j])
            
            # Calculate occlusion percentages (what % of each object is covered)
            occ_i_j = inter / max(areas[j], 1.0)  # What % of j is covered by i
            occ_j_i = inter / max(areas[i], 1.0)  # What % of i is covered by j
            
            # Skip if overlap is too small (either by IoU or percentage)
            if iou_ij < occ_min_iou:
                continue
            if occ_i_j < occ_min_percent and occ_j_i < occ_min_percent:
                continue
            
            # Determine which is in front based on asymmetric occlusion
            if (occ_i_j >= occ_thr_rel) and ((occ_i_j - occ_j_i) >= occ_asym_margin):
                # i covers >= threshold % of j, and significantly more than j covers i
                # Therefore: i is IN FRONT of j, j is BEHIND i
                front_of[i].add(j); behind[j].add(i)
            elif (occ_j_i >= occ_thr_rel) and ((occ_j_i - occ_i_j) >= occ_asym_margin):
                # j covers >= threshold % of i, and significantly more than i covers j
                # Therefore: j is IN FRONT of i, i is BEHIND j
                front_of[j].add(i); behind[i].add(j)
    return front_of, behind

def find_multi_overlap_indices(det_masks, k=3, min_percent=0.15):
    """
    3+ maskenin aynı piksel bölgesini paylaştığı (crowd) durumları bulur.
    Alan yüzdesi üzerinden değerlendirir (occlusion detection gibi).
    Dönen: crowd'a dahil olan detection indexlerinin set'i.
    """
    if len(det_masks) < k:
        return set()
    masks = [m for (m, _) in det_masks]
    areas = [float(m.sum()) for m in masks]
    H, W = masks[0].shape
    accum = np.zeros((H, W), dtype=np.uint16)
    for m in masks:
        if m is not None and m.any():
            accum[m] += 1
    crowd_region = accum >= k
    if not crowd_region.any():
        return set()
    involved = set()
    for i, m in enumerate(masks):
        inter = np.logical_and(m, crowd_region).sum()
        overlap_percent = inter / max(areas[i], 1.0)  # What % of mask i overlaps with crowd
        if overlap_percent >= min_percent:
            involved.add(i)
    return involved 

# ============ ReID Özellik Çıkarıcı (ResNet18/50 + GeM opsiyonu) ============
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.mean(x, dim=(-1, -2)).pow(1.0/self.p)  # HxW -> 1x1 (global pooling)
        return x

class ReIDEmbedder(nn.Module):
    def __init__(self, device, backbone="r18", use_gem=True):
        super().__init__()
        self.device = device
        if backbone == "r50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048
        else:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        # trunk (conv..layer4)
        self.trunk = nn.Sequential(*list(base.children())[:-2]).to(device).eval()  # (B,C,H,W)
        self.pool = GeM().to(device).eval() if use_gem else nn.AdaptiveAvgPool2d(1).to(device).eval()
        self.bn = nn.BatchNorm1d(feat_dim, affine=False).to(device).eval()  # feature whitening-like
        self.out_dim = feat_dim
        self.tx = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        logging.getLogger('tracker').info(f"[INFO] ReID backbone: {backbone.upper()} (dim={feat_dim}) | GeM={'on' if use_gem else 'off'}")

    @torch.no_grad()
    def forward(self, img_bgr, mask, bbox):
        x1,y1,x2,y2 = map(int, bbox)
        x1,y1 = max(0,x1), max(0,y1)
        crop = img_bgr[y1:y2+1, x1:x2+1, :]
        if crop.size == 0:
            return torch.zeros(self.out_dim, device=self.device)
        # arka planı karart (mask crop)
        m = mask[y1:y2+1, x1:x2+1]
        crop = crop.copy()
        if m.shape[:2] == crop.shape[:2]:
            bg = ~m
            crop[bg] = (0,0,0)
        # BGR->RGB PIL ve trunk
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ten = self.tx(pil).unsqueeze(0).to(self.device)  # (1,3,256,256)
        f = self.trunk(ten)                               # (1,C,h,w)
        if isinstance(self.pool, GeM):
            f = self.pool(f)                              # (1,C)
        else:
            f = self.pool(f).squeeze(-1).squeeze(-1)     # (1,C)
        f = f.view(-1)
        # BN (istatistikleri eval'de stabilize)
        f = self.bn(f.unsqueeze(0)).view(-1)
        f = F.normalize(f, p=2, dim=0)                   # 1D, L2 norm
        return f

def cosine_distance(f1: torch.Tensor, f2: torch.Tensor) -> float:
    if f1 is None or f2 is None or f1.numel() == 0 or f2.numel() == 0:
        return 1.0
    f1 = F.normalize(f1.view(-1), p=2, dim=0)
    f2 = F.normalize(f2.view(-1), p=2, dim=0)
    sim = F.cosine_similarity(f1, f2, dim=0).clamp(-1.0, 1.0).item()
    return 1.0 - sim

# --- Ek Özellik Çıkarıcı (Geometrik + Renk) ---
class FeatureExtractor:
    def __init__(self):
        pass

    def extract(self, frame, mask, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return torch.zeros(8)

        # --- Geometrik özellikler ---
        h, w = crop.shape[:2]
        area = h * w
        mask_area = mask.sum()
        geom = torch.tensor([
            (x1 + x2) / (2 * frame.shape[1]),  # cx norm
            (y1 + y2) / (2 * frame.shape[0]),  # cy norm
            w / frame.shape[1],
            h / frame.shape[0],
            mask_area / max(area, 1e-6)
        ], dtype=torch.float32)

        # --- Renk istatistikleri (ortalama HSV) ---
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_color = hsv.reshape(-1, 3).mean(axis=0) / 255.0
        color = torch.tensor(mean_color, dtype=torch.float32)

        return torch.cat([geom, color])  # toplam 8 özellik

# ============ Track ============
class Track:
    def __init__(self, tid, mask, bbox, fidx, device, use_reid=True, reid_dim=2048):
        self.id = tid
        self.mask = mask
        self.bbox = bbox
        self.last_frame = fidx
        self.hits = 1
        self.frozen_until = -1
        # ReID
        self.feat = torch.zeros(reid_dim, device=device) if use_reid else None
        self.extra_feat = torch.zeros(8, device=device)
        # Occlusion state
        self.is_behind = False
        self.is_in_front = False
        self.is_crowd_frozen = False

# ============ Galeri (TTL'li uzun süreli hafıza) ============
class Gallery:
    def __init__(self, ttl=150):
        self.mem = {}  # id -> {"feat": tensor(…), "last_seen": frame_idx}
        self.ttl = ttl
    def update_from_tracks(self, tracks, now):
        for t in tracks:
            if t.feat is None: 
                continue
            self.mem[t.id] = {"feat": t.feat.detach().clone(), "last_seen": now}
    def prune(self, now):
        drop = [gid for gid,rec in self.mem.items() if (now - rec["last_seen"]) > self.ttl]
        for gid in drop:
            self.mem.pop(gid, None)
    def best_match(self, feat, thr=0.30, verbose=False):
        if feat is None or feat.numel() == 0:
            return None, 1e9, []
        best_d, best_id = 1e9, None
        all_matches = []  # (id, distance)
        for gid, rec in self.mem.items():
            d = cosine_distance(feat, rec["feat"])
            all_matches.append((gid, d))
            if d < best_d:
                best_d, best_id = d, gid
        # Sort by distance for verbose output
        all_matches.sort(key=lambda x: x[1])
        return (best_id if best_d <= thr else None), best_d, all_matches

# ============ Logging Setup ============
def setup_logger(video_path, log_dir="logs"):
    """Setup logger to write to both console and file"""
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create log filename based on video name and timestamp
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"tracker_{video_name}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('tracker')
    logger.info(f"=" * 80)
    logger.info(f"SAM HF Freeze Tracker - Log Started")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Video: {video_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"=" * 80)
    logger.info("")
    
    return logger

# ============ Main ============
def main(args):
    # Setup logging
    logger = setup_logger(args.video, args.log_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    use_reid = (not args.no_reid)

    # YOLO
    yolo = YOLO(args.yolo)
    yolo.model.to(device.type)

    # HF SAM
    processor = SamProcessor.from_pretrained(args.sam_ckpt)
    sam = SamModel.from_pretrained(args.sam_ckpt).to(device).eval()

    # ReID embedder
    embedder = ReIDEmbedder(device, backbone=args.reid_backbone, use_gem=not args.no_gem) if use_reid else None
    reid_dim = 2048 if args.reid_backbone == "r50" else 512

    # Feature extractor
    feature_extractor = FeatureExtractor()

    # Gallery
    gallery = Gallery(ttl=args.gallery_ttl)

    # Video IO
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    # Parametreler
    FREEZE_FRAMES = args.freeze_frames
    FREEZE_DIST   = args.freeze_dist
    W_IOU = args.w_iou
    W_APP = args.w_app if use_reid else 0.0

    tracks = []
    next_id = 0
    frame_idx = 0
    t0 = time.time()
    
    # Frame buffer for debug mode (stores frames and processing state)
    frame_buffer = []  # [(frame_bgr, vis, tracks_snapshot)]
    debug_mode = args.debug
    current_debug_idx = 0
    paused = False  # Pause state

    # Debug mode enables frame buffering and detailed logging
    logger.info(f"Debug mode: {'ENABLED' if debug_mode else 'DISABLED'}")
    logger.info("Controls: SPACE=pause/resume, A=prev (when paused), D=next (when paused), Q=quit")
    if debug_mode:
        logger.info("[DEBUG] Frame buffering enabled - detailed logging active")
    
    while True:
        # Handle pause state - works in both debug and normal mode
        if paused and frame_buffer:
            logger.info(f"[PAUSED] Frame {current_debug_idx}/{len(frame_buffer)-1} | SPACE=resume, A=prev, D=next, Q=quit")
            while paused:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord(' '):  # Space - resume
                    paused = False
                    logger.info("[RESUMED] Processing continues...")
                    break
                elif key == ord('a'):  # Previous frame
                    if current_debug_idx > 0:
                        current_debug_idx -= 1
                        buffered = frame_buffer[current_debug_idx]
                        cv2.imshow("SAM HF Tracks (Freeze + ReID Gallery, No Kalman)", buffered['vis'])
                        logger.info(f"[PAUSED] ← Frame {current_debug_idx}/{len(frame_buffer)-1} - {len(buffered['tracks'])} tracks")
                        for t in buffered['tracks']:
                            logger.info(f"  - Track ID{t.id}: bbox={t.bbox}, frozen_until={t.frozen_until}")
                    else:
                        logger.info("[PAUSED] Already at first frame")
                elif key == ord('d'):  # Next frame
                    if current_debug_idx < len(frame_buffer) - 1:
                        current_debug_idx += 1
                        buffered = frame_buffer[current_debug_idx]
                        cv2.imshow("SAM HF Tracks (Freeze + ReID Gallery, No Kalman)", buffered['vis'])
                        logger.info(f"[PAUSED] → Frame {current_debug_idx}/{len(frame_buffer)-1} - {len(buffered['tracks'])} tracks")
                        for t in buffered['tracks']:
                            logger.info(f"  - Track ID{t.id}: bbox={t.bbox}, frozen_until={t.frozen_until}")
                    else:
                        logger.info(f"[PAUSED] Already at last buffered frame ({len(frame_buffer)-1})")
                elif key == ord('q'):  # Quit
                    logger.info("[PAUSED] User quit")
                    cap.release(); writer.release()
                    if args.show: cv2.destroyAllWindows()
                    return
                elif key == ord('l'):  # Toggle logging verbosity
                    if debug_mode:
                        logger.info("[PAUSED] Debug logging already enabled")
                    else:
                        logger.info("[PAUSED] Logging level unchanged (use --debug flag at startup for detailed logging)")
        
        # Debug mode and normal mode now behave the same - continuous play with pause capability
        
        ok, frame_bgr = cap.read()
        if not ok:
            if debug_mode and frame_buffer:
                logger.info("[DEBUG] End of video. Press 'q' to quit, 'a' to go back")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('a') and current_debug_idx > 0:
                        current_debug_idx -= 1
                        buffered = frame_buffer[current_debug_idx]
                        cv2.imshow("SAM HF Tracks (Freeze + ReID Gallery, No Kalman)", buffered['vis'])
                        logger.info(f"[DEBUG] ← Frame {current_debug_idx}/{len(frame_buffer)-1}")
            break

        # ---- YOLO ----
        if debug_mode:
            logger.info(f"\n{'='*80}")
            logger.info(f"[F{frame_idx:04d}] FRAME PROCESSING START")
            logger.info(f"{'='*80}")
        
        res = yolo.predict(source=frame_bgr, conf=args.conf, iou=args.iou, verbose=False)[0]
        det_boxes, det_confs, det_classes = [], [], []
        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            c = int(cls.item())
            if args.classes is not None and c not in args.classes:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 <= x1 or y2 <= y1: continue
            det_boxes.append([x1, y1, x2, y2])
            det_confs.append(float(conf.item()))
            det_classes.append(c)
        
        if debug_mode:
            logger.info(f"\n[YOLO] Detected {len(det_boxes)} objects (after filtering)")
            for idx, (box, conf, cls) in enumerate(zip(det_boxes, det_confs, det_classes)):
                area_val = (box[2]-box[0]) * (box[3]-box[1])
                logger.info(f"  Det{idx}: bbox={box}, conf={conf:.3f}, class={cls}, area={area_val}px")

        # ---- SAM masks (+ ReID feat opsiyonel) ----
        det_masks, det_feats = [], []
        if det_boxes:
            if debug_mode:
                logger.info(f"\n[SAM] Generating masks for {len(det_boxes)} detections...")
            pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            inputs = processor(pil_img, input_boxes=[[det_boxes]], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = sam(**inputs)
            masks_list = processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu()
            )
            masks_np = masks_list[0]
            for j, mset in enumerate(masks_np):
                m = (mset[0].numpy() > 0.5)
                box = det_boxes[j]
                det_masks.append((m, box))
                if debug_mode:
                    mask_area = m.sum()
                    bbox_area = (box[2]-box[0]) * (box[3]-box[1])
                    fill_ratio = mask_area / max(bbox_area, 1)
                    logger.info(f"  Det{j}: mask_pixels={mask_area}, fill_ratio={fill_ratio:.2%}")
                
                # Always extract extra features (geometry + color)
                extra_feat = feature_extractor.extract(frame_bgr, m, box).to(device)
                
                if use_reid:
                    # Combine extra features with ReID features
                    feat = embedder(frame_bgr, m, box)
                    feat_full = torch.cat([extra_feat, feat], dim=0)
                    det_feats.append(feat_full)
                else:
                    # Only use extra features when ReID is disabled
                    det_feats.append(extra_feat)
        
        # Update debug logging
        if det_masks and not use_reid:
            pass  # det_feats already populated with extra features
        
        if debug_mode and use_reid and det_feats:
            logger.info(f"\n[ReID] Extracted {len(det_feats)} feature vectors (dim={det_feats[0].shape[0]})")

        # ---- Occlusion & behind flags ----
        behind_flags = []
        front_flags = []
        if det_masks:
            if debug_mode:
                logger.info(f"\n[OCCLUSION] Analyzing {len(det_masks)} masks...")
            front_of, behind = occlusion_scores(
                det_masks,
                occ_thr_rel=args.occ_thr_rel,
                occ_min_percent=args.occ_min_percent,
                occ_min_iou=args.occ_min_iou,
                occ_asym_margin=args.occ_asym_margin
            )
            behind_flags = [(len(behind[i]) > 0) for i in range(len(det_masks))]
            front_flags = [(len(front_of[i]) > 0) for i in range(len(det_masks))]
            if debug_mode:
                # Calculate and show detailed occlusion percentages
                masks = [m for (m, _) in det_masks]
                areas = [float(m.sum()) for m in masks]
                
                for i in range(len(det_masks)):
                    if behind_flags[i]:
                        # Calculate what % of this object is occluded
                        occlusion_info = []
                        for occluder_idx in behind[i]:
                            inter = float(np.logical_and(masks[i], masks[occluder_idx]).sum())
                            occ_percent = (inter / areas[i]) * 100 if areas[i] > 0 else 0
                            occlusion_info.append(f"Det{occluder_idx}:{occ_percent:.1f}%")
                        logger.info(f"  Det{i}: BEHIND (occluded by: {', '.join(occlusion_info)})")
                    elif len(front_of[i]) > 0:
                        # Calculate what % of other objects this one occludes
                        occlusion_info = []
                        for occluded_idx in front_of[i]:
                            inter = float(np.logical_and(masks[i], masks[occluded_idx]).sum())
                            occ_percent = (inter / areas[occluded_idx]) * 100 if areas[occluded_idx] > 0 else 0
                            occlusion_info.append(f"Det{occluded_idx}:{occ_percent:.1f}%")
                        logger.info(f"  Det{i}: IN FRONT (occluding: {', '.join(occlusion_info)})")
                n_behind = sum(behind_flags)
                logger.info(f"  Summary: {n_behind} detections behind, {len(det_masks)-n_behind} in front/clear")

        # ---- Crowd (3+ overlap) dondurma ----
        if det_masks:
            crowd_idxs = find_multi_overlap_indices(
                det_masks,
                k=args.multi_overlap_k,
                min_percent=args.multi_overlap_min_percent
            )
            if crowd_idxs:
                logger.info(f"[F{frame_idx:04d}] CROWD: {len(crowd_idxs)} detection overlap (k>={args.multi_overlap_k})")
                # Crowd içindeki her detection için en yakın track'i bul ve dondur
                for j in crowd_idxs:
                    _, det_box = det_masks[j]
                    bx, by = center(det_box)
                    near_t, best = None, 1e18
                    for t_idx, t in enumerate(tracks):
                        cx, cy = center(t.bbox)
                        d2 = (cx - bx)**2 + (cy - by)**2
                        if d2 < best:
                            best, near_t = d2, t_idx
                    if near_t is not None and (best**0.5) <= FREEZE_DIST:
                        t = tracks[near_t]
                        t.frozen_until = max(t.frozen_until, frame_idx + args.freeze_frames_multi)
                        t.is_crowd_frozen = True
                        # Occlusion state'leri crowd sırasında anlamlı olmayabilir; nötrle
                        t.is_behind = False
                        t.is_in_front = False
                        logger.info(f"  -> Freeze ID{t.id} due to CROWD (dist={best**0.5:.1f}px) until {t.frozen_until}")

        # ---- Assignment (IoU + Appearance) ----
        if not det_masks:
            # expire
            tracks = [t for t in tracks if frame_idx - t.last_frame <= args.max_age]
        else:
            active_idx = [i for i,t in enumerate(tracks) if t.frozen_until < frame_idx]
            m = len(active_idx); n = len(det_masks)
            if debug_mode:
                logger.info(f"\n[ASSIGNMENT] {m} active tracks vs {n} detections")
                frozen = [i for i,t in enumerate(tracks) if t.frozen_until >= frame_idx]
                if frozen:
                    logger.info(f"  Frozen tracks: {[tracks[i].id for i in frozen]}")
                if active_idx:
                    logger.info(f"  Active tracks: {[tracks[i].id for i in active_idx]}")
            else:
                logger.info(f"[F{frame_idx:04d}] Assignment: {m} active tracks vs {n} detections")
            if m and n:
                cost = np.ones((m, n), dtype=np.float32) * 1e3
                if debug_mode:
                    logger.info(f"\n  Computing cost matrix ({m}x{n})...")
                for ii, ti in enumerate(active_idx):
                    tr = tracks[ti]
                    for j in range(n):
                        mask_j, _ = det_masks[j]
                        # IoU mask cost
                        iou_val = iou_mask(tr.mask, mask_j)
                        iou_c = 1.0 - iou_val
                        # Appearance cost (ReID)
                        if use_reid and (tr.feat is not None) and (det_feats[j] is not None) and len(det_feats[j]) > 8:
                            # Extract ReID features (skip first 8 which are extra features)
                            det_reid_feat = det_feats[j][8:]
                            app_c = cosine_distance(tr.feat, det_reid_feat)
                        else:
                            app_c = 0.0
                        
                        # Geometric and color costs (always available)
                        geo_c = 0.0
                        col_c = 0.0
                        if hasattr(tr, 'extra_feat') and det_feats[j] is not None and len(det_feats[j]) >= 8:
                            tr_extra = tr.extra_feat
                            det_extra = det_feats[j][:8] if len(det_feats[j]) > 8 else det_feats[j]
                            # Normalize geometric distance to [0, 1] range
                            geo_c = (torch.norm(tr_extra[:5] - det_extra[:5], p=2).item() / 2.236)  # sqrt(5)
                            # Normalize color distance to [0, 1] range
                            col_c = (torch.norm(tr_extra[5:] - det_extra[5:], p=2).item() / 1.732)  # sqrt(3)
                        
                        # Normalize weights to sum to 1.0
                        w_total = W_IOU + W_APP + args.w_geo + args.w_col
                        w_iou_norm = W_IOU / w_total
                        w_app_norm = W_APP / w_total
                        w_geo_norm = args.w_geo / w_total
                        w_col_norm = args.w_col / w_total
                        
                        fused = (
                                w_iou_norm * iou_c +
                                w_app_norm * app_c +
                                w_geo_norm * geo_c +
                                w_col_norm * col_c
                                )
                        cost[ii, j] = fused
                        if debug_mode:
                            logger.info(f"    Track ID{tr.id} <-> Det{j}: IoU={iou_val:.3f}(c={iou_c:.3f}), App_c={app_c:.3f}, Geo_c={geo_c:.3f}, Col_c={col_c:.3f}, Fused={fused:.3f}")
                if debug_mode:
                    logger.info(f"\n  Running Hungarian algorithm...")
                row_ind, col_ind = linear_sum_assignment(cost)
            else:
                row_ind, col_ind = np.array([], int), np.array([], int)

            assigned_t, assigned_d = set(), set()
            if debug_mode and (m and n):
                logger.info(f"\n[MATCHES] Hungarian results:")
                for rr, cc in zip(row_ind, col_ind):
                    cval = cost[rr, cc]
                    ti = active_idx[rr]
                    status = "✓ ACCEPT" if (np.isfinite(cval) and cval < args.max_cost) else "✗ REJECT"
                    logger.info(f"  Track ID{tracks[ti].id} <-> Det{cc}: cost={cval:.3f} (max={args.max_cost:.3f}) {status}")
            
            for rr, cc in zip(row_ind, col_ind):
                cval = cost[rr, cc] if (m and n) else 1e9
                if np.isfinite(cval) and cval < args.max_cost:
                    ti = active_idx[rr]
                    mask_j, box_j = det_masks[cc]
                    if not debug_mode:
                        logger.info(f"[F{frame_idx:04d}] ✓ MATCHED: Track ID{tracks[ti].id} <- Det{cc} (cost={cval:.3f})")
                    # update track core
                    tracks[ti].mask = mask_j
                    tracks[ti].bbox = box_j
                    tracks[ti].last_frame = frame_idx
                    tracks[ti].hits += 1
                    # Update occlusion state
                    tracks[ti].is_behind = behind_flags[cc] if cc < len(behind_flags) else False
                    tracks[ti].is_in_front = front_flags[cc] if cc < len(front_flags) else False
                    # Update features
                    if use_reid and len(det_feats[cc]) > 8:
                        # ReID EMA - Extract ReID features (skip first 8 which are extra features)
                        det_reid_feat = det_feats[cc][8:]
                        newf = F.normalize(det_reid_feat.view(-1), p=2, dim=0)
                        oldf = F.normalize(tracks[ti].feat.view(-1), p=2, dim=0) if tracks[ti].feat is not None else newf
                        m_ema = float(args.app_momentum)
                        mixed = (1.0 - m_ema) * oldf + m_ema * newf
                        tracks[ti].feat = F.normalize(mixed, p=2, dim=0)
                        if debug_mode:
                            sim_before = F.cosine_similarity(oldf, newf, dim=0).item()
                            logger.info(f"    ReID EMA: momentum={m_ema:.2f}, feat_similarity={sim_before:.3f}")
                    
                    # Update extra features (no EMA, direct replacement) - works with or without ReID
                    if det_feats[cc] is not None and len(det_feats[cc]) >= 8:
                        tracks[ti].extra_feat = det_feats[cc][:8] if len(det_feats[cc]) > 8 else det_feats[cc]
                    assigned_t.add(ti); assigned_d.add(cc)

            # arkada kalan detection → yakın track'i freeze et
            if debug_mode and any(behind_flags):
                logger.info(f"\n[FREEZE] Processing occluded detections...")
            for j, is_behind in enumerate(behind_flags):
                if not is_behind or j in assigned_d: continue
                bx, by = center(det_masks[j][1])
                near_t = None; best = 1e18
                for t_idx, t in enumerate(tracks):
                    cx, cy = center(t.bbox)
                    d2 = (cx - bx)**2 + (cy - by)**2
                    if d2 < best: best, near_t = d2, t_idx
                if near_t is not None and (best**0.5) <= FREEZE_DIST:
                    old_frozen = tracks[near_t].frozen_until
                    tracks[near_t].frozen_until = max(tracks[near_t].frozen_until, frame_idx + FREEZE_FRAMES)
                    if debug_mode:
                        logger.info(f"  Det{j} (behind) -> Freezing Track ID{tracks[near_t].id} (dist={best**0.5:.1f}px) until frame {tracks[near_t].frozen_until}")

            # ---- Galeri ile ID re-use (eşleşmeyen det'ler) ----
            # önce aktif ID setini çıkar
            active_ids = set(t.id for t in tracks)
            unassigned_count = len([j for j in range(len(det_masks)) if j not in assigned_d])
            if unassigned_count > 0:
                if debug_mode:
                    unassigned_dets = [j for j in range(len(det_masks)) if j not in assigned_d]
                    logger.info(f"\n[GALLERY] Processing {unassigned_count} unassigned detection(s): {unassigned_dets}")
                else:
                    logger.info(f"[F{frame_idx:04d}] Gallery: {unassigned_count} unassigned detection(s)")
            for j in range(len(det_masks)):
                if j in assigned_d: 
                    continue
                mask_j, box_j = det_masks[j]

                reuse_id = None
                if use_reid:
                    # galeri eşleşmesi - use only ReID features (skip first 8 extra features)
                    det_reid_feat = det_feats[j][8:] if len(det_feats[j]) > 8 else det_feats[j]
                    g_feat = F.normalize(det_reid_feat.view(-1), p=2, dim=0)
                    cand_id, dval, all_matches = gallery.best_match(g_feat, thr=args.gallery_app_thr, verbose=True)
                    logger.info(f"  Det{j} -> Gallery query: best_id={cand_id}, dist={dval:.3f}, thr={args.gallery_app_thr:.3f}")
                    # Show top 3 candidates
                    if all_matches:
                        logger.info(f"    Gallery candidates (top 3):")
                        for rank, (gid, dist) in enumerate(all_matches[:3], 1):
                            match_status = "✓ MATCH" if dist <= args.gallery_app_thr else "✗ TOO FAR"
                            active_status = "(ACTIVE)" if gid in active_ids else "(available)"
                            logger.info(f"      {rank}. ID{gid}: dist={dist:.3f} {match_status} {active_status}")
                    # aynı anda aktif bir track bu id'yi kullanmıyorsa yeniden ata
                    if cand_id is not None and cand_id not in active_ids:
                        reuse_id = cand_id
                        logger.info(f"    ✓ RE-USING ID {reuse_id} (not currently active)")
                    elif cand_id is not None:
                        logger.info(f"    ✗ Cannot reuse ID {cand_id} (already active)")
                    else:
                        if not all_matches:
                            logger.info(f"    ✗ FAILED: Gallery is empty")
                        else:
                            logger.info(f"    ✗ FAILED: All gallery IDs too far (best={dval:.3f} > thr={args.gallery_app_thr:.3f})")

                if reuse_id is not None:
                    # var olan id ile yeni track başlat
                    t = Track(reuse_id, mask_j, box_j, frame_idx, device, use_reid=use_reid, reid_dim=reid_dim)
                    if use_reid and g_feat is not None:
                        t.feat = g_feat
                    # Set extra features (geometry + color) - always available
                    if det_feats[j] is not None and len(det_feats[j]) >= 8:
                        t.extra_feat = det_feats[j][:8] if len(det_feats[j]) > 8 else det_feats[j]
                    # Set occlusion state
                    t.is_behind = behind_flags[j] if j < len(behind_flags) else False
                    t.is_in_front = front_flags[j] if j < len(front_flags) else False
                    if behind_flags and behind_flags[j]:
                        t.frozen_until = frame_idx + FREEZE_FRAMES
                        if debug_mode:
                            logger.info(f"    + Starting frozen (behind)")
                    tracks.append(t)
                    active_ids.add(reuse_id)
                    if not debug_mode:
                        logger.info(f"[F{frame_idx:04d}] 🔄 RE-ASSIGNED: Det{j} -> ID{reuse_id} (from gallery)")
                else:
                    # yeni id ata
                    new_id = max([next_id] + [t.id for t in tracks]) + 1 if tracks else next_id
                    t = Track(new_id, mask_j, box_j, frame_idx, device, use_reid=use_reid, reid_dim=reid_dim)
                    if use_reid and det_feats[j] is not None and len(det_feats[j]) > 8:
                        # Extract only ReID features (skip first 8 extra features)
                        t.feat = F.normalize(det_feats[j][8:].view(-1), p=2, dim=0)
                    # Set extra features (geometry + color) - always available
                    if det_feats[j] is not None and len(det_feats[j]) >= 8:
                        t.extra_feat = det_feats[j][:8] if len(det_feats[j]) > 8 else det_feats[j]
                    # Set occlusion state
                    t.is_behind = behind_flags[j] if j < len(behind_flags) else False
                    t.is_in_front = front_flags[j] if j < len(front_flags) else False
                    if behind_flags and behind_flags[j]:
                        t.frozen_until = frame_idx + FREEZE_FRAMES
                        if debug_mode:
                            logger.info(f"    + Starting frozen (behind)")
                    tracks.append(t)
                    next_id = new_id + 1
                    if not debug_mode:
                        logger.info(f"[F{frame_idx:04d}] ✨ NEW ID: Det{j} -> ID{new_id} (fresh track)")
                        if use_reid and all_matches:
                            logger.info(f"    Reason: Closest gallery ID was {all_matches[0][0]} with dist={all_matches[0][1]:.3f}")

            # yaşat/düşür
            alive = []
            dropped_ids = []
            for idx, t in enumerate(tracks):
                if idx in assigned_t:
                    alive.append(t)
                else:
                    age = frame_idx - t.last_frame
                    if age <= args.max_age:
                        # Reset occlusion state for LOST tracks (no detection this frame)
                        t.is_behind = False
                        t.is_in_front = False
                        alive.append(t)
                        if debug_mode and age > 0:
                            logger.info(f"  Track ID{t.id}: LOST for {age} frames (max={args.max_age})")
                    else:
                        dropped_ids.append((t.id, age))
            if dropped_ids:
                if debug_mode:
                    logger.info(f"\n[DROPPED] Removing {len(dropped_ids)} old tracks:")
                for tid, age in dropped_ids:
                    logger.info(f"  ID{tid}: age={age} frames > max_age={args.max_age}")
            tracks = alive

        # ---- Galeri güncelle/prune ----
        if use_reid:
            old_gallery_ids = set(gallery.mem.keys())
            gallery.update_from_tracks(tracks, frame_idx)
            gallery.prune(frame_idx)
            new_gallery_ids = set(gallery.mem.keys())
            
            # Show what was pruned from gallery
            pruned = old_gallery_ids - new_gallery_ids
            if pruned:
                if debug_mode:
                    logger.info(f"[GALLERY PRUNED] {len(pruned)} IDs: {list(pruned)}")
                else:
                    for pid in pruned:
                        logger.info(f"[F{frame_idx:04d}] 🗑️  GALLERY PRUNED: ID{pid} (TTL expired, not seen for >{args.gallery_ttl} frames)")
            
            if debug_mode:
                logger.info(f"\n[GALLERY] State: {len(gallery.mem)} IDs stored - {sorted(list(gallery.mem.keys()))}")
            elif frame_idx % 30 == 0:  # Every 30 frames in normal mode
                logger.info(f"[F{frame_idx:04d}] 📚 Gallery state: {len(gallery.mem)} IDs stored - {sorted(list(gallery.mem.keys()))}")

        # ---- Görselleştirme ----
        vis = frame_bgr.copy()
        overlay = np.zeros_like(vis, np.uint8)
        for t in tracks:
            if t.mask.sum() > 0:
                # Color based on occlusion state
                if hasattr(t, 'is_behind') and t.is_behind:
                    color = RED  # Behind/occluded = RED
                elif hasattr(t, 'is_in_front') and t.is_in_front:
                    color = ORANGE  # In front/occluding = ORANGE
                else:
                    color = LIGHT_BLUE  # Normal = LIGHT_BLUE
                overlay[t.mask] = color
        vis = cv2.addWeighted(vis, 1.0, overlay, MASK_ALPHA, 0)

        overlap_mask = None
        if len(tracks) > 1:
            Hh, Ww = vis.shape[:2]
            accum = np.zeros((Hh, Ww), dtype=np.uint16)
            for t in tracks:
                if t.mask.sum() > 0:
                    accum[t.mask] += 1
            overlap_mask = accum >= 2
        if overlap_mask is not None and overlap_mask.any():
            vis[overlap_mask] = PURE_GREEN

        for t in tracks:
            x1,y1,x2,y2 = map(int, t.bbox)
            # Color based on occlusion state
            if hasattr(t, 'is_behind') and t.is_behind:
                color = RED
            elif hasattr(t, 'is_in_front') and t.is_in_front:
                color = ORANGE
            else:
                color = LIGHT_BLUE
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID {t.id}"
            if t.frozen_until >= frame_idx:
                if getattr(t, 'is_crowd_frozen', False):
                    label += " (CROWD)"
                else:
                    label += " (FROZEN)"
            cv2.putText(vis, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Store frame in buffer (enabled in debug mode, or when paused in normal mode)
        if debug_mode or paused:
            import copy
            # Create lightweight track snapshot
            tracks_snapshot = []
            for t in tracks:
                snap = type('obj', (object,), {
                    'id': t.id, 'bbox': t.bbox.copy() if hasattr(t.bbox, 'copy') else list(t.bbox),
                    'frozen_until': t.frozen_until, 'hits': t.hits, 'last_frame': t.last_frame
                })()
                tracks_snapshot.append(snap)
            frame_buffer.append({
                'frame': frame_bgr.copy(),
                'vis': vis.copy(),
                'tracks': tracks_snapshot,
                'frame_idx': frame_idx
            })
            current_debug_idx = len(frame_buffer) - 1
        
        if debug_mode:
            behind_count = sum(1 for t in tracks if hasattr(t, 'is_behind') and t.is_behind)
            front_count = sum(1 for t in tracks if hasattr(t, 'is_in_front') and t.is_in_front)
            logger.info(f"\n[SUMMARY] Frame {frame_idx}:")
            logger.info(f"  Active tracks: {len(tracks)} ({[t.id for t in tracks]})")
            logger.info(f"  Track hits: {[(t.id, t.hits) for t in tracks]}")
            logger.info(f"  Frozen: {sum(1 for t in tracks if t.frozen_until >= frame_idx)}")
            logger.info(f"  Occlusion: {behind_count} RED (behind), {front_count} ORANGE (in front), {len(tracks)-behind_count-front_count} BLUE (clear)")
            logger.info(f"  Gallery size: {len(gallery.mem) if use_reid else 'N/A'}")
            logger.info(f"{'='*80}\n")
        
        writer.write(vis)
        if args.show:
            cv2.imshow("SAM HF Tracks (Freeze + ReID Gallery, No Kalman)", vis)
            # Check for user input (works in both debug and normal mode)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - pause
                paused = True
                current_debug_idx = len(frame_buffer) - 1 if frame_buffer else 0
                logger.info(f"[PAUSED] Press SPACE to resume, A/D to navigate, Q to quit")

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release(); writer.release()
    if args.show: cv2.destroyAllWindows()
    dt = time.time() - t0
    logger.info("")
    logger.info(f"=" * 80)
    logger.info(f"Done. {frame_idx} frames in {dt:.1f}s -> {frame_idx/max(dt,1e-6):.2f} FPS")
    logger.info(f"Log saved to: logs/tracker_{Path(args.video).stem}_*.log")
    logger.info(f"=" * 80)

# ============ Argümanlar ============
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="tracked_out.mp4")
    ap.add_argument("--yolo", default="yolov8s.pt")
    ap.add_argument("--sam_ckpt", default="facebook/sam-vit-base")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--classes", nargs="*", type=int, default=None)
    ap.add_argument("--min_iou_match", type=float, default=0.30)
    ap.add_argument("--max_age", type=int, default=15)
    ap.add_argument("--freeze_frames", type=int, default=10)
    ap.add_argument("--freeze_dist", type=float, default=60.0)
    # Occlusion eşikleri
    ap.add_argument("--occ_thr_rel", type=float, default=0.40,
                    help="Main occlusion threshold (0-1): if object A covers >= this % of B, A is in front")
    ap.add_argument("--occ_min_percent", type=float, default=0.05,
                    help="Minimum occlusion % to consider (0-1, default 0.05 = 5%%)")
    ap.add_argument("--occ_min_iou", type=float, default=0.03,
                    help="Minimum IoU between masks to consider overlap")
    ap.add_argument("--occ_asym_margin", type=float, default=0.02,
                    help="Required occlusion % difference for asymmetry detection")
    # Füzyon ağırlıkları (Kalman yok)
    ap.add_argument("--w_iou", type=float, default=0.55, help="IoU cost ağırlığı")
    ap.add_argument("--w_app", type=float, default=0.45, help="Appearance (1-cos) ağırlığı")
    ap.add_argument("--max_cost", type=float, default=0.85, help="Eşleşme kabul üst sınırı")
    ap.add_argument("--w_geo", type=float, default=0.15, help="Geometrik uzaklık ağırlığı (konum, oran, alan)")
    ap.add_argument("--w_col", type=float, default=0.10, help="Renk istatistikleri uzaklık ağırlığı")
    # ReID EMA + backbone + GeM
    ap.add_argument("--app_momentum", type=float, default=0.20,
                    help="ReID özelliği EMA karışım katsayısı (0..1)")
    ap.add_argument("--reid_backbone", choices=["r18","r50"], default="r50",
                    help="ReID backbone seçimi (r18=512d, r50=2048d)")
    ap.add_argument("--no_gem", action="store_true",
                    help="GeM yerine AvgPool kullan")
    # ReID galerisi
    ap.add_argument("--gallery_ttl", type=int, default=150,
                    help="Galeri hafızasında ID’nin tutulacağı maksimum kopukluk (frame)")
    ap.add_argument("--gallery_app_thr", type=float, default=0.30,
                    help="Galeri re-ID eşiği (1-cos); küçükse daha katı")
    # ReID aç/kapat
    ap.add_argument("--no_reid", action="store_true",
                    help="ReID appearance maliyetini devre dışı bırak")

    ap.add_argument("--multi_overlap_k", type=int, default=3,
                help="Aynı bölgede en az kaç maske varsa crowd sayılır")
    ap.add_argument("--multi_overlap_min_percent", type=float, default=0.15,
                    help="Crowd bölgesiyle kesişim için minimum alan yüzdesi (0.0-1.0)")
    ap.add_argument("--freeze_frames_multi", type=int, default=20,
                    help="3+ overlap için freeze süresi (frame)")

    # Görselleştirme
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max_frames", type=int, default=0)
    # Logging
    ap.add_argument("--log_dir", default="logs", help="Directory to save debug log files")
    # Debug mode
    ap.add_argument("--debug", action="store_true", help="Enable debug mode with frame buffering and detailed logging (SPACE=pause, A/D=navigate when paused, Q=quit)")

    args = ap.parse_args()
    if args.classes is not None and len(args.classes) == 0:
        args.classes = None
    main(args)
