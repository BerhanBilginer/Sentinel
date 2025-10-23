"""
PersonTracker with variable-dt Kalman, dt-aware gating and association
+ TorchReID appearance (opsiyonel)
+ ReID terminal logları
+ anti-ID-switch: min_hits, stationary lock, birth suppression, lazy death
"""
import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy
from collections import deque

from kalman import KalmanBBox, KalmanParams, KalmanEdgePoints, KalmanEdgeParams, bbox_to_corners
from gating import GatingParams, build_gate_matrix, HistoryParams, build_history_mask, iou_cxcywh
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
        reid_log_interval: int = 10,
        # --- anti-ID-switch kontroller ---
        min_hits: int = 2,                 # ID yayınlamak/çizmek için min ardışık eşleşme
        birth_suppress_radius: float = 50, # aktif track yakınında yeni doğumu bastır (px)
        stationary_speed_thr: float = 0.2, # px/s altı hız → durgun
        stationary_frames: int = 3,        # bu kadar frame durgun → kilitle
        lazy_death: Optional[int] = None,  # None: max_age kullan; yoksa override
        # --- size filtering ---
        min_bbox_width: float = 20.0,      # minimum bbox width in pixels
        min_bbox_height: float = 40.0,     # minimum bbox height in pixels
        min_bbox_area: float = 800.0,      # minimum bbox area in pixels²
        # --- edge point tracking ---
        use_edge_tracking: bool = False,   # enable edge point tracking
        edge_params: Optional[KalmanEdgeParams] = None
    ):
        self.kalman_params = kalman_params or KalmanParams()
        self.gating_params = gating_params or GatingParams(
            dof=4, chi2_p=0.95, tau_iou=0.15, tau_log_s=0.7, tau_ratio=0.5)
        self.assoc_params  = assoc_params  or AssocParams(
            w_m0=1.0, w_i0=1.5, w_a0=0.5, alpha=0.3, beta=0.5,
            stage1_conf_thr=0.6, l2_norm=150.0, app_default_cost=1.0,
            gamma_dt=0.2, delta_dt=0.15
        )

        self.tracks: Dict[int, KalmanBBox] = {}
        self.track_ages: Dict[int, int] = {}
        self.track_time_since_update: Dict[int, int] = {}
        self.track_hits: Dict[int, int] = {}            # ardışık eşleşme sayacı
        self.track_feats: Dict[int, np.ndarray] = {}
        self.track_feat_bank: Dict[int, deque] = {}     # son M feature
        self.track_stationary_ctr: Dict[int, int] = {}  # durgunluk sayacı
        self.next_id = 0
        self.max_age = max_age
        self.lazy_death = int(lazy_death) if lazy_death is not None else max_age
        self.min_hits = int(min_hits)
        self.birth_suppress_radius = float(birth_suppress_radius)
        self.stationary_speed_thr = float(stationary_speed_thr)
        self.stationary_frames_req = int(stationary_frames)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        # KalmanParams içindeki R kullanılacak (gating için sadece epsilon değil)
        self.R = np.diag([self.kalman_params.r_cx, self.kalman_params.r_cy,
                          self.kalman_params.r_w, self.kalman_params.r_h]).astype(float)

        self.last_gate_matrix: Optional[np.ndarray] = None
        self.last_assoc_result: Optional[Dict] = None

        self.reid = reid_encoder
        self.reid_momentum = float(reid_momentum)
        self.reid_debug = bool(reid_debug)
        self.reid_log_interval = int(reid_log_interval)
        self._frame_idx = 0

        self.track_feat_bank: Dict[int, deque] = {}     # vardı, koruduk
        self.track_bbox_hist: Dict[int, deque] = {}     # YENİ: kutu geçmişi

        self.history_params = HistoryParams(
            K_boxes=8,
            tau_hist_iou=0.25,
            tau_hist_cos=0.70,
            use_or_logic=True
        )
        self.dup_iou_thr = 0.75
        self.dup_cos_thr = 0.75

        # ID recovery (stable ID) system
        self.stable_id: Dict[int, int] = {}   # runtime track_id -> stable_id
        self.next_stable_id: int = 0          # yeni kimlik sayacı
        self.recovery_count_per_sid = {}      # stable_id -> recovery count

        # ID recovery (yeniden bağlama) ayarları
        self.recovery_tsu_max = 5             # kaç frame kayıp olan eski ID geri alınabilir
        self.recovery_cos_thr = 0.75          # min cosine (varsa)
        self.recovery_iouhist_thr = 0.30      # min IoU (geçmiş kutularla)
        self.recovery_dist_norm = 100.0       # merkez mesafesini bu değere bölerek 0-1 ölçek
        self.recovery_w_cos = 0.6             # füzyon skor ağırlıkları
        self.recovery_w_dist = 0.3
        self.recovery_w_size = 0.1

        # Size filtering parameters
        self.min_bbox_width = float(min_bbox_width)
        self.min_bbox_height = float(min_bbox_height)
        self.min_bbox_area = float(min_bbox_area)

        # Edge point tracking
        self.use_edge_tracking = bool(use_edge_tracking)
        self.edge_params = edge_params or KalmanEdgeParams()
        self.track_edge_filters: Dict[int, KalmanEdgePoints] = {}  # separate edge filters

    # ---------- yardımcılar ----------
    def bbox_to_measurement(self, box: Dict) -> np.ndarray:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h], dtype=float)

    def bbox_to_centroid_and_size(self, box):  # legacy alias
        return self.bbox_to_measurement(box)

    def _is_stationary(self, kalman: KalmanBBox) -> bool:
        vx, vy = float(kalman.x[2]), float(kalman.x[3])
        return (vx*vx + vy*vy) ** 0.5 <= self.stationary_speed_thr

    def _near_existing_track(self, z: np.ndarray, radius: float) -> Optional[int]:
        """Yeni doğum öncesi: yakında aktif track var mı? (cx,cy mesafesi)"""
        cx, cy = float(z[0]), float(z[1])
        best_tid, best_d = None, 1e9
        for tid, kf in self.tracks.items():
            tcx, tcy = kf.center()
            d = ((tcx - cx)**2 + (tcy - cy)**2) ** 0.5
            if d < radius and d < best_d and self.track_time_since_update.get(tid, 0) <= self.lazy_death:
                best_tid, best_d = tid, d
        return best_tid

    def _center_from_z(self, z: np.ndarray):
        return float(z[0]), float(z[1])

    def _area_from_z(self, z: np.ndarray):
        return max(1e-12, float(z[2]) * float(z[3]))

    def _is_bbox_too_small(self, box: Dict) -> bool:
        """Check if bounding box is too small to be tracked reliably."""
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        w = x2 - x1
        h = y2 - y1
        area = w * h
        
        return (w < self.min_bbox_width or 
                h < self.min_bbox_height or 
                area < self.min_bbox_area)

    def _filter_small_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter out detections with bounding boxes that are too small."""
        filtered = []
        for det in detections:
            if not self._is_bbox_too_small(det):
                filtered.append(det)
        return filtered

    def _best_old_for_new(self, tid_new: int, z_new: np.ndarray, feat_new: Optional[np.ndarray],
                          candidate_old_tids: List[int]) -> Optional[int]:
        """Yeni track için en iyi eski track (ID) adayını döndür (eşiklere uygunsa)."""
        best_tid, best_score = None, 1e9
        cxn, cyn = self._center_from_z(z_new)
        An = self._area_from_z(z_new)

        for to in candidate_old_tids:
            # eski track'in son ölçümü (history'den)
            hist = self.track_bbox_hist.get(to, None)
            iou_hist = 0.0
            if hist and len(hist) > 0:
                # yeni det ile geçmiş kutuların maks IoU'su
                iou_hist = max(iou_cxcywh(z_new, b) for b in hist)

            # görünüş benzerliği
            cos = -1.0
            if feat_new is not None:
                f_old = self.track_feats.get(to, None)
                if f_old is not None:
                    den = max(1e-12, np.linalg.norm(f_old) * np.linalg.norm(feat_new))
                    cos = float(np.dot(f_old, feat_new) / den)

            # hareket/mesafe
            cxo, cyo = self.tracks[to].center()
            dist = ((cxo - cxn)**2 + (cyo - cyn)**2) ** 0.5 / self.recovery_dist_norm

            # boyut tutarlılığı (log-alan farkı)
            Ao = self._area_from_z(self.tracks[to].bbox())
            sz = abs(np.log(An) - np.log(Ao))

            # Füzyon maliyeti: düşük daha iyi
            cos_cost = (1.0 - max(0.0, cos)) if cos >= 0.0 else 0.5  # feat yoksa orta maliyet
            score = self.recovery_w_cos * cos_cost + self.recovery_w_dist * dist + self.recovery_w_size * sz

            # Ön-eşikler (erken eleme)
            if (cos >= 0.0 and cos < self.recovery_cos_thr) and (iou_hist < self.recovery_iouhist_thr):
                continue

            if score < best_score:
                best_score, best_tid = score, to

        return best_tid

    def _recover_identities(self, new_tids: List[int], unmatched_old_tids: List[int], dets: List):
        """Yeni doğan track'lere, kısa süre önce kaybolan eski ID'leri geri bağla."""
        # recovery adayları: az önce unmatched olmuş ve çok yaşlı olmayanlar
        candidates = []
        for tk in unmatched_old_tids:
            tsu = self.track_time_since_update.get(tk, 9999)
            if 1 <= tsu <= self.recovery_tsu_max:
                candidates.append(tk)

        if not candidates or not new_tids:
            return

        for tid_new in new_tids:
            # yeni track'in bu frame'deki det'i:
            z_new = self.tracks[tid_new].bbox()
            feat_new = self.track_feats.get(tid_new, None)
            best_old = self._best_old_for_new(tid_new, z_new, feat_new, candidates)
            if best_old is None:
                continue

            # stable_id devrini yap: yeni track ekranda eski kimlikle görünsün
            old_sid = self.stable_id.get(best_old, best_old)
            self.stable_id[tid_new] = old_sid
            # recovery count tracking
            sid = old_sid
            self.recovery_count_per_sid[sid] = self.recovery_count_per_sid.get(sid, 0) + 1
            # (opsiyonel) hits'i devir: yeni track'in hits'ini yükselt
            self.track_hits[tid_new] = max(self.track_hits.get(tid_new, 1), self.track_hits.get(best_old, 1))

            # Artık eski track'i tamamen bırakabiliriz (unmatched'dı zaten)
            for d in (self.tracks, self.track_ages, self.track_time_since_update,
                      self.track_hits, self.track_feats, self.track_feat_bank,
                      self.track_bbox_hist, self.stable_id, self.track_stationary_ctr):
                d.pop(best_old, None)

            # candidate listesinden de çıkar
            if best_old in candidates:
                candidates.remove(best_old)

            # Log
            print(f"[ID-RECOVERY] new tid {tid_new} -> stable_id {old_sid} (from old tid {best_old})")

    def _suppress_duplicates(self):
        """Çok benzer iki track'i birleştir: yaşlı / daha çok 'hits' olan kalır."""
        tids = list(self.tracks.keys())
        removed = set()
        for i in range(len(tids)):
            ti = tids[i]
            if ti in removed or ti not in self.tracks:
                continue
            ci = self.tracks[ti].bbox()  # [cx,cy,w,h]
            fi = self.track_feats.get(ti, None)

            for j in range(i+1, len(tids)):
                tj = tids[j]
                if tj in removed or tj not in self.tracks:
                    continue
                cj = self.tracks[tj].bbox()
                fj = self.track_feats.get(tj, None)

                # IoU yüksek mi?
                iou = iou_cxcywh(ci, cj)
                if iou < self.dup_iou_thr:
                    continue

                # cosine uygun mu?
                cos_ok = False
                if fi is not None and fj is not None:
                    den = max(1e-12, np.linalg.norm(fi)*np.linalg.norm(fj))
                    cos = float(np.dot(fi, fj)/den)
                    cos_ok = cos >= self.dup_cos_thr
                else:
                    # feat yoksa yalnızca IoU'ya bakarak da bastırmak isteyebilirsin
                    cos = None

                if cos_ok or cos is None:
                    # hangisini tutacağız?
                    # kriter: hits yüksek olan > age > daha az kayıp
                    hi = self.track_hits.get(ti, 0); hj = self.track_hits.get(tj, 0)
                    ai = self.track_ages.get(ti, 0); aj = self.track_ages.get(tj, 0)
                    li = self.track_time_since_update.get(ti, 0); lj = self.track_time_since_update.get(tj, 0)
                    keep, drop = (ti, tj)
                    if (hj > hi) or (hj == hi and aj > ai) or (hj == hi and aj == ai and lj < li):
                        keep, drop = (tj, ti)

                    # drop'ı sil
                    for d in (self.tracks, self.track_ages, self.track_time_since_update,
                              self.track_hits, self.track_feats,
                              self.track_feat_bank, self.track_bbox_hist, self.track_stationary_ctr,
                              self.track_edge_filters):
                        d.pop(drop, None)
                    removed.add(drop)

    # ---------- çekirdek ----------
    def update(self, detections: List[Dict], frame_bgr=None, dt: float = None):
        """
        dt: bu frame ile bir önceki işlenen frame arasındaki gerçek süre (saniye).
        """
        # 0) Filter small detections first
        original_count = len(detections)
        detections = self._filter_small_detections(detections)
        filtered_count = len(detections)
        if original_count != filtered_count:
            print(f"[SIZE-FILTER] Filtered {original_count - filtered_count} small detections "
                  f"(min: {self.min_bbox_width}×{self.min_bbox_height}px, {self.min_bbox_area}px²), "
                  f"{filtered_count} remaining")

        # 1) Predict (dt ile)
        for track in self.tracks.values():
            track.predict(dt=dt)
        
        # Predict edge filters if enabled
        if self.use_edge_tracking:
            for edge_filter in self.track_edge_filters.values():
                edge_filter.predict(dt=dt)
        
        self._frame_idx += 1

        # Periodic recovery statistics logging
        if self._frame_idx % 100 == 0 and self.reid_debug:
            noisy = sorted(self.recovery_count_per_sid.items(), key=lambda x: -x[1])[:10]
            print("[RECOVERY-STATS] top:", noisy)

        if not detections:
            for track_id in self.tracks:
                self.track_ages[track_id] = self.track_ages.get(track_id, 0) + 1
                self.track_time_since_update[track_id] = self.track_time_since_update.get(track_id, 0) + 1
                # durgunsa sayaç artır
                self.track_stationary_ctr[track_id] = self.track_stationary_ctr.get(track_id, 0) + (1 if self._is_stationary(self.tracks[track_id]) else 0)
            self._remove_old_tracks()
            self.last_gate_matrix = None
            self.last_assoc_result = None
            return

        # 2) Tracks → TrackPred (ayrıca hist/bank listelerini hazırla)
        track_ids = list(self.tracks.keys())
        tracks_pred: List[TrackPred] = []
        hist_box_list, hist_feat_list = [], []
        for tid in track_ids:
            kalman = self.tracks[tid]
            tracks_pred.append(TrackPred(
                track_id=tid,
                x_pred=kalman.x.copy(),
                P_pred=kalman.P.copy(),
                feat=(self.track_feats.get(tid, None)),
                age=self.track_ages.get(tid, 0),
                time_since_update=self.track_time_since_update.get(tid, 0)
            ))
            # history containers
            hist_box_list.append(self.track_bbox_hist.get(tid, deque(maxlen=self.history_params.K_boxes)))
            hist_feat_list.append(self.track_feat_bank.get(tid, deque(maxlen=10)))

        # 3) Detections (ReID features)
        dets: List[Detection] = []
        det_feats = None
        if self.reid is not None and frame_bgr is not None and len(detections) > 0:
            det_feats = self.reid.encode_from_frame(frame_bgr, detections)
            if det_feats is not None and len(det_feats) == len(detections):
                det_feats = det_feats.astype(np.float32)
        det_feat_list = []  # history mask için liste
        for idx, det in enumerate(detections):
            z = self.bbox_to_measurement(det)
            conf = float(det.get('conf', 1.0))
            feat = None
            if det_feats is not None and idx < det_feats.shape[0]:
                feat = _l2norm(det_feats[idx])
            dets.append(Detection(z=z, conf=conf, feat=feat))
            det_feat_list.append(feat)

        # 4) Gating — klasik gate
        if tracks_pred:
            gp = deepcopy(self.gating_params)
            if dt is not None:
                base = self.gating_params.tau_iou
                gp.tau_iou = max(0.01, float(base) * float(np.exp(-0.15 * float(dt))))
            tracks_x = [tp.x_pred for tp in tracks_pred]
            tracks_P = [tp.P_pred for tp in tracks_pred]
            dets_z = [d.z for d in dets]
            gate_matrix = build_gate_matrix(tracks_x, tracks_P, dets_z, self.H, self.R, gp)

            # --- NEW: History mask & AND kombinasyonu ---
            hist_mask = build_history_mask(
                tracks_bbox_hist=hist_box_list,
                tracks_feat_bank=hist_feat_list,
                dets=dets_z,
                det_feats=det_feat_list,
                hp=self.history_params
            )
            gate_matrix = gate_matrix & hist_mask  # ikisini de sağlamalı
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

            # ReID DEBUG LOGS
            if self.reid_debug and (self.reid is not None) and (self._frame_idx % self.reid_log_interval == 0):
                num_det = len(dets); num_det_feat = sum(1 for d in dets if d.feat is not None)
                num_trk = len(tracks_pred); num_trk_feat = sum(1 for t in tracks_pred if t.feat is not None)
                print(f"[ReID] frame={self._frame_idx} dt={(0.0 if dt is None else dt):.3f} reid_active=True "
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

            # 6) Matches → update state & feat/hits & HISTORY APPEND
            for (track_idx, det_idx) in assoc_result['matches']:
                tid = track_ids[track_idx]
                z = dets[det_idx].z
                self.tracks[tid].update(z)
                
                # Update edge filter if enabled
                if self.use_edge_tracking and tid in self.track_edge_filters:
                    corners = bbox_to_corners(z)
                    self.track_edge_filters[tid].update(corners)
                
                self.track_ages[tid] = self.track_ages.get(tid, 0) + 1
                self.track_time_since_update[tid] = 0
                self.track_hits[tid] = self.track_hits.get(tid, 0) + 1
                # stationary sayacı
                self.track_stationary_ctr[tid] = self.track_stationary_ctr.get(tid, 0) + (1 if self._is_stationary(self.tracks[tid]) else 0)
                # feat EMA + bank
                df = dets[det_idx].feat
                if df is not None:
                    prev = self.track_feats.get(tid, df)
                    newf = _l2norm(self.reid_momentum * prev + (1.0 - self.reid_momentum) * df)
                    self.track_feats[tid] = newf
                    self.track_feat_bank.setdefault(tid, deque(maxlen=10)).append(df.astype(np.float32))
                # --- HISTORY kutu ekle ---
                self.track_bbox_hist.setdefault(tid, deque(maxlen=self.history_params.K_boxes)).append(z.copy())

            # 7) Unmatched tracks → age & stationary
            for track_idx in assoc_result['unmatched_tracks']:
                tid = track_ids[track_idx]
                self.track_time_since_update[tid] = self.track_time_since_update.get(tid, 0) + 1
                self.track_ages[tid] = self.track_ages.get(tid, 0) + 1
                if self._is_stationary(self.tracks[tid]):
                    self.track_stationary_ctr[tid] = self.track_stationary_ctr.get(tid, 0) + 1
                else:
                    self.track_stationary_ctr[tid] = 0

            # 8) Unmatched detections → doğum (mevcut mantığın içinde new_tids topla)
            new_tids = []
            for det_idx in assoc_result['unmatched_dets']:
                z = dets[det_idx].z
                # yakında yaşayan/stationary track var mı?
                near_tid = self._near_existing_track(z, self.birth_suppress_radius)
                if near_tid is not None and self.track_stationary_ctr.get(near_tid, 0) >= self.stationary_frames_req:
                    # yakındaki durgun track'i güncelle, yeni ID açma
                    self.tracks[near_tid].update(z)
                    
                    # Update edge filter if enabled
                    if self.use_edge_tracking and near_tid in self.track_edge_filters:
                        corners = bbox_to_corners(z)
                        self.track_edge_filters[near_tid].update(corners)
                    
                    self.track_time_since_update[near_tid] = 0
                    self.track_hits[near_tid] = self.track_hits.get(near_tid, 0) + 1
                    df = dets[det_idx].feat
                    if df is not None:
                        prev = self.track_feats.get(near_tid, df)
                        self.track_feats[near_tid] = _l2norm(self.reid_momentum * prev + (1.0 - self.reid_momentum) * df)
                        self.track_feat_bank.setdefault(near_tid, deque(maxlen=10)).append(df.astype(np.float32))
                    # history güncelle
                    self.track_bbox_hist.setdefault(near_tid, deque(maxlen=self.history_params.K_boxes)).append(z.copy())
                    continue  # yeni ID yok

                # normal doğum
                tid_new = self._create_new_track(z)
                new_tids.append(tid_new)
                df = dets[det_idx].feat
                if df is not None:
                    self.track_feats[tid_new] = _l2norm(df)
                    self.track_feat_bank.setdefault(tid_new, deque(maxlen=10)).append(df.astype(np.float32))
                # doğan için history başlat
                self.track_bbox_hist[tid_new] = deque([z.copy()], maxlen=self.history_params.K_boxes)

            # --- DOĞUMLARDAN SONRA: ID RECOVERY ---
            if new_tids:
                self._recover_identities(
                    new_tids=new_tids,
                    unmatched_old_tids=[track_ids[i] for i in assoc_result.get('unmatched_tracks', [])],
                    dets=dets
                )
        else:
            # hiç track yoksa tüm detlerden başlat
            for det in dets:
                tid_new = self._create_new_track(det.z)
                if det.feat is not None:
                    self.track_feats[tid_new] = _l2norm(det.feat)
                    self.track_feat_bank.setdefault(tid_new, deque(maxlen=10)).append(det.feat.astype(np.float32))
                # doğan için history başlat
                self.track_bbox_hist[tid_new] = deque([det.z.copy()], maxlen=self.history_params.K_boxes)
            self.last_assoc_result = None

        self._remove_old_tracks()

        # --- NEW: duplicate suppression step (opsiyonel) ---
        self._suppress_duplicates()

    # ---------- track yaşam döngüsü ----------
    def _create_new_track(self, z: np.ndarray) -> int:
        kalman = KalmanBBox(self.kalman_params)
        kalman.initialize(z)
        tid = self.next_id
        self.tracks[tid] = kalman
        self.track_ages[tid] = 1
        self.track_time_since_update[tid] = 0
        self.track_hits[tid] = 1  # ilk eşleşme sayımı
        self.track_stationary_ctr[tid] = 0
        self.track_feat_bank[tid] = deque(maxlen=10)
        # PROVISIONAL stable_id: şimdilik kendi tid'si
        self.stable_id[tid] = self.next_stable_id
        self.next_stable_id += 1
        
        # Initialize edge point tracking if enabled
        if self.use_edge_tracking:
            corners = bbox_to_corners(z)
            edge_filter = KalmanEdgePoints(self.edge_params)
            edge_filter.initialize(corners)
            self.track_edge_filters[tid] = edge_filter
        
        self.next_id += 1
        return tid

    def _remove_old_tracks(self):
        to_remove = [tid for tid, tsu in self.track_time_since_update.items() if tsu > self.lazy_death]
        for tid in to_remove:
            for d in (self.tracks, self.track_ages, self.track_time_since_update,
                      self.track_hits, self.track_feats, self.track_feat_bank,
                      self.track_bbox_hist, self.stable_id, self.track_stationary_ctr,
                      self.track_edge_filters):
                d.pop(tid, None)

    # ---------- arayüz ----------
    def get_predictions(self) -> List[Dict]:
        predictions = []
        for track_id in sorted(self.tracks.keys()):
            kalman = self.tracks[track_id]
            cx, cy = kalman.center()
            bbox = kalman.bbox()
            
            # Get edge points if available
            corners = None
            if self.use_edge_tracking and track_id in self.track_edge_filters:
                corners = self.track_edge_filters[track_id].get_corners()
            
            predictions.append({
                'track_id': track_id,                          # internal
                'display_id': self.stable_id.get(track_id, track_id),  # EKRANA BU
                'center': (cx, cy),
                'bbox': bbox,
                'corners': corners,                            # edge points if available
                'age': self.track_ages.get(track_id, 0),
                'time_since_update': self.track_time_since_update.get(track_id, 0),
                'hits': self.track_hits.get(track_id, 0),
            })
        return predictions

    def get_gate_info(self) -> Optional[np.ndarray]:
        return self.last_gate_matrix

    def get_assoc_info(self) -> Optional[Dict]:
        return self.last_assoc_result
