# reid_encoder.py
# Minimal TorchReID encoder (OSNet) for person appearance embeddings.

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchreid
    _HAS_TREID = True
except Exception as e:
    print(f"[ReID] Import failed: {type(e).__name__}: {e}")
    _HAS_TREID = False


class ReIDEncoder:
    """
    Person ReID encoder using TorchReID model zoo.
    Default: osnet_x0_25 pretrained on 'msmt17' (light & fast).
    """
    def __init__(self,
                 model_name: str = "osnet_x0_25",
                 pretrained_dataset: str = "msmt17",
                 weight_path: str = None,
                 device: str = None,
                 input_size=(256, 128),
                 batch_size: int = 32,
                 half: bool = False):
        if not _HAS_TREID:
            raise ImportError("torchreid not installed. pip install torchreid")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = half and self.device.type == "cuda"
        self.batch_size = batch_size
        self.h, self.w = int(input_size[0]), int(input_size[1])

        # Build model
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, loss="softmax", pretrained=False
        )
        
        # Load weights
        import os
        if weight_path and os.path.exists(weight_path):
            print(f"[ReID] Loading custom weights from {weight_path}")
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[ReID] Auto-downloading {model_name}_{pretrained_dataset}")
            torchreid.utils.load_pretrained_weights(self.model, f"{model_name}_{pretrained_dataset}")
        
        self.model.eval().to(self.device)
        if self.half:
            self.model.half()

        # Normalization (TorchReID default)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        print(f"[ReID] device={self.device.type}, half={self.half}, input_size=({self.h},{self.w})")


    @torch.no_grad()
    def _preprocess(self, imgs_bgr: np.ndarray) -> torch.Tensor:
        """
        imgs_bgr: (N, H, W, 3) uint8 BGR
        returns:  (N, 3, h, w) float tensor
        """
        imgs_rgb = imgs_bgr[..., ::-1].copy()  # BGR->RGB
        t = torch.from_numpy(imgs_rgb).to(self.device).float() / 255.0  # [0,1]
        t = t.permute(0, 3, 1, 2)  # NCHW
        t = F.interpolate(t, size=(self.h, self.w), mode="bilinear", align_corners=False)
        if self.half:
            t = t.half()
            mean = self.mean.half()
            std = self.std.half()
        else:
            mean = self.mean
            std = self.std
        t = (t - mean) / std
        return t

    @torch.no_grad()
    def encode_crops(self, crops_bgr: list) -> np.ndarray:
        """
        crops_bgr: list of (H, W, 3) uint8 BGR images (can have different sizes)
        returns L2-normalized embeddings: (N, D) float32
        """
        if crops_bgr is None or len(crops_bgr) == 0:
            return np.empty((0, 512), dtype=np.float32)

        # Process crops individually and batch the preprocessed tensors
        all_feats = []
        batch_tensors = []
        
        for crop in crops_bgr:
            # Preprocess single crop: adds batch dimension [1, 3, H, W]
            crop_batch = np.expand_dims(crop, axis=0)
            tensor = self._preprocess(crop_batch)
            batch_tensors.append(tensor)
            
            # Process batch when full
            if len(batch_tensors) >= self.batch_size:
                batch = torch.cat(batch_tensors, dim=0)
                f = self.model(batch)
                f = F.normalize(f, dim=1)
                all_feats.append(f.float().cpu().numpy())
                batch_tensors = []
        
        # Process remaining crops
        if batch_tensors:
            batch = torch.cat(batch_tensors, dim=0)
            f = self.model(batch)
            f = F.normalize(f, dim=1)
            all_feats.append(f.float().cpu().numpy())
        
        return np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 512), dtype=np.float32)

    def encode_from_frame(self, frame_bgr: np.ndarray, det_xyxy_list: list) -> np.ndarray:
        """
        frame_bgr: HxWx3 BGR
        det_xyxy_list: [{"x1":..., "y1":..., "x2":..., "y2":...}, ...]
        returns: (N, D) or empty
        """
        if frame_bgr is None or not det_xyxy_list:
            return np.empty((0, 512), dtype=np.float32)

        H, W = frame_bgr.shape[:2]
        crops = []
        for det in det_xyxy_list:
            x1 = int(max(0, min(W - 1, det["x1"])))
            y1 = int(max(0, min(H - 1, det["y1"])))
            x2 = int(max(0, min(W - 1, det["x2"])))
            y2 = int(max(0, min(H - 1, det["y2"])))
            # minimal size guard
            if x2 <= x1 or y2 <= y1:
                # fallback tiny crop at least 2x2
                x2 = min(W - 1, x1 + 2)
                y2 = min(H - 1, y1 + 2)
            crop = frame_bgr[y1:y2, x1:x2]
            crops.append(crop)
        
        # Pass list of crops (different sizes OK)
        return self.encode_crops(crops)
