from dataclasses import dataclass
import numpy as np

@dataclass
class KalmanParams:
    # başlangıç dt; her adım predict(dt) ile dinamik güncellenecek
    dt: float = 1.0
    # Sürekli beyaz ivme gürültüsü (CWNA) gücü (px/s^2)^2
    q_acc: float = 1.0        # cx,cy için
    # w,h için random-walk gücü
    q_w:   float = 0.05
    q_h:   float = 0.05
    # Ölçüm gürültüsü (pixels^2): makul değerler ver (std^2)
    r_cx: float = 4.0   # ~2px std
    r_cy: float = 4.0   # ~2px std
    r_w:  float = 9.0   # ~3px std
    r_h:  float = 9.0   # ~3px std

class KalmanBBox:
    """
    State: x = [cx, cy, vx, vy, w, h]
    CV (cx,cy,vx,vy) + RW(w,h)
    """
    def __init__(self, p: KalmanParams):
        self.p = p
        self.I6 = np.eye(6)
        self.x = None
        self.P = None
        # Ölçüm modeli z=[cx,cy,w,h]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        # R: artık epsilon değil; her bileşen için variance
        self.R = np.diag([p.r_cx, p.r_cy, p.r_w, p.r_h]).astype(float)

    def _build_F(self, dt: float) -> np.ndarray:
        return np.array([
            [1, 0, dt, 0,  0, 0],
            [0, 1, 0,  dt, 0, 0],
            [0, 0, 1,  0,  0, 0],
            [0, 0, 0,  1,  0, 0],
            [0, 0, 0,  0,  1, 0],
            [0, 0, 0,  0,  0, 1],
        ], dtype=float)

    def _build_Q(self, dt: float) -> np.ndarray:
        qa = self.p.q_acc
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
        Qcv = qa * np.array([
            [dt4/4,    0,    dt3/2,   0],
            [   0,  dt4/4,     0,   dt3/2],
            [dt3/2,   0,     dt2,     0],
            [   0,  dt3/2,     0,    dt2],
        ], dtype=float)
        Q = np.zeros((6,6), dtype=float)
        Q[:2, :2] = Qcv[:2, :2]
        Q[:2, 2:4] = Qcv[:2, 2:4]
        Q[2:4, :2] = Qcv[2:4, :2]
        Q[2:4, 2:4] = Qcv[2:4, 2:4]
        Q[4,4] = self.p.q_w * dt2
        Q[5,5] = self.p.q_h * dt2
        return Q

    def initialize(self, z, vx0: float = 0.0, vy0: float = 0.0, p0_scale: float = 50.0):
        cx, cy, w, h = z
        self.x = np.array([cx, cy, vx0, vy0, w, h], dtype=float)
        self.P = np.eye(6) * p0_scale

    def predict(self, dt: float = None):
        dt = float(self.p.dt if dt is None else max(1e-6, dt))
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        # SNAP YOK — klasik Kalman düzeltmesi
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self.x = self.x + K @ y
        self.P = (self.I6 - K @ self.H) @ self.P

    def center(self): return float(self.x[0]), float(self.x[1])
    def bbox(self):   return self.x[[0,1,4,5]].copy()


@dataclass
class KalmanEdgeParams:
    """Parameters for edge point tracking Kalman filter"""
    dt: float = 1.0
    # Process noise for each corner point (px/s^2)^2
    q_acc: float = 1.0
    # Measurement noise for each corner point (px^2)
    r_corner: float = 4.0  # ~2px std per corner


class KalmanEdgePoints:
    """
    Extended Kalman filter tracking 4 corner points of bounding box
    State: x = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3, x4, y4, vx4, vy4]
    Where corners are: (x1,y1)=top-left, (x2,y2)=top-right, (x3,y3)=bottom-right, (x4,y4)=bottom-left
    """
    def __init__(self, p: KalmanEdgeParams):
        self.p = p
        self.I16 = np.eye(16)  # 4 corners × 4 states each (x,y,vx,vy)
        self.x = None
        self.P = None
        
        # Measurement model: observe all 8 coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
        self.H = np.zeros((8, 16), dtype=float)
        for i in range(4):  # 4 corners
            self.H[2*i, 4*i] = 1.0      # x coordinate
            self.H[2*i+1, 4*i+1] = 1.0  # y coordinate
            
        # Measurement noise covariance
        self.R = np.eye(8) * p.r_corner

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix for 4 independent CV models"""
        F = np.zeros((16, 16), dtype=float)
        # Each corner follows CV model: [x, y, vx, vy]
        cv_block = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        for i in range(4):  # 4 corners
            F[4*i:4*i+4, 4*i:4*i+4] = cv_block
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        """Process noise covariance for 4 independent CV models"""
        qa = self.p.q_acc
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
        
        # CV process noise for one corner
        cv_q = qa * np.array([
            [dt4/4, 0, dt3/2, 0],
            [0, dt4/4, 0, dt3/2],
            [dt3/2, 0, dt2, 0],
            [0, dt3/2, 0, dt2]
        ], dtype=float)
        
        Q = np.zeros((16, 16), dtype=float)
        for i in range(4):  # 4 corners
            Q[4*i:4*i+4, 4*i:4*i+4] = cv_q
        return Q

    def initialize(self, bbox_corners, p0_scale: float = 50.0):
        """
        Initialize with 4 corner points
        bbox_corners: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        # Initialize state: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        self.x = np.zeros(16, dtype=float)
        for i, (x, y) in enumerate(bbox_corners):
            self.x[4*i] = x      # x coordinate
            self.x[4*i+1] = y    # y coordinate
            # velocities start at 0
            
        self.P = np.eye(16) * p0_scale

    def predict(self, dt: float = None):
        dt = float(self.p.dt if dt is None else max(1e-6, dt))
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, bbox_corners):
        """
        Update with observed corner points
        bbox_corners: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        # Convert corners to measurement vector
        z = np.zeros(8, dtype=float)
        for i, (x, y) in enumerate(bbox_corners):
            z[2*i] = x
            z[2*i+1] = y
            
        # Kalman update
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self.x = self.x + K @ y
        self.P = (self.I16 - K @ self.H) @ self.P

    def get_corners(self):
        """Get current corner estimates"""
        corners = []
        for i in range(4):
            x = float(self.x[4*i])
            y = float(self.x[4*i+1])
            corners.append((x, y))
        return corners

    def get_bbox_from_corners(self):
        """Convert corner points back to center-width-height format"""
        corners = self.get_corners()
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        
        return np.array([cx, cy, w, h], dtype=float)

    def center(self):
        """Get center point from corners"""
        bbox = self.get_bbox_from_corners()
        return float(bbox[0]), float(bbox[1])

    def bbox(self):
        """Get bounding box in [cx, cy, w, h] format"""
        return self.get_bbox_from_corners()


def bbox_to_corners(bbox):
    """
    Convert [cx, cy, w, h] to 4 corner points
    Returns: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    Where: top-left, top-right, bottom-right, bottom-left
    """
    cx, cy, w, h = bbox
    half_w, half_h = w/2, h/2
    
    corners = [
        (cx - half_w, cy - half_h),  # top-left
        (cx + half_w, cy - half_h),  # top-right
        (cx + half_w, cy + half_h),  # bottom-right
        (cx - half_w, cy + half_h),  # bottom-left
    ]
    return corners


def corners_to_bbox(corners):
    """
    Convert 4 corner points to [cx, cy, w, h]
    corners: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    
    return np.array([cx, cy, w, h], dtype=float)
