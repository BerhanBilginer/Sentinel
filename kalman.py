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
    # Ölçüm gürültüsü (pixels²) - must be reasonable to avoid numerical issues
    r_epsilon: float = 1.0  # 1 pixel standard deviation

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
        self.R = np.eye(4) * p.r_epsilon

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
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self.x = self.x + K @ y
        self.P = (self.I6 - K @ self.H) @ self.P
        # R≈0 durumunda sayısal güvenlik
        self.x[[0,1,4,5]] = z

    def center(self): return float(self.x[0]), float(self.x[1])
    def bbox(self):   return self.x[[0,1,4,5]].copy()
