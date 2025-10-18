from dataclasses import dataclass
import numpy as np

@dataclass
class KalmanParams:
    dt: float = 1.0
    # Süreç gürültüsü (sahne dinamizmine göre ayarla)
    q_cx: float = 0.5
    q_cy: float = 0.5
    q_vx: float = 1.5
    q_vy: float = 1.5
    q_w:  float = 0.1
    q_h:  float = 0.1
    # Ölçüm gürültüsü ~0 (R=0 hedefi için epsilon)
    r_epsilon: float = 1e-9   # sayısal kararlılık için çok küçük pozitif değer

class KalmanBBox:
    def __init__(self, p: KalmanParams):
        self.p = p
        dt = p.dt

        # F: sabit hız modeli
        self.F = np.array([
            [1, 0, dt, 0,  0, 0],
            [0, 1, 0,  dt, 0, 0],
            [0, 0, 1,  0,  0, 0],
            [0, 0, 0,  1,  0, 0],
            [0, 0, 0,  0,  1, 0],
            [0, 0, 0,  0,  0, 1],
        ], dtype=float)

        # H: z = [cx, cy, w, h]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Q: süreç gürültüsü
        self.Q = np.diag([
            p.q_cx, p.q_cy, p.q_vx, p.q_vy, p.q_w, p.q_h
        ]).astype(float)

        # R ≈ 0: epsilon ile
        self.R = np.eye(4) * p.r_epsilon

        self.I6 = np.eye(6)
        self.x = None
        self.P = None

    def initialize(self, z: np.ndarray, vx0: float = 0.0, vy0: float = 0.0, p0_scale: float = 50.0):
        cx, cy, w, h = z
        self.x = np.array([cx, cy, vx0, vy0, w, h], dtype=float)
        self.P = np.eye(6) * p0_scale

    def predict(self):
        # x̂_k^- = F x̂_{k-1}
        self.x = self.F @ self.x
        # P_k^-  = F P_{k-1} F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        # y_k = z_k - H x̂_k^-
        y = z - (self.H @ self.x)
        # S_k = H P_k^- H^T + R  (R ≈ 0)
        S = self.H @ self.P @ self.H.T + self.R
        # K_k = P_k^- H^T S_k^{-1}
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # x̂_k = x̂_k^- + K_k y_k
        self.x = self.x + K @ y
        # P_k = (I - K_k H) P_k^-
        self.P = (self.I6 - K @ self.H) @ self.P

        # (İsteğe bağlı) tam ‘snap’: ölçülen bileşenleri doğrudan z ile eşitle
        # Bu satır R→0 sınırındaki teorik sonucu numerikte garantiler:
        self.x[[0,1,4,5]] = z  # cx, cy, w, h

    def center(self):
        return float(self.x[0]), float(self.x[1])

    def bbox(self):
        return self.x[[0,1,4,5]].copy()
