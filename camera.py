# =============================
# Орбитальная камера
# =============================
import numpy as np


class OrbitCamera:
    def __init__(self, target=None, distance=4.0, yaw=0.0, pitch=0.0):
        self.target = target if target is not None else np.array([0.0, 0.0, -2.0], dtype='f4')
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self._cached_mvp = None
        self._cached_frustum_planes = None
    
    def reset(self):
        """Сброс камеры в начальное положение."""
        self.target = np.array([0.0, 0.0, -2.0], dtype='f4')
        self.distance = 4.0
        self.yaw = 0.0
        self.pitch = 0.0
    
    def rotate(self, dx, dy, sensitivity=0.005):
        """Вращение камеры (LMB drag)"""
        self.yaw -= dx * sensitivity
        self.pitch += dy * sensitivity
        self.pitch = np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
    
    def pan(self, dx, dy, sensitivity=0.005):
        """Панорамирование камеры (RMB drag)"""
        right = np.array([np.cos(self.yaw), 0.0, -np.sin(self.yaw)], dtype='f4')
        up = np.array([0.0, 1.0, 0.0], dtype='f4')
        self.target += right * dx * sensitivity
        self.target -= up * dy * sensitivity
    
    def zoom(self, delta, sensitivity=0.3):
        """Зум камеры (scroll)"""
        self.distance = max(0.3, self.distance - delta * sensitivity)
    
    def get_eye_position(self):
        """Позиция камеры в мировых координатах"""
        return self.target + self.distance * np.array([
            np.cos(self.pitch) * np.sin(self.yaw),
            np.sin(self.pitch),
            np.cos(self.pitch) * np.cos(self.yaw),
        ], dtype='f4')
    
    def get_mvp(self, width, height, fov=60.0, near=0.01, far=100.0):
        """Возвращает MVP матрицу для OpenGL"""
        aspect = max(width, 1) / max(height, 1)
        proj = self._perspective(np.radians(fov), aspect, near, far)
        view = self._look_at(self.get_eye_position(), self.target)
        mvp = proj @ view
        # Кэшируем плоскости фрустума (row-major)
        self._cached_frustum_planes = self._extract_frustum_planes(mvp)
        # Транспонируем: numpy row-major → OpenGL column-major
        self._cached_mvp = mvp.T.astype('f4')
        return self._cached_mvp
    
    def get_frustum_planes(self):
        """Возвращает 6 плоскостей фрустума (Nx4 array: [a,b,c,d])"""
        return self._cached_frustum_planes
    
    @staticmethod
    def _extract_frustum_planes(mvp):
        """Извлекает 6 плоскостей фрустума из MVP (row-major, до транспонирования)."""
        # Gribb-Hartmann method: row-major mvp
        planes = np.empty((6, 4), dtype='f4')
        planes[0] = mvp[3] + mvp[0]   # left
        planes[1] = mvp[3] - mvp[0]   # right
        planes[2] = mvp[3] + mvp[1]   # bottom
        planes[3] = mvp[3] - mvp[1]   # top
        planes[4] = mvp[3] + mvp[2]   # near
        planes[5] = mvp[3] - mvp[2]   # far
        # Нормализация
        norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        planes /= norms
        return planes
    
    @staticmethod
    def _perspective(fov, aspect, near, far):
        f = 1.0 / np.tan(fov / 2.0)
        nf = near - far
        return np.array([
            [f / aspect, 0.0, 0.0,                        0.0],
            [0.0,        f,   0.0,                        0.0],
            [0.0,        0.0, (far + near) / nf, 2.0 * far * near / nf],
            [0.0,        0.0, -1.0,                       0.0],
        ], dtype='f4')
    
    @staticmethod
    def _look_at(eye, target, up=None):
        if up is None:
            up = np.array([0, 1, 0], dtype='f4')
        f = np.float32(target - eye)
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        sn = np.linalg.norm(s)
        if sn < 1e-6:
            up = np.array([1, 0, 0], dtype='f4')
            s = np.cross(f, up)
            sn = np.linalg.norm(s)
        s = s / sn
        u = np.cross(s, f)
        m = np.eye(4, dtype='f4')
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(s, eye)
        m[1, 3] = -np.dot(u, eye)
        m[2, 3] = np.dot(f, eye)
        return m
