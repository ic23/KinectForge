# =============================
# Конфигурация Kinect 360
# =============================
from pathlib import Path
from typing import List, TypedDict

# Путь к Kinect SDK
KINECT_DLL_PATH = Path(r"C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies\Microsoft.Kinect.dll")

# Разрешение depth/color
DEPTH_W, DEPTH_H = 640, 480

# Параметры камер Kinect v1 (intrinsics для 640x480)
# IR/Depth камера (FOV ~57°×43°)
FX_DEPTH, FY_DEPTH = 580.0, 580.0
CX_DEPTH, CY_DEPTH = 319.5, 239.5
# RGB камера (FOV ~62°×49°, шире!)
FX_COLOR, FY_COLOR = 525.0, 525.0
CX_COLOR, CY_COLOR = 319.5, 239.5

# Downsampling для point cloud
DOWN_W, DOWN_H = 640, 480

# Пресеты разрешений
# Kinect depth: (формат_индекс, ширина, высота, label)
KINECT_DEPTH_PRESETS = [
    (0, 640, 480, "640x480"),
    (1, 320, 240, "320x240"),
    (2, 80,  60,  "80x60"),
]

# Point cloud resolution presets
POINTCLOUD_PRESETS = [
    (640, 480, "640x480 (Full)"),
    (320, 240, "320x240"),
    (160, 120, "160x120"),
    (80,  60,  "80x60"),
]

# Лимиты глубины
DEPTH_MIN_M = 0.8
DEPTH_MAX_M = 4.0
MIN_GAP_CM = 10
KINECT_ABS_MIN_CM = 40
KINECT_ABS_MAX_CM = 400

# Угол наклона (None = не менять)
ELEVATION_ANGLE = None
ELEVATION_MIN = -27
ELEVATION_MAX = 27

# Рендеринг
TARGET_FPS = 144
MAX_POINTS = 1_000_000

# Ghost Particles
GHOST_TTL = 14       # время жизни призрачных точек (кадры)
GHOST_SAMPLE = 0.08  # доля точек, сохраняемых с каждого кадра

# Cyberspace grid / noise
GRID_Y_FLOOR = -0.85
GRID_SPACING = 0.25
GRID_DENSITY = 0.03
GRID_X_RANGE = 2.5
GRID_COLOR = 0.25
NOISE_COUNT = 400

# Noise particle bounds
NOISE_X_RANGE = (-2.2, 2.2)
NOISE_Y_RANGE = (-1.3, 0.9)

# Temporal smoothing
TEMPORAL_SNAP_THRESHOLD = 150.0  # мм — порог «скачка» для snap-копирования

# Post-processing defaults
GHOST_DECAY = 0.92
DRIP_SPEED = 0.0015
FRUSTUM_MARGIN = 0.15

# Trail freeze probability & ranges
TRAIL_FREEZE_CHANCE = 0.03
TRAIL_FREEZE_N_RANGE = (15, 60)
TRAIL_FREEZE_LIFE_RANGE = (0.8, 2.5)

# =============================
# Глобальные настройки (runtime)
# =============================


class Settings(TypedDict):
    """Typed schema for the runtime settings dict.

    Using ``int`` for boolean toggles (0/1) to match ImGui checkbox
    and GLSL uniform conventions.
    """
    # Depth range
    depth_min_cm: int
    depth_max_cm: int
    # Kinect hardware
    elevation: int
    near_mode: int
    # Rendering basics
    point_size: int
    point_shape: int          # 0 = circle, 1 = square
    bg_color: List[float]     # [r, g, b] 0‒1
    # Cyberspace & effects (int 0/1 toggles)
    cyberspace: int
    cyber_invert: int
    cyber_ghosts: int
    cyber_glitch_bands: int
    cyber_glitch_color: int
    cyber_glitch_double: int
    cyber_drip_trails: int
    cyber_particle_trails: int
    cyber_grid: int
    cyber_noise: int
    cyber_ghost_particles: int
    cyber_jitter: int
    # Cluster coloring
    cluster_coloring: int
    cluster_blend: float
    # Depth scaling
    depth_scale_points: int
    depth_scale_factor: float
    # SSAO
    ssao_enabled: int
    ssao_strength: float
    # Bloom
    bloom_enabled: int
    bloom_strength: float
    # Filters
    bilateral_filter: int
    edge_filter: int
    edge_filter_threshold: float
    temporal_smooth: int
    temporal_alpha: float
    frustum_culling: int
    # Resolution presets (indices)
    kinect_depth_preset: int
    pointcloud_preset: int
    # Wave distortion
    wave_distortion: int
    wave_amplitude: float
    wave_frequency: float
    # Voxelize
    voxelize: int
    voxel_size: float
    # Pulse
    pulse: int
    pulse_speed: float
    # Chromatic aberration
    chromatic_aberration: int
    chromatic_strength: float
    # Edge glow
    edge_glow: int
    edge_glow_strength: float
    # Color palette (0=off, 1=thermal, 2=nightvision, 3=retro)
    color_palette: int
    # Pixelate
    pixelate: int
    pixelate_size: float
    # Depth‑color alignment
    depth_color_align: int
    baseline_x_mm: float
    baseline_y_mm: float
    # Feed windows
    show_video_feed: int
    show_depth_feed: int
    # Frame limiter
    frame_limiter: int
    # VSync
    vsync: int


settings: Settings = {
    "depth_min_cm": int(DEPTH_MIN_M * 100),
    "depth_max_cm": int(DEPTH_MAX_M * 100),
    "elevation": 0,
    "near_mode": 0,
    "point_size": 4,
    "point_shape": 0,
    "cyberspace": 0,
    "cyber_invert": 0,
    "cyber_ghosts": 0,
    "cyber_glitch_bands": 0,
    "cyber_glitch_color": 0,
    "cyber_glitch_double": 0,
    "cyber_drip_trails": 0,
    "cyber_particle_trails": 0,
    "cyber_grid": 0,
    "cyber_noise": 0,
    "cyber_ghost_particles": 0,
    "cyber_jitter": 0,
    "cluster_coloring": 0,
    "cluster_blend": 0.6,
    "bg_color": [0.0, 0.0, 0.0],
    "depth_scale_points": 0,
    "depth_scale_factor": 0.5,
    "ssao_enabled": 0,
    "ssao_strength": 0.8,
    "bloom_enabled": 0,
    "bloom_strength": 0.5,
    "bilateral_filter": 1,
    "edge_filter": 1,
    "edge_filter_threshold": 50.0,  # мм — порог разрыва глубины на границах
    "temporal_smooth": 0,
    "temporal_alpha": 0.3,
    "frustum_culling": 1,
    "kinect_depth_preset": 0,    # индекс в KINECT_DEPTH_PRESETS (0=640x480)
    "pointcloud_preset": 0,      # индекс в POINTCLOUD_PRESETS (0=640x480 Full)
    # Новые эффекты
    "chromatic_aberration": 0,
    "chromatic_strength": 2.0,
    "wave_distortion": 0,
    "wave_amplitude": 0.03,
    "wave_frequency": 4.0,
    "voxelize": 0,
    "voxel_size": 0.04,
    "pulse": 0,
    "pulse_speed": 2.0,
    "edge_glow": 0,
    "edge_glow_strength": 1.5,
    "color_palette": 0,       # 0=off, 1=thermal, 2=nightvision, 3=retro
    "pixelate": 0,
    "pixelate_size": 8.0,
    # Depth-to-Color alignment (перепроецирование IR→RGB)
    "depth_color_align": 1,
    "baseline_x_mm": 10.4,     # горизонтальное смещение IR↔RGB, мм
    "baseline_y_mm": -10.9,    # вертикальное смещение IR↔RGB, мм
    "show_video_feed": 0,      # показывать окно видеопотока
    "show_depth_feed": 0,      # показывать окно карты глубины
    "frame_limiter": 0,        # ограничение FPS (фоллбэк если VSync недоступен)
    "vsync": 1,                # VSync вкл по умолчанию
}
