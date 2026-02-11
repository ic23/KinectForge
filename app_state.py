# =============================
# Centralized Application State
# =============================
"""
Replaces the 30+ module-level ``global`` variables that were scattered
across main.py functions.

One ``AppState`` instance is created in ``Application.__init__`` and
passed by reference to all subsystems.  Mutations are immediately
visible everywhere — no need for ``global`` declarations.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import config
from config import settings, KINECT_ABS_MIN_CM, KINECT_ABS_MAX_CM, MIN_GAP_CM


@dataclass
class AppState:
    """All mutable runtime state, grouped logically."""

    # ── Kinect connection ──
    kinect_connected: bool = True
    kinect_reconnect_timer: float = 0.0

    # ── Frame data (downsampled for point cloud) ──
    prev_depth_down: Optional[np.ndarray] = None
    prev_color_down: Optional[np.ndarray] = None
    curr_depth_down: Optional[np.ndarray] = None
    curr_color_down: Optional[np.ndarray] = None
    curr_color: Optional[np.ndarray] = None       # full-res for video feed
    curr_depth: Optional[np.ndarray] = None        # guard flag
    last_kinect_time: float = 0.0
    new_kinect_frame: bool = False                  # set per-frame by _handle_kinect

    # ── FPS tracking ──
    fps_counter: int = 0
    fps_timer: float = field(default_factory=time.perf_counter)
    current_fps: int = 0

    # ── Screenshot ──
    f12_was_pressed: bool = False
    screenshot_flash: float = 0.0
    screenshot_requested: bool = False
    screenshot_data: Optional[np.ndarray] = None

    # ── Fullscreen ──
    windowed_pos: list = field(default_factory=lambda: [100, 100])
    windowed_size: list = field(default_factory=lambda: [1280, 720])
    is_fullscreen: bool = False
    f11_was_pressed: bool = False

    # ── Camera / mouse ──
    last_mx: float = 0.0
    last_my: float = 0.0
    mouse_initialized: bool = False
    scroll_accum: float = 0.0
    p_was_pressed: bool = False
    home_was_pressed: bool = False

    # ── Presets ──
    preset_names: list = field(default_factory=list)
    selected_preset_idx: int = 0
    preset_name_buf: str = ""
    preset_status_msg: str = ""
    preset_status_time: float = 0.0

    # ── PLY export ──
    ply_status_msg: str = ""
    ply_status_time: float = 0.0

    # ── Depth histogram ──
    depth_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(64, dtype=np.float32))
    depth_hist_update_timer: float = 0.0

    # ── GPU deferred readback ──
    need_gpu_reads: bool = False
    pending_ghost_swap: bool = False
    pending_trail_swap: bool = False

    # ── Resolution (mutable — changed by UI / Kinect preset) ──
    down_w: int = 640
    down_h: int = 480

    # ── Timing ──
    start_time: float = field(default_factory=time.perf_counter)
    prev_frame_time: float = field(default_factory=time.perf_counter)

    # ── Kinect reconnect interval ──
    RECONNECT_INTERVAL: float = 3.0

    def reset_frame_data(self):
        """Reset downsampled frame buffers (after resolution change)."""
        self.prev_depth_down = None
        self.prev_color_down = None
        self.curr_depth_down = None
        self.curr_color_down = None
        self.curr_color = None
        self.curr_depth = None

    @staticmethod
    def enforce_depth_gap():
        """Maintain min + gap <= max in one pass."""
        lo = max(settings["depth_min_cm"], KINECT_ABS_MIN_CM)
        hi = max(settings["depth_max_cm"], lo + MIN_GAP_CM)
        hi = min(hi, KINECT_ABS_MAX_CM)
        lo = min(lo, hi - MIN_GAP_CM)
        settings["depth_min_cm"] = lo
        settings["depth_max_cm"] = hi
