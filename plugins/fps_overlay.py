# =============================
# Example Plugin: FPS Overlay
# =============================
"""
Пример плагина для Kinect 360 3D Viewer.
Показывает overlay с FPS и количеством точек в углу экрана.
"""
import imgui
import struct
import logging

from plugin_api import KinectPlugin

log = logging.getLogger(__name__)


class FPSOverlayPlugin(KinectPlugin):
    name = "FPS Overlay"
    version = "1.0"
    author = "Kinect 360"
    description = "Показывает FPS и кол-во точек поверх 3D-сцены"

    def __init__(self):
        super().__init__()
        self._show_overlay = True
        self._point_count = 0
        self._corner = 1  # 0=top-left, 1=top-right, 2=bot-left, 3=bot-right
        self._readback_timer = 0.0
        self._READBACK_INTERVAL = 0.5  # read GPU buffer max 2×/sec

    def on_frame_start(self, app, dt):
        # Throttled readback — GPU buffer read causes pipeline stall
        self._readback_timer += dt
        if self._readback_timer >= self._READBACK_INTERVAL:
            self._readback_timer = 0.0
            try:
                data = app.indirect_buf.read(4)
                self._point_count = struct.unpack('I', data[:4])[0]
            except Exception:
                self._point_count = 0

    def on_draw_ui(self, app):
        if not self._show_overlay:
            return

        # Позиция overlay в углу
        pad = 10.0
        if self._corner == 0:
            imgui.set_next_window_position(pad, pad, imgui.FIRST_USE_EVER)
        elif self._corner == 1:
            imgui.set_next_window_position(
                app.width - 180, pad, imgui.FIRST_USE_EVER)
        elif self._corner == 2:
            imgui.set_next_window_position(
                pad, app.height - 80, imgui.FIRST_USE_EVER)
        else:
            imgui.set_next_window_position(
                app.width - 180, app.height - 80, imgui.FIRST_USE_EVER)

        imgui.set_next_window_bg_alpha(0.35)
        flags = (imgui.WINDOW_NO_DECORATION
                 | imgui.WINDOW_ALWAYS_AUTO_RESIZE
                 | imgui.WINDOW_NO_SAVED_SETTINGS
                 | imgui.WINDOW_NO_FOCUS_ON_APPEARING
                 | imgui.WINDOW_NO_NAV)

        imgui.begin("##fps_overlay", flags=flags)
        imgui.text(f"FPS: {app.fps}")
        imgui.text(f"Points: {self._point_count:,}")
        imgui.end()

    def get_settings(self):
        return {
            "show_overlay": self._show_overlay,
            "corner": self._corner,
        }

    def set_settings(self, data):
        self._show_overlay = data.get("show_overlay", True)
        self._corner = data.get("corner", 1)
