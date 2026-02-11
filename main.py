# =============================
# KinectForge 3D Viewer — Application
# =============================
"""
Refactored entry point.

The monolithic ~2 000-line script has been split into focused modules:

  ``app_state.py``     — ``AppState`` dataclass (replaces 30+ globals)
  ``kinect_capture.py`` — ``KinectCapture`` (sensor I/O, reconnect)
  ``gpu_pipeline.py``   — ``GPUPipeline``  (compute shaders, SSBOs)
  ``renderer.py``       — ``Renderer``     (programs, FBOs, draw)
  ``ui_manager.py``     — ``UIManager``    (ImGui panels, presets)

This file contains the thin ``Application`` class: window creation,
event loop, and delegation to the subsystems above.

Original code preserved verbatim in ``main_old.py`` for reference.
"""

import logging
import time
import ctypes
import os
import threading

import numpy as np
import cv2
import glfw
import moderngl
import imgui
from imgui.integrations.glfw import GlfwRenderer

import config
from config import (
    DEPTH_W, DEPTH_H, TARGET_FPS, settings,
    POINTCLOUD_PRESETS, KINECT_ABS_MIN_CM,
)
from camera import OrbitCamera
from app_state import AppState
from kinect_capture import KinectCapture
from gpu_pipeline import GPUPipeline
from renderer import Renderer
from ui_manager import UIManager
from plugin_api import PluginManager, AppContext
from perf_metrics import PerfMetrics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── Directories ──
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PRESETS_DIR = os.path.join(_BASE_DIR, "presets")
_SCREENSHOTS_DIR = os.path.join(_BASE_DIR, "screenshots")
_EXPORTS_DIR = os.path.join(_BASE_DIR, "exports")
_PLUGINS_DIR = os.path.join(_BASE_DIR, "plugins")
for d in (_PRESETS_DIR, _SCREENSHOTS_DIR, _EXPORTS_DIR, _PLUGINS_DIR):
    os.makedirs(d, exist_ok=True)

FRAME_TIME = 1.0 / TARGET_FPS
_DEPTH_HIST_INTERVAL = 0.15
_KINECT_RECONNECT_INTERVAL = 3.0

# ── Windows high-resolution timer (1 ms instead of default ~15.6 ms) ──
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

_SPIN_THRESHOLD = 0.0015  # switch from sleep to spin-wait at 1.5 ms remaining


class Application:
    """Top-level application object.  Owns all subsystems."""

    def __init__(self):
        # ── State (replaces all module-level globals) ──
        self.state = AppState()

        # ── Kinect ──
        self.kinect = KinectCapture()

        # ── Plugins ──
        self.plugin_mgr = PluginManager(_PLUGINS_DIR)
        self.plugin_mgr.discover_and_load()
        self.app_ctx = AppContext()

        # ── GLFW + OpenGL ──
        if not glfw.init():
            log.critical("GLFW init failed")
            self.kinect.close()
            raise SystemExit(1)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(
            1280, 720, "KinectForge 3D", None, None)
        if not self.window:
            glfw.terminate()
            self.kinect.close()
            raise SystemExit(1)

        glfw.make_context_current(self.window)

        # ── VSync: попробовать включить, иначе фоллбэк на frame_limiter ──
        glfw.swap_interval(settings["vsync"])
        if settings["vsync"]:
            # Узнаём refresh rate монитора для порога проверки
            _mon = glfw.get_primary_monitor()
            _mode = glfw.get_video_mode(_mon)
            _hz = _mode.refresh_rate if _mode else 60
            # Ожидаемый vblank: 1/hz. Порог = 40% от vblank
            # (180 Hz → 5.56ms, порог 2.2ms; 144 Hz → 6.94ms, порог 2.8ms)
            _expected_vblank = 1.0 / max(_hz, 30)
            _threshold = _expected_vblank * 0.4

            # Показываем окно и делаем прогревочные кадры с glClear,
            # чтобы драйвер начал реально блокировать на vblank.
            import ctypes as _ct
            _gl = _ct.windll.opengl32
            _glClear = _gl.glClear
            _GL_COLOR_BUFFER_BIT = 0x00004000
            glfw.show_window(self.window)
            # 4 кадра прогрева (без замера)
            for _ in range(4):
                _glClear(_GL_COLOR_BUFFER_BIT)
                glfw.swap_buffers(self.window)
            # 6 кадров замера
            _MEASURE = 6
            _t0 = time.perf_counter()
            for _ in range(_MEASURE):
                _glClear(_GL_COLOR_BUFFER_BIT)
                glfw.swap_buffers(self.window)
            _swap_dt = (time.perf_counter() - _t0) / _MEASURE
            # Если swap < порога — драйвер игнорирует VSync
            if _swap_dt < _threshold:
                log.warning("VSync requested but driver did not apply it "
                            "(swap=%.1fms, expected>=%.1fms @%dHz). "
                            "Falling back to frame_limiter.",
                            _swap_dt * 1000, _expected_vblank * 1000, _hz)
                settings["vsync"] = 0
                glfw.swap_interval(0)
                settings["frame_limiter"] = 1
            else:
                log.info("VSync active (swap=%.1fms @%dHz)",
                         _swap_dt * 1000, _hz)
        self._last_vsync = settings["vsync"]

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # ── ImGui ──
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

        # From this point, failures must clean up GL resources.
        try:
            # ── Subsystems ──
            self.camera = OrbitCamera()
            self.gpu = GPUPipeline(self.ctx)
            self.renderer = Renderer(self.ctx, self.gpu)
            self.ui = UIManager(_PRESETS_DIR, _SCREENSHOTS_DIR, _EXPORTS_DIR)

            # ── Performance metrics ──
            self.metrics = PerfMetrics(self.ctx)

            # ── Scroll callback chain ──
            self._imgui_scroll_cb = self.impl.scroll_callback

            def _scroll(w, xoff, yoff):
                self.state.scroll_accum += yoff
                self._imgui_scroll_cb(w, xoff, yoff)

            glfw.set_scroll_callback(self.window, _scroll)

            # ── Plugin AppContext ──
            ac = self.app_ctx
            ac.ctx = self.ctx
            ac.window = self.window
            ac.camera = self.camera
            ac.settings = settings
            ac.sensor = self.kinect.sensor
            ac.pc_ssbo = self.gpu.pc_ssbo
            ac.indirect_buf = self.gpu.indirect_buf

            # ── Presets ──
            UIManager.refresh_preset_list(self.state, _PRESETS_DIR)

            # ── Init plugins ──
            self.plugin_mgr.call_init(self.app_ctx)
        except Exception:
            log.exception("Error during Application init — cleaning up")
            self._cleanup()
            raise

    # ────────────────────── Main loop ──────────────────────

    def run(self):
        s = self.state
        try:
            while not glfw.window_should_close(self.window):
                now = time.perf_counter()
                dt = now - s.prev_frame_time
                s.prev_frame_time = now
                self.metrics.begin_frame()

                # Update app context
                ac = self.app_ctx
                ac.time = now
                ac.dt = dt
                ac.fps = s.current_fps

                # FPS
                s.fps_counter += 1
                if now - s.fps_timer >= 1.0:
                    s.current_fps = s.fps_counter
                    s.fps_counter = 0
                    s.fps_timer = now

                # Events
                self.metrics.begin("events")
                glfw.poll_events()
                self.impl.process_inputs()
                self.plugin_mgr.call_frame_start(ac, dt)

                if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    break

                self._handle_hotkeys(now)
                self._handle_mouse()
                self.metrics.end("events")

                # ── Deferred GPU readback (from PREVIOUS frame) ──
                # Placed here so GPU has a full frame of CPU work to finish.
                self.metrics.begin("readback")
                self.gpu.deferred_reads(s)
                self.metrics.end("readback")

                self.metrics.begin("kinect")
                self._handle_kinect(now)
                self.metrics.end("kinect")

                # ── GPU pipeline ──
                new_frame = s.new_kinect_frame
                self.metrics.begin("gpu_upload")
                self.gpu.upload_frame(s, new_frame)
                self.metrics.end("gpu_upload")

                self.metrics.begin("compute")
                self.metrics.begin_gpu("compute")
                self.gpu.dispatch_compute(s, self.camera, now, new_frame)
                self.metrics.end_gpu("compute")
                self.metrics.end("compute")

                # ── Video / depth textures ──
                self.metrics.begin("tex_upload")
                if settings["show_video_feed"] and new_frame:
                    self.renderer.update_video_texture(s.curr_color)
                if settings["show_depth_feed"] and new_frame and s.curr_depth is not None:
                    self.renderer.update_depth_texture(s.curr_depth)
                self.metrics.end("tex_upload")

                # ── 3D render ──
                w, h = glfw.get_framebuffer_size(self.window)
                if w < 1 or h < 1:
                    continue
                ac.width = w
                ac.height = h
                self.plugin_mgr.call_pre_render(ac)
                self.metrics.begin("render")
                self.metrics.begin_gpu("render")
                self.renderer.render(
                    w, h, self.camera, self.gpu, now, s.start_time)
                self.metrics.end_gpu("render")
                self.metrics.end("render")
                self.plugin_mgr.call_post_render(ac)

                # ── Screenshot ──
                if s.screenshot_requested:
                    s.screenshot_requested = False
                    try:
                        s.screenshot_data = self.renderer.capture_screenshot(
                            self.window)
                    except Exception as e:
                        log.error("Screenshot capture error: %s", e)
                        s.screenshot_data = None
                    s.screenshot_flash = 0.15
                    self.renderer.gl_reset_state()

                # ── ImGui ──
                self.metrics.begin("imgui")
                imgui.new_frame()
                self.ui.draw(
                    s, self.camera, self.kinect, self.gpu,
                    self.renderer, self.plugin_mgr, ac, now, w, h,
                    self.metrics)
                imgui.render()
                self.impl.render(imgui.get_draw_data())
                self.metrics.end("imgui")

                self.metrics.begin("swap")
                glfw.swap_buffers(self.window)
                self.metrics.end("swap")

                # ── VSync: apply changes dynamically ──
                if settings["vsync"] != self._last_vsync:
                    glfw.swap_interval(settings["vsync"])
                    self._last_vsync = settings["vsync"]

                # ── Frame limiter (active when vsync is off) ──
                if settings["frame_limiter"] and not settings["vsync"]:
                    target = now + FRAME_TIME
                    # Sleep for bulk of wait (minus spin margin)
                    sleep_time = target - time.perf_counter() - _SPIN_THRESHOLD
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    # Spin-wait for remaining sub-ms precision
                    while time.perf_counter() < target:
                        pass

                # ── Deferred saves (background thread) ──
                if s.screenshot_data is not None:
                    _img = s.screenshot_data
                    s.screenshot_data = None
                    threading.Thread(
                        target=self.renderer.save_screenshot_to_disk,
                        args=(_img, _SCREENSHOTS_DIR),
                        daemon=True,
                    ).start()

                if s.screenshot_flash > 0:
                    s.screenshot_flash = max(
                        s.screenshot_flash - dt, 0.0)

                self.metrics.end_frame()

        finally:
            self._cleanup()

    # ────────────────────── Hotkeys ──────────────────────

    def _handle_hotkeys(self, now):
        s = self.state
        w = self.window

        # F11 — fullscreen
        f11 = glfw.get_key(w, glfw.KEY_F11) == glfw.PRESS
        if f11 and not s.f11_was_pressed:
            self._toggle_fullscreen()
        s.f11_was_pressed = f11

        # F12 — screenshot
        f12 = glfw.get_key(w, glfw.KEY_F12) == glfw.PRESS
        if f12 and not s.f12_was_pressed:
            s.screenshot_requested = True
        s.f12_was_pressed = f12

        # Home — camera reset
        home = glfw.get_key(w, glfw.KEY_HOME) == glfw.PRESS
        if home and not s.home_was_pressed:
            self.camera.reset()
        s.home_was_pressed = home

        # P — PLY export
        io = imgui.get_io()
        p_now = glfw.get_key(w, glfw.KEY_P) == glfw.PRESS
        if (p_now and not s.p_was_pressed
                and not io.want_capture_keyboard):
            result = self.gpu.export_ply(
                _EXPORTS_DIR, self.plugin_mgr, self.app_ctx)
            if result[0] is not None:
                s.ply_status_msg = f"Exported {len(result[0]):,} pts"
                s.ply_status_time = time.perf_counter()
        s.p_was_pressed = p_now

    # ────────────────────── Mouse ──────────────────────

    def _handle_mouse(self):
        s = self.state
        io = imgui.get_io()
        mx, my = glfw.get_cursor_pos(self.window)

        if not io.want_capture_mouse:
            if s.mouse_initialized:
                dx = mx - s.last_mx
                dy = my - s.last_my
                if glfw.get_mouse_button(
                        self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                    self.camera.rotate(dx, dy)
                if glfw.get_mouse_button(
                        self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
                    self.camera.pan(dx, dy)
            if s.scroll_accum != 0.0:
                self.camera.zoom(s.scroll_accum)

        s.scroll_accum = 0.0
        s.last_mx, s.last_my = mx, my
        s.mouse_initialized = True

    # ────────────────────── Kinect frame ──────────────────────

    def _handle_kinect(self, now):
        s = self.state
        s.new_kinect_frame = False

        # Status polling
        self.kinect.poll_status(now)
        s.kinect_connected = self.kinect.connected

        # Reconnect attempts (poll_status above already checks;
        # only retry on the reconnect interval — no duplicate call)
        if not self.kinect.connected:
            if now - s.kinect_reconnect_timer > _KINECT_RECONNECT_INTERVAL:
                self.kinect.try_reconnect()
                s.kinect_reconnect_timer = now
                # Update plugin context in case sensor object changed
                self.app_ctx.sensor = self.kinect.sensor
            return

        # Acquire frame
        result = self.kinect.acquire_frame(now)
        if result is None:
            s.kinect_connected = self.kinect.connected
            return

        depth_array, color_array = result

        # Downsample + optional bilateral filter (encapsulated in GPU pipeline)
        dd, dc = self.gpu.preprocess_frame(
            s, depth_array, color_array, settings["bilateral_filter"])

        s.prev_depth_down = s.curr_depth_down
        s.prev_color_down = s.curr_color_down
        s.curr_depth_down = dd
        s.curr_color_down = dc
        s.curr_depth = depth_array
        s.curr_color = color_array
        s.last_kinect_time = now
        s.new_kinect_frame = True
        s.kinect_connected = True

        # Plugin hook
        self.plugin_mgr.call_kinect_frame(
            self.app_ctx, depth_array, color_array)

        # Depth histogram (throttled)
        if now - s.depth_hist_update_timer > _DEPTH_HIST_INTERVAL:
            dmin_mm = settings["depth_min_cm"] * 10
            dmax_mm = settings["depth_max_cm"] * 10
            # Use .ravel() (view, no copy) and pass range to
            # np.histogram directly — avoids 2 temp boolean arrays
            # from the previous (hd > dmin) & (hd < dmax) masking.
            hd = dd.ravel()
            s.depth_histogram[:], _ = np.histogram(
                hd, bins=64, range=(dmin_mm, dmax_mm))
            hmax = s.depth_histogram.max()
            if hmax > 0:
                s.depth_histogram *= (1.0 / hmax)
            else:
                s.depth_histogram[:] = 0
            s.depth_hist_update_timer = now

    # ────────────────────── Fullscreen ──────────────────────

    def _toggle_fullscreen(self):
        s = self.state
        if s.is_fullscreen:
            glfw.set_window_monitor(
                self.window, None,
                s.windowed_pos[0], s.windowed_pos[1],
                s.windowed_size[0], s.windowed_size[1], 0)
            s.is_fullscreen = False
        else:
            s.windowed_pos = list(glfw.get_window_pos(self.window))
            s.windowed_size = list(glfw.get_window_size(self.window))
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self.window, monitor, 0, 0,
                mode.size.width, mode.size.height,
                mode.refresh_rate)
            s.is_fullscreen = True

    # ────────────────────── Cleanup ──────────────────────

    def _cleanup(self):
        # Release in reverse init order; guard each in case init failed partway.
        if hasattr(self, 'plugin_mgr') and hasattr(self, 'app_ctx'):
            self.plugin_mgr.call_cleanup(self.app_ctx)
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        if hasattr(self, 'impl'):
            self.impl.shutdown()
        if hasattr(self, 'renderer'):
            self.renderer.release()
        if hasattr(self, 'gpu'):
            self.gpu.release()
        if hasattr(self, 'ctx'):
            self.ctx.release()
        glfw.terminate()
        if hasattr(self, 'kinect'):
            self.kinect.close()


# ── Entry point ──
if __name__ == "__main__":
    app = Application()
    app.run()
