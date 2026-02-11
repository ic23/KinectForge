# =============================
# KinectCapture — Kinect v1 sensor management
# =============================
"""
Encapsulates all Microsoft Kinect v1 (Xbox 360) hardware interaction:
  - Sensor discovery and stream initialisation
  - Zero-copy C#→numpy frame acquisition
  - Reconnection on power loss / USB disconnect
  - Resolution, near-mode, and elevation changes

All C# interop (``clr``, ``System``, ``Marshal``) is confined to this
module so the rest of the application never touches .NET objects directly.
"""

import logging
import time
import ctypes
import numpy as np
import cv2

import clr
import System
from System.Runtime.InteropServices import Marshal

import config
from config import (
    KINECT_DLL_PATH, KINECT_DEPTH_PRESETS,
    ELEVATION_ANGLE, ELEVATION_MIN, ELEVATION_MAX,
    settings,
)

log = logging.getLogger(__name__)

# ── .NET Kinect SDK assembly ──
if not KINECT_DLL_PATH.exists():
    log.critical("Microsoft.Kinect.dll not found: %s", KINECT_DLL_PATH)
    raise SystemExit(1)

clr.AddReference(str(KINECT_DLL_PATH))
from Microsoft.Kinect import (
    KinectSensor, DepthImageFormat, ColorImageFormat, DepthRange,
)

# Depth formats indexed to match KINECT_DEPTH_PRESETS
_DEPTH_FORMATS = [
    DepthImageFormat.Resolution640x480Fps30,
    DepthImageFormat.Resolution320x240Fps30,
    DepthImageFormat.Resolution80x60Fps30,
]
_COLOR_FORMAT = ColorImageFormat.RgbResolution640x480Fps30
_COLOR_W, _COLOR_H = 640, 480

# Reconnect tuning
_MAX_FRAME_ERRORS = 30       # ~1 second of consecutive failures
_STATUS_CHECK_INTERVAL = 0.5


class KinectCapture:
    """Manages Kinect v1 sensor: streams, frames, reconnect."""

    def __init__(self):
        if KinectSensor.KinectSensors.Count == 0:
            log.critical("Kinect not detected! Connect sensor and restart.")
            raise SystemExit(1)

        self.sensor = KinectSensor.KinectSensors[0]
        if self.sensor is None:
            log.critical("Kinect sensor found but unavailable")
            raise SystemExit(1)

        self._wait_for_connection()
        self._start_streams(settings.get("kinect_depth_preset", 0))

        if ELEVATION_ANGLE is not None:
            self.apply_elevation(
                max(ELEVATION_MIN, min(ELEVATION_MAX, ELEVATION_ANGLE)))

        # ── Pre-allocated C# buffers ──
        self._depth_pixel_count = config.DEPTH_W * config.DEPTH_H
        self._color_pixel_count = _COLOR_W * _COLOR_H * 4
        self._depth_cs_buf = System.Array.CreateInstance(
            System.Int16, self._depth_pixel_count)
        self._color_cs_buf = System.Array.CreateInstance(
            System.Byte, self._color_pixel_count)

        # ── Pre-allocated numpy work buffers (zero-allocation hot path) ──
        self._depth_np_buf = np.empty(
            self._depth_pixel_count * 2, dtype=np.uint8)
        self._color_np_buf = np.empty(self._color_pixel_count, dtype=np.uint8)
        self._depth_i16_buf = np.empty(
            self._depth_pixel_count, dtype=np.int16)
        self._color_bgr_buf = np.empty(
            (_COLOR_H, _COLOR_W, 3), dtype=np.uint8)

        # ── Pre-allocated resize buffer (avoids alloc when depth≠color res) ──
        if (_COLOR_W, _COLOR_H) != (config.DEPTH_W, config.DEPTH_H):
            self._color_resized_buf = np.empty(
                (config.DEPTH_H, config.DEPTH_W, 3), dtype=np.uint8)
        else:
            self._color_resized_buf = None

        # ── Double-buffered output arrays (avoid .copy() per frame) ──
        self._depth_out = [
            np.empty((config.DEPTH_H, config.DEPTH_W), dtype=np.uint16),
            np.empty((config.DEPTH_H, config.DEPTH_W), dtype=np.uint16),
        ]
        self._color_out = [
            np.empty((_COLOR_H, _COLOR_W, 3), dtype=np.uint8),
            np.empty((_COLOR_H, _COLOR_W, 3), dtype=np.uint8),
        ]
        self._out_idx = 0  # ping-pong index

        # ── Connection state ──
        self.connected = True
        self._frame_errors = 0
        self._grace_until = 0.0
        self._last_status = "Connected"
        self._status_check_timer = 0.0

    # ──────────────────── Public API ────────────────────

    def acquire_frame(self, now: float):
        """Try to read one depth+color frame pair from the sensor.

        Returns ``(depth_u16, color_bgr)`` on success, ``None`` otherwise.

        *  ``depth_u16`` — ``(H, W)`` uint16 array, depth in **millimetres**.
        *  ``color_bgr`` — ``(H, W, 3)`` uint8 array, BGR colour.
        """
        try:
            depth_frame = self.sensor.DepthStream.OpenNextFrame(0)
        except Exception:
            depth_frame = None
            self._frame_errors += 1
            if self.connected and now > self._grace_until:
                if self._frame_errors >= _MAX_FRAME_ERRORS:
                    self.connected = False
                    self._frame_errors = 0
                    log.warning("Kinect: connection lost")
            return None

        if depth_frame is None:
            return None

        # ── Copy depth (in-place, zero extra allocations) ──
        depth_frame.CopyPixelDataTo(self._depth_cs_buf)
        depth_frame.Dispose()
        ptr = Marshal.UnsafeAddrOfPinnedArrayElement(self._depth_cs_buf, 0)
        ctypes.memmove(
            self._depth_np_buf.ctypes.data, ptr.ToInt64(),
            self._depth_pixel_count * 2)

        # Bit-shift to extract real depth (Kinect v1 packs player index in lower 3 bits)
        np.copyto(self._depth_i16_buf, self._depth_np_buf.view(np.int16))
        np.right_shift(self._depth_i16_buf, 3, out=self._depth_i16_buf)
        # Double-buffer: copy into pre-allocated output array (no alloc)
        depth_array = self._depth_out[self._out_idx]
        np.copyto(
            depth_array,
            self._depth_i16_buf.view(np.uint16).reshape(
                (config.DEPTH_H, config.DEPTH_W)))

        # ── Copy colour (always 640×480 BGRA) ──
        color_frame = self.sensor.ColorStream.OpenNextFrame(0)
        if color_frame is None:
            return None
        color_frame.CopyPixelDataTo(self._color_cs_buf)
        color_frame.Dispose()
        ptr = Marshal.UnsafeAddrOfPinnedArrayElement(self._color_cs_buf, 0)
        ctypes.memmove(
            self._color_np_buf.ctypes.data, ptr.ToInt64(),
            self._color_pixel_count)

        # BGRA→BGR using pre-allocated buffer
        cv2.cvtColor(
            self._color_np_buf.reshape((_COLOR_H, _COLOR_W, 4)),
            cv2.COLOR_BGRA2BGR, dst=self._color_bgr_buf)
        # Double-buffer: copy into pre-allocated output array (no alloc)
        color_array = self._color_out[self._out_idx]
        np.copyto(color_array, self._color_bgr_buf)

        # Resize colour to match depth if resolutions differ
        if (_COLOR_W, _COLOR_H) != (config.DEPTH_W, config.DEPTH_H):
            if self._color_resized_buf is not None:
                cv2.resize(
                    color_array, (config.DEPTH_W, config.DEPTH_H),
                    dst=self._color_resized_buf)
                color_array = self._color_resized_buf
            else:
                color_array = cv2.resize(
                    color_array, (config.DEPTH_W, config.DEPTH_H))

        self.connected = True
        self._frame_errors = 0
        self._out_idx ^= 1  # flip ping-pong for next frame
        return depth_array, color_array

    def poll_status(self, now: float):
        """Check sensor status for power loss / USB disconnect.
        Call every frame; internally throttled to ~2 Hz."""
        if now - self._status_check_timer < _STATUS_CHECK_INTERVAL:
            return
        self._status_check_timer = now

        try:
            status_str = str(self.sensor.Status)
        except Exception:
            status_str = "Error"

        if status_str == self._last_status:
            return

        old = self._last_status
        self._last_status = status_str

        if status_str != "Connected" and self.connected:
            self.connected = False
            log.warning("Kinect: %s → %s (connection lost)", old, status_str)
        elif status_str == "Connected" and not self.connected:
            log.info("Kinect: %s → Connected", old)
            self.try_reconnect()

    def try_reconnect(self):
        """Restart Kinect streams using the current depth preset."""
        log.info("Kinect reconnecting: restarting streams...")
        try:
            idx = settings.get("kinect_depth_preset", 0)
            self.apply_resolution(idx)
            self._frame_errors = 0
            self._grace_until = time.perf_counter() + 2.0
            log.info("Kinect streams restored (preset %d)", idx)
        except Exception as e:
            log.error("Kinect reconnect failed: %s", e)
            self.connected = False

    def apply_resolution(self, preset_idx: int):
        """Change depth resolution.  Requires full Stop → Enable → Start.

        Returns ``True`` on success.  Caller should also:
        - call ``Renderer.on_kinect_resolution_changed()``
        - call ``State.reset_frame_data()``
        """
        preset_idx = max(0, min(preset_idx, len(KINECT_DEPTH_PRESETS) - 1))
        _, new_dw, new_dh, label = KINECT_DEPTH_PRESETS[preset_idx]

        try:
            self.sensor.Stop()
            time.sleep(0.3)  # give SDK time to release resources

            self.sensor.DepthStream.Enable(_DEPTH_FORMATS[preset_idx])
            self.sensor.ColorStream.Enable(_COLOR_FORMAT)
            self.sensor.Start()
            time.sleep(0.2)

            config.DEPTH_W = new_dw
            config.DEPTH_H = new_dh

            # Reallocate buffers for new depth resolution
            self._depth_pixel_count = new_dw * new_dh
            self._depth_cs_buf = System.Array.CreateInstance(
                System.Int16, self._depth_pixel_count)
            self._depth_np_buf = np.empty(
                self._depth_pixel_count * 2, dtype=np.uint8)
            self._depth_i16_buf = np.empty(
                self._depth_pixel_count, dtype=np.int16)
            # Reallocate double-buffered output arrays
            self._depth_out = [
                np.empty((new_dh, new_dw), dtype=np.uint16),
                np.empty((new_dh, new_dw), dtype=np.uint16),
            ]
            # Reallocate resize buffer for new depth resolution
            if (_COLOR_W, _COLOR_H) != (new_dw, new_dh):
                self._color_resized_buf = np.empty(
                    (new_dh, new_dw, 3), dtype=np.uint8)
            else:
                self._color_resized_buf = None
            # Note: colour C# buffers stay the same size (always 640×480)

            # Restore depth range
            if settings["near_mode"]:
                self.sensor.DepthStream.Range = DepthRange.Near
            else:
                self.sensor.DepthStream.Range = DepthRange.Default

            self.connected = True
            settings["kinect_depth_preset"] = preset_idx
            log.info("Kinect depth: %s", label)
            return True
        except Exception as e:
            log.error("Resolution change error: %s", e)
            self.connected = False
            return False

    def apply_near_mode(self, val: int):
        """Toggle near mode (0.4–3.0 m range)."""
        settings["near_mode"] = val
        try:
            self.sensor.DepthStream.Range = (
                DepthRange.Near if val else DepthRange.Default)
            if val:
                log.info("Near mode: ON (0.4–3.0 m)")
                if settings["depth_min_cm"] > 40:
                    settings["depth_min_cm"] = 40
            else:
                log.info("Near mode: OFF (0.8–4.0 m)")
                if settings["depth_min_cm"] < 80:
                    settings["depth_min_cm"] = 80
        except (System.InvalidOperationException, Exception):
            log.warning("Near mode not supported by this sensor")
            settings["near_mode"] = 0

    def apply_elevation(self, val: int):
        """Set sensor tilt angle (−27 to +27 degrees)."""
        settings["elevation"] = val
        try:
            self.sensor.ElevationAngle = val
            log.info("Elevation angle: %d°", val)
        except Exception:
            log.warning("Failed to set elevation: %d", val)

    def close(self):
        """Stop Kinect sensor.  Must be called on shutdown."""
        try:
            self.sensor.Stop()
        except Exception:
            pass

    # ──────────────────── Internal ────────────────────

    def _wait_for_connection(self, timeout: float = 10.0):
        """Block until sensor reports Connected status (or timeout → exit)."""
        status = str(self.sensor.Status)
        log.info("Kinect Status: %s", status)
        if status == "Connected":
            return

        log.info("Waiting for Kinect connection (max %.0fs)...", timeout)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            time.sleep(0.5)
            status = str(self.sensor.Status)
            if status == "Connected":
                log.info("Kinect connected!")
                return
            log.info("  Status: %s ...", status)

        log.critical(
            "Kinect not ready (Status: %s). Check:\n"
            "  - Power supply connected to Kinect\n"
            "  - USB cable connected to PC\n"
            "  - Kinect SDK v1.8 drivers installed", status)
        raise SystemExit(1)

    def _start_streams(self, depth_preset: int = 0):
        """Enable depth + colour streams and start sensor."""
        depth_preset = max(0, min(depth_preset, len(_DEPTH_FORMATS) - 1))
        self.sensor.DepthStream.Enable(_DEPTH_FORMATS[depth_preset])
        self.sensor.ColorStream.Enable(_COLOR_FORMAT)
        self.sensor.Start()

        try:
            mode = DepthRange.Near if settings.get("near_mode") else DepthRange.Default
            self.sensor.DepthStream.Range = mode
            log.info("Depth Range: %s",
                     "Near" if mode == DepthRange.Near else "Default")
        except System.InvalidOperationException:
            log.warning("Near mode not supported, using Default")
            self.sensor.DepthStream.Range = DepthRange.Default

        log.info("Depth format: %s", _DEPTH_FORMATS[depth_preset])
