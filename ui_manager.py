# =============================
# UIManager — ImGui panels and presets
# =============================
"""
All ImGui drawing code lives here: settings panel, presets,
export buttons, plugin list, video/depth feed windows.

Keeps ``main.py`` free of draw-call clutter.
"""

import logging
import os
import json
import time

import imgui
import numpy as np

from config import (
    KINECT_DEPTH_PRESETS, POINTCLOUD_PRESETS,
    KINECT_ABS_MIN_CM, KINECT_ABS_MAX_CM,
    settings,
)

log = logging.getLogger(__name__)


class UIManager:
    """Manages ImGui panels, presets, and input helpers."""

    def __init__(self, presets_dir, screenshots_dir, exports_dir):
        self._presets_dir = presets_dir
        self._screenshots_dir = screenshots_dir
        self._exports_dir = exports_dir

    # ────────────────── Settings helpers ──────────────────

    @staticmethod
    def _checkbox(label, key, plugin_mgr, app_ctx):
        """imgui checkbox bound to settings[key]. Returns True if changed."""
        changed, val = imgui.checkbox(label, bool(settings[key]))
        if changed:
            old_val = settings[key]
            settings[key] = int(val)
            plugin_mgr.call_settings_changed(
                app_ctx, key, old_val, int(val))
        return changed

    @staticmethod
    def _slider_float(label, key, v_min, v_max,
                      plugin_mgr, app_ctx, fmt="%.1f"):
        """imgui slider_float bound to settings[key]."""
        changed, val = imgui.slider_float(
            label, settings[key], v_min, v_max, fmt)
        if changed:
            old_val = settings[key]
            settings[key] = val
            plugin_mgr.call_settings_changed(
                app_ctx, key, old_val, val)
        return changed

    # ────────────────── Presets ──────────────────

    @staticmethod
    def refresh_preset_list(state, presets_dir):
        state.preset_names = sorted([
            f[:-5] for f in os.listdir(presets_dir)
            if f.endswith(".json")
        ])

    @staticmethod
    def save_preset(name, presets_dir, plugin_mgr, app_ctx, state):
        path = os.path.join(presets_dir, f"{name}.json")
        data = {k: v for k, v in settings.items()}
        data.update(plugin_mgr.collect_plugin_settings())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        plugin_mgr.call_preset_save(app_ctx, name)
        log.info("Preset saved: %s", path)
        UIManager.refresh_preset_list(state, presets_dir)
        state.preset_status_msg = f"Saved: {name}"
        state.preset_status_time = time.perf_counter()

    @staticmethod
    def load_preset(name, presets_dir, plugin_mgr, app_ctx, kinect, state):
        path = os.path.join(presets_dir, f"{name}.json")
        if not os.path.exists(path):
            log.warning("Preset not found: %s", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k in settings:
                settings[k] = v
        plugin_mgr.apply_plugin_settings(data)
        kinect.apply_elevation(settings["elevation"])
        kinect.apply_near_mode(settings["near_mode"])
        state.enforce_depth_gap()
        plugin_mgr.call_preset_load(app_ctx, name)
        log.info("Preset loaded: %s", name)
        state.preset_status_msg = f"Loaded: {name}"
        state.preset_status_time = time.perf_counter()

    @staticmethod
    def delete_preset(name, presets_dir, state):
        path = os.path.join(presets_dir, f"{name}.json")
        if os.path.exists(path):
            os.remove(path)
            log.info("Preset deleted: %s", name)
        UIManager.refresh_preset_list(state, presets_dir)
        state.preset_status_msg = f"Deleted: {name}"
        state.preset_status_time = time.perf_counter()

    # ────────────────── Main draw ──────────────────

    def draw(self, state, camera, kinect, gpu, renderer,
             plugin_mgr, app_ctx, now, w, h, metrics=None):
        """Draw all ImGui panels.  Delegates to _draw_*() sub-methods."""
        if not state.show_ui:
            return

        pm = plugin_mgr
        ac = app_ctx
        cb = lambda l, k: self._checkbox(l, k, pm, ac)
        sf = lambda l, k, lo, hi, fmt="%.1f": self._slider_float(
            l, k, lo, hi, pm, ac, fmt)

        # ── Styles ──
        imgui.push_style_color(
            imgui.COLOR_TITLE_BACKGROUND_ACTIVE, 0.78, 0.55, 0.20, 1.0)
        imgui.push_style_color(
            imgui.COLOR_SLIDER_GRAB, 0.78, 0.55, 0.20, 1.0)
        imgui.push_style_color(
            imgui.COLOR_SLIDER_GRAB_ACTIVE, 0.90, 0.67, 0.27, 1.0)
        imgui.push_style_color(
            imgui.COLOR_CHECK_MARK, 0.31, 0.75, 0.31, 1.0)
        imgui.push_style_color(
            imgui.COLOR_FRAME_BACKGROUND, 0.20, 0.20, 0.20, 1.0)

        imgui.begin("KinectPyEffects Settings")

        self._draw_depth_section(state, kinect, cb)
        self._draw_resolution_section(state, kinect, renderer, cb)
        self._draw_rendering_section(cb, sf)
        self._draw_effects_section(cb, sf)
        self._draw_status_section(state, now, pm, ac, kinect, gpu)
        self._draw_presets_section(state, pm, ac, kinect, now)
        self._draw_export_section(state, gpu, pm, ac, now)
        self._draw_plugins_section(pm, ac)
        if metrics is not None:
            self._draw_metrics_section(metrics)

        # ── Plugin custom UI panels (inside main window so
        #    collapsing_header-based plugins render inline) ──
        pm.call_draw_ui(ac)

        imgui.separator()
        imgui.text("ESC - quit | F10 - toggle UI | F11 - fullscreen")
        imgui.text("F12 - screenshot | Home - reset cam")
        imgui.text("P - export PLY")
        imgui.text("LMB - orbit | RMB - pan | Scroll - zoom")
        imgui.end()

        # ── Feed windows (outside main panel) ──
        if settings["show_video_feed"] and state.curr_depth is not None:
            self._draw_feed_window(
                "Video Feed", renderer.video_texture, w, h)
        if settings["show_depth_feed"] and state.curr_depth is not None:
            self._draw_feed_window(
                "Video Depth", renderer.depth_feed_texture, w, h)

        imgui.pop_style_color(5)

    # ────────────────── Section: Depth range & sensor ──────────────────

    @staticmethod
    def _draw_depth_section(state, kinect, cb):
        depth_min_limit = (KINECT_ABS_MIN_CM
                           if settings["near_mode"] else 80)
        changed, val = imgui.slider_int(
            "Depth Min (cm)", settings["depth_min_cm"],
            depth_min_limit, KINECT_ABS_MAX_CM)
        if changed:
            settings["depth_min_cm"] = max(val, depth_min_limit)
            state.enforce_depth_gap()
        imgui.text(f"= {settings['depth_min_cm'] / 100:.2f} m")

        changed, val = imgui.slider_int(
            "Depth Max (cm)", settings["depth_max_cm"],
            KINECT_ABS_MIN_CM, KINECT_ABS_MAX_CM)
        if changed:
            settings["depth_max_cm"] = val
            state.enforce_depth_gap()
        imgui.text(f"= {settings['depth_max_cm'] / 100:.2f} m")
        imgui.separator()

        # Elevation
        changed, val = imgui.slider_int(
            "Elevation", settings["elevation"], -27, 27)
        if changed:
            kinect.apply_elevation(val)
        imgui.text(f"{settings['elevation']} deg")
        imgui.separator()

        # Near Mode
        changed, val = imgui.checkbox(
            "Near Mode", bool(settings["near_mode"]))
        if changed:
            kinect.apply_near_mode(int(val))
        if settings["near_mode"]:
            imgui.same_line()
            imgui.push_style_color(
                imgui.COLOR_TEXT, 0.31, 0.75, 0.31, 1.0)
            imgui.text("ON (0.4-3.0 m)")
            imgui.pop_style_color()
        imgui.separator()

    # ────────────────── Section: Resolution ──────────────────

    @staticmethod
    def _draw_resolution_section(state, kinect, renderer, cb):
        # Kinect Depth Resolution
        imgui.text("Kinect Depth:")
        kp = settings["kinect_depth_preset"]
        for i, (_, _, _, label) in enumerate(KINECT_DEPTH_PRESETS):
            if i > 0:
                imgui.same_line()
            clicked = imgui.radio_button(f"{label}##kinect", kp == i)
            if clicked and i != kp:
                kinect.apply_resolution(i)
                renderer.on_kinect_resolution_changed()
                state.reset_frame_data()

        # Point Cloud Resolution
        imgui.text("Point Cloud:")
        pp = settings["pointcloud_preset"]
        for i, (_, _, label) in enumerate(POINTCLOUD_PRESETS):
            if i > 0:
                imgui.same_line()
            clicked = imgui.radio_button(f"{label}##pc", pp == i)
            if clicked and i != pp:
                pp_idx = max(0, min(i, len(POINTCLOUD_PRESETS) - 1))
                nw, nh, plbl = POINTCLOUD_PRESETS[pp_idx]
                state.down_w = nw
                state.down_h = nh
                settings["pointcloud_preset"] = pp_idx
                state.reset_frame_data()
                log.info("Point cloud resolution: %s", plbl)
        imgui.text(f"~{state.down_w * state.down_h:,} pts max")

        cb("Video Feed", "show_video_feed")
        cb("Video Depth", "show_depth_feed")
        imgui.separator()

    # ────────────────── Section: Rendering options ──────────────────

    @staticmethod
    def _draw_rendering_section(cb, sf):
        # Point Size
        changed, val = imgui.slider_int(
            "Point Size", settings["point_size"], 1, 10)
        if changed:
            settings["point_size"] = val

        # Point Shape
        if imgui.radio_button("Circle", settings["point_shape"] == 0):
            settings["point_shape"] = 0
        imgui.same_line()
        if imgui.radio_button("Square", settings["point_shape"] == 1):
            settings["point_shape"] = 1

        # Background Color
        changed, color = imgui.color_edit3(
            "Background", *settings["bg_color"])
        if changed:
            settings["bg_color"] = list(color)
        imgui.separator()

        # Depth Scaling
        cb("Depth Scale", "depth_scale_points")
        if settings["depth_scale_points"]:
            sf("Scale Factor", "depth_scale_factor", 0.5, 4.0)

        # SSAO
        cb("Point Shading", "ssao_enabled")
        if settings["ssao_enabled"]:
            sf("AO Strength", "ssao_strength", 0.1, 2.0)

        # Bloom
        cb("Bloom", "bloom_enabled")
        if settings["bloom_enabled"]:
            sf("Glow", "bloom_strength", 0.1, 2.0)

        # Filters
        cb("Smooth Depth", "bilateral_filter")
        cb("Edge Filter", "edge_filter")
        if settings["edge_filter"]:
            sf("Edge Threshold (mm)", "edge_filter_threshold",
               10.0, 200.0, "%.0f")
        cb("Temporal Smooth", "temporal_smooth")
        if settings["temporal_smooth"]:
            sf("Smooth Alpha", "temporal_alpha", 0.05, 1.0, "%.2f")
        cb("Frustum Culling", "frustum_culling")

        # GPU Pipeline status
        imgui.push_style_color(
            imgui.COLOR_TEXT, 1.0, 0.65, 0.0, 1.0)
        imgui.text("GPU Pipeline: ACTIVE")
        imgui.pop_style_color()

        # Depth-Color Alignment
        cb("Depth-Color Align", "depth_color_align")
        if settings["depth_color_align"]:
            sf("Baseline X mm", "baseline_x_mm", -50.0, 50.0)
            sf("Baseline Y mm", "baseline_y_mm", -50.0, 50.0)
            imgui.same_line()
            imgui.text("(?)")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Perspective + parallax correction.\n"
                    "Corrects different FOV between IR & RGB cameras\n"
                    "and physical offset between them.")
        imgui.separator()

        # Cluster Coloring
        cb("Cluster Coloring", "cluster_coloring")
        if settings["cluster_coloring"]:
            imgui.same_line()
            imgui.push_style_color(
                imgui.COLOR_TEXT, 0.15, 0.85, 0.95, 1.0)
            imgui.text("NEAR")
            imgui.pop_style_color()
            imgui.same_line()
            imgui.push_style_color(
                imgui.COLOR_TEXT, 0.95, 0.30, 0.20, 1.0)
            imgui.text("/ FAR")
            imgui.pop_style_color()
            sf("Blend", "cluster_blend", 0.1, 1.0, "%.2f")

    # ────────────────── Section: Effects ──────────────────

    @staticmethod
    def _draw_effects_section(cb, sf):
        cb("Ghost Particles", "cyber_ghost_particles")
        cb("Particle Trails", "cyber_particle_trails")
        cb("Jitter", "cyber_jitter")
        cb("Glitch Bands", "cyber_glitch_bands")
        cb("Double Glitch", "cyber_glitch_double")
        cb("Ghost Render", "cyber_ghosts")
        cb("Render Trails", "cyber_drip_trails")
        cb("Grid Floor", "cyber_grid")
        cb("Noise Particles", "cyber_noise")

        # Wave Distortion
        cb("Wave Distortion", "wave_distortion")
        if settings["wave_distortion"]:
            sf("Amplitude", "wave_amplitude", 0.005, 0.15, "%.3f")
            sf("Frequency", "wave_frequency", 0.5, 20.0)

        # Voxelize
        cb("Voxelize", "voxelize")
        if settings["voxelize"]:
            sf("Voxel Size", "voxel_size", 0.005, 0.15, "%.3f")

        # Pulse
        cb("Pulse", "pulse")
        if settings["pulse"]:
            sf("Pulse Speed", "pulse_speed", 0.5, 8.0)

        # Chromatic Aberration
        cb("Chromatic Aberration", "chromatic_aberration")
        if settings["chromatic_aberration"]:
            sf("CA Strength", "chromatic_strength", 0.5, 8.0)

        # Edge Glow
        cb("Edge Glow", "edge_glow")
        if settings["edge_glow"]:
            sf("Glow Strength##edge", "edge_glow_strength", 0.3, 5.0)

        # Pixelate
        cb("Pixelate", "pixelate")
        if settings["pixelate"]:
            sf("Pixel Size", "pixelate_size", 1.0, 40.0, "%.0f")

        # Color Palette
        imgui.text("Color Palette:")
        for i, pname in enumerate(
                ["Off", "Thermal", "Night Vision", "Retro Amber"]):
            if i > 0:
                imgui.same_line()
            if imgui.radio_button(
                    f"{pname}##palette", settings["color_palette"] == i):
                settings["color_palette"] = i
        imgui.separator()

        # Cyberspace
        cb("Cyberspace", "cyberspace")
        if settings["cyberspace"]:
            imgui.same_line()
            imgui.push_style_color(
                imgui.COLOR_TEXT, 1.0, 0.0, 0.6, 1.0)
            imgui.text("NETRUNNER")
            imgui.pop_style_color()
            cb("Invert Colors", "cyber_invert")
            cb("Color Glitch", "cyber_glitch_color")
        imgui.separator()

    # ────────────────── Section: Status ──────────────────

    @staticmethod
    def _draw_status_section(state, now, pm, ac, kinect, gpu):
        imgui.text(f"FPS: {state.current_fps}")
        imgui.text(f"Points: ~{state.down_w * state.down_h:,} (GPU)")

        if state.kinect_connected:
            imgui.push_style_color(
                imgui.COLOR_TEXT, 0.31, 0.75, 0.31, 1.0)
            imgui.text("Kinect: CONNECTED")
            imgui.pop_style_color()
        else:
            imgui.push_style_color(
                imgui.COLOR_TEXT, 0.95, 0.30, 0.20, 1.0)
            imgui.text("Kinect: DISCONNECTED (reconnecting...)")
            imgui.pop_style_color()

        # Depth histogram
        if imgui.collapsing_header("Depth Histogram")[0]:
            dmin_cm = settings["depth_min_cm"]
            dmax_cm = settings["depth_max_cm"]
            imgui.plot_histogram(
                "##depth_hist", state.depth_histogram,
                graph_size=(
                    imgui.get_content_region_available()[0], 60))
            imgui.text(f"{dmin_cm} cm")
            imgui.same_line(
                spacing=imgui.get_content_region_available()[0] - 60)
            imgui.text(f"{dmax_cm} cm")

    # ────────────────── Section: Presets ──────────────────

    def _draw_presets_section(self, state, pm, ac, kinect, now):
        if not imgui.collapsing_header("Presets")[0]:
            return

        changed, state.preset_name_buf = imgui.input_text(
            "Name##preset", state.preset_name_buf, 64)
        imgui.same_line()
        if imgui.button("Save##preset"):
            name = state.preset_name_buf.strip()
            if name:
                self.save_preset(
                    name, self._presets_dir, pm, ac, state)
                state.preset_name_buf = ""

        if state.preset_names:
            if state.selected_preset_idx >= len(state.preset_names):
                state.selected_preset_idx = 0
            changed, state.selected_preset_idx = imgui.combo(
                "##preset_list", state.selected_preset_idx,
                state.preset_names)
            imgui.same_line()
            if imgui.button("Load##preset"):
                self.load_preset(
                    state.preset_names[state.selected_preset_idx],
                    self._presets_dir, pm, ac, kinect, state)
            imgui.same_line()
            if imgui.button("Delete##preset"):
                self.delete_preset(
                    state.preset_names[state.selected_preset_idx],
                    self._presets_dir, state)
        else:
            imgui.text("No presets saved yet")

        if (state.preset_status_msg
                and (now - state.preset_status_time < 3.0)):
            imgui.push_style_color(
                imgui.COLOR_TEXT, 0.31, 0.75, 0.31, 1.0)
            imgui.text(state.preset_status_msg)
            imgui.pop_style_color()

    # ────────────────── Section: Export ──────────────────

    def _draw_export_section(self, state, gpu, pm, ac, now):
        if not imgui.collapsing_header("Export")[0]:
            return

        if imgui.button("Screenshot (F12)"):
            state.screenshot_requested = True
        imgui.same_line()
        if imgui.button("Export PLY (P)"):
            result = gpu.export_ply(
                self._exports_dir, pm, ac)
            if result[0] is not None:
                state.ply_status_msg = (
                    f"Exported {len(result[0]):,} pts")
                state.ply_status_time = time.perf_counter()
            else:
                state.ply_status_msg = "Export failed"
                state.ply_status_time = time.perf_counter()

        if (state.ply_status_msg
                and (now - state.ply_status_time < 4.0)):
            imgui.push_style_color(
                imgui.COLOR_TEXT, 0.31, 0.75, 0.31, 1.0)
            imgui.text(state.ply_status_msg)
            imgui.pop_style_color()

    # ────────────────── Section: Plugins ──────────────────

    @staticmethod
    def _draw_plugins_section(pm, ac):
        if not imgui.collapsing_header("Plugins")[0]:
            return

        p_list = pm.get_plugin_list()
        if p_list:
            imgui.text(f"{pm.active_count}/{pm.count} active")
            for pi, pinfo in enumerate(p_list):
                p_changed, p_en = imgui.checkbox(
                    f"{pinfo['name']} v{pinfo['version']}##plg{pi}",
                    pinfo['enabled'])
                if p_changed:
                    pm.set_enabled(pinfo['name'], p_en)
                if pinfo['description']:
                    imgui.same_line()
                    imgui.text_disabled("(?)")
                    if imgui.is_item_hovered():
                        imgui.begin_tooltip()
                        imgui.text(pinfo['description'])
                        if pinfo['author']:
                            imgui.text(f"Author: {pinfo['author']}")
                        imgui.end_tooltip()
        else:
            imgui.text_disabled("No plugins found")
        p_errors = pm.get_load_errors()
        if p_errors:
            imgui.push_style_color(
                imgui.COLOR_TEXT, 0.95, 0.30, 0.20, 1.0)
            for pe in p_errors:
                imgui.text(f"! {pe}")
            imgui.pop_style_color()
        if imgui.button("Reload Plugins"):
            pm.reload_all(ac)

    # ────────────────── Section: Performance Metrics ──────────────────

    @staticmethod
    def _draw_metrics_section(metrics):
        expanded, _ = imgui.collapsing_header("Performance")
        if not expanded:
            # Still show toggle even when collapsed
            _, metrics.enabled = imgui.checkbox(
                "Enable Profiler", metrics.enabled)
            return

        _, metrics.enabled = imgui.checkbox(
            "Enable Profiler", metrics.enabled)
        if not metrics.enabled:
            imgui.text_disabled("Enable to see detailed metrics")
            return

        if imgui.button("Reset##perf"):
            metrics.reset()

        avail_w = imgui.get_content_region_available()[0]

        # ── Frame time sparkline ──
        imgui.separator()
        imgui.text(f"Frame: {metrics.frame_avg_ms:.2f} ms avg"
                   f"  |  {metrics.frame_max_ms:.2f} ms max"
                   f"  |  1%% low: {metrics.fps_1pct_low:.0f} FPS")
        imgui.plot_lines(
            "##frame_time", metrics.frame_history,
            overlay_text="frame ms",
            graph_size=(avail_w, 40),
            scale_min=0.0, scale_max=max(metrics.frame_max_ms * 1.2, 1.0))

        # ── CPU sections ──
        cpu_sections = metrics.get_cpu_sections()
        if cpu_sections:
            imgui.separator()
            imgui.text("CPU Sections:")
            total_cpu = 0.0
            for name, avg_ms, _ in cpu_sections:
                total_cpu += avg_ms
                # Color bar proportional to 16.6 ms budget
                frac = min(avg_ms / 16.6, 1.0)
                if frac > 0.6:
                    imgui.push_style_color(
                        imgui.COLOR_PLOT_HISTOGRAM, 0.95, 0.30, 0.20, 1.0)
                elif frac > 0.3:
                    imgui.push_style_color(
                        imgui.COLOR_PLOT_HISTOGRAM, 0.95, 0.75, 0.20, 1.0)
                else:
                    imgui.push_style_color(
                        imgui.COLOR_PLOT_HISTOGRAM, 0.31, 0.75, 0.31, 1.0)
                imgui.progress_bar(
                    frac, (avail_w * 0.5, 14),
                    f"{avg_ms:.2f} ms")
                imgui.pop_style_color()
                imgui.same_line()
                imgui.text(name)
            imgui.text(f"  Total CPU: {total_cpu:.2f} ms")

        # ── GPU sections ──
        gpu_sections = metrics.get_gpu_sections()
        if gpu_sections:
            imgui.separator()
            imgui.text("GPU Sections:")
            total_gpu = 0.0
            for name, avg_ms, _ in gpu_sections:
                total_gpu += avg_ms
                frac = min(avg_ms / 16.6, 1.0)
                if frac > 0.6:
                    imgui.push_style_color(
                        imgui.COLOR_PLOT_HISTOGRAM, 0.95, 0.30, 0.20, 1.0)
                elif frac > 0.3:
                    imgui.push_style_color(
                        imgui.COLOR_PLOT_HISTOGRAM, 0.95, 0.75, 0.20, 1.0)
                else:
                    imgui.push_style_color(
                        imgui.COLOR_PLOT_HISTOGRAM, 0.31, 0.75, 0.31, 1.0)
                imgui.progress_bar(
                    frac, (avail_w * 0.5, 14),
                    f"{avg_ms:.2f} ms")
                imgui.pop_style_color()
                imgui.same_line()
                imgui.text(name)
            imgui.text(f"  Total GPU: {total_gpu:.2f} ms")

    @staticmethod
    def _draw_feed_window(title, texture, w, h):
        aspect = 4.0 / 3.0
        default_w = max(160, int(w * 0.25))
        default_h = int(default_w / aspect)
        min_w = 160
        min_h = int(min_w / aspect)

        imgui.set_next_window_size(
            default_w + 16, default_h + 38, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size_constraints(
            (min_w + 16, min_h + 38), (w, h))
        imgui.begin(title,
                     flags=(imgui.WINDOW_NO_SCROLLBAR
                            | imgui.WINDOW_NO_SCROLL_WITH_MOUSE))
        avail_w, avail_h = imgui.get_content_region_available()
        fit_w = avail_w
        fit_h = fit_w / aspect
        if fit_h > avail_h:
            fit_h = avail_h
            fit_w = fit_h * aspect
        imgui.image(texture.glo, fit_w, fit_h)
        imgui.end()
