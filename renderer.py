# =============================
# Renderer — OpenGL programs, FBOs, scene rendering
# =============================
"""
Owns vertex/fragment shader programs, framebuffers, VAOs and VBOs
for the 3D viewport, post-processing pipeline, and video/depth
feed textures.

The ``Renderer`` never touches Kinect hardware or compute shaders;
it renders whatever the ``GPUPipeline`` has produced into the SSBO.
"""

import logging
import os
from datetime import datetime

import numpy as np
import cv2
import moderngl
import glfw

import config
from config import (
    DEPTH_W, DEPTH_H, settings,
    GRID_Y_FLOOR, GRID_SPACING, GRID_DENSITY, GRID_X_RANGE, GRID_COLOR,
    NOISE_COUNT, NOISE_X_RANGE, NOISE_Y_RANGE,
    GHOST_DECAY, DRIP_SPEED,
)
from shaders import (
    VERTEX_SHADER, FRAGMENT_SHADER,
    FULLSCREEN_VERTEX_SHADER, ACCUM_FRAGMENT_SHADER,
    COMPOSITE_FRAGMENT_SHADER,
)
import pointcloud as _pc_module

log = logging.getLogger(__name__)


class Renderer:
    """OpenGL rendering — programs, FBOs, VAOs, scene + post-processing."""

    def __init__(self, ctx: moderngl.Context, gpu):
        """
        Parameters
        ----------
        ctx : moderngl.Context
        gpu : GPUPipeline
            Provides ``pc_ssbo`` and ``indirect_buf`` for the point cloud VAO.
        """
        self.ctx = ctx

        # ── Main shader program ──
        self.prog = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER)

        # ── Post-processing programs ──
        self.accum_prog = ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=ACCUM_FRAGMENT_SHADER)
        self.composite_prog = ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=COMPOSITE_FRAGMENT_SHADER)

        # ── Screen quad VBO (single overdraw triangle) ──
        self._screen_quad_vbo = ctx.buffer(np.array([
            -1.0, -1.0, 0.0, 0.0,
             3.0, -1.0, 2.0, 0.0,
            -1.0,  3.0, 0.0, 2.0,
        ], dtype='f4'))
        self._accum_vao = ctx.vertex_array(
            self.accum_prog,
            [(self._screen_quad_vbo, '2f 2f', 'in_pos', 'in_uv')])
        self._composite_vao = ctx.vertex_array(
            self.composite_prog,
            [(self._screen_quad_vbo, '2f 2f', 'in_pos', 'in_uv')])

        # ── Point cloud VAO (bound to GPU pipeline's SSBO) ──
        self.pc_vao = ctx.vertex_array(
            self.prog,
            [(gpu.pc_ssbo, '3f 3f', 'in_position', 'in_color')])

        # ── Post-processing FBOs ──
        self._scene_fbo = None
        self._scene_color_tex = None
        self._scene_depth_rb = None
        self._accum_fbo = [None, None]
        self._accum_color_tex = [None, None]
        self._postproc_size = (0, 0)
        self._accum_read = 0
        self._accum_write = 1

        # ── Grid floor VBO ──
        self._grid_vbo = ctx.buffer(reserve=50000 * 6 * 4)
        self._grid_vao = ctx.vertex_array(
            self.prog,
            [(self._grid_vbo, '3f 3f', 'in_position', 'in_color')])
        self._n_grid = 0
        self._grid_key = (0, 0)

        # ── Noise particles VBO ──
        self._noise_vbo = ctx.buffer(reserve=NOISE_COUNT * 6 * 4)
        self._noise_vao = ctx.vertex_array(
            self.prog,
            [(self._noise_vbo, '3f 3f', 'in_position', 'in_color')])
        self._n_noise = 0
        self._noise_key = (0, 0)

        # ── Video feed texture ──
        self.video_texture = ctx.texture((DEPTH_W, DEPTH_H), 3)
        self.video_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._vid_rgb_buf = np.empty((DEPTH_H, DEPTH_W, 3), dtype=np.uint8)

        # ── Depth feed texture ──
        self.depth_feed_texture = ctx.texture((DEPTH_W, DEPTH_H), 3)
        self.depth_feed_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._depth_rgb_buf = np.empty((DEPTH_H, DEPTH_W, 3), dtype=np.uint8)
        self._depth_f32_buf = np.empty((DEPTH_H, DEPTH_W), dtype=np.float32)
        self._depth_u8_buf = np.empty((DEPTH_H, DEPTH_W), dtype=np.uint8)

    # ────────────────── Resolution change ──────────────────

    def on_kinect_resolution_changed(self):
        """Recreate video/depth feed textures after Kinect resolution change."""
        w, h = config.DEPTH_W, config.DEPTH_H

        self.video_texture.release()
        self.video_texture = self.ctx.texture((w, h), 3)
        self.video_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._vid_rgb_buf = np.empty((h, w, 3), dtype=np.uint8)

        self.depth_feed_texture.release()
        self.depth_feed_texture = self.ctx.texture((w, h), 3)
        self.depth_feed_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._depth_rgb_buf = np.empty((h, w, 3), dtype=np.uint8)
        self._depth_f32_buf = np.empty((h, w), dtype=np.float32)
        self._depth_u8_buf = np.empty((h, w), dtype=np.uint8)

    # ────────────────── Post-processing FBOs ──────────────────

    def _ensure_postproc_fbos(self, w, h):
        if self._postproc_size == (w, h) and self._scene_fbo is not None:
            return
        for obj in [self._scene_fbo, self._scene_color_tex,
                     self._scene_depth_rb]:
            if obj is not None:
                obj.release()
        for i in range(2):
            if self._accum_fbo[i] is not None:
                self._accum_fbo[i].release()
                self._accum_color_tex[i].release()

        self._scene_color_tex = self.ctx.texture((w, h), 4)
        self._scene_color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._scene_depth_rb = self.ctx.depth_renderbuffer((w, h))
        self._scene_fbo = self.ctx.framebuffer(
            color_attachments=[self._scene_color_tex],
            depth_attachment=self._scene_depth_rb)
        for i in range(2):
            self._accum_color_tex[i] = self.ctx.texture((w, h), 4)
            self._accum_color_tex[i].filter = (
                moderngl.NEAREST, moderngl.NEAREST)
            self._accum_fbo[i] = self.ctx.framebuffer(
                color_attachments=[self._accum_color_tex[i]])
        self._postproc_size = (w, h)

    # ────────────────── Grid / noise VBO updates ──────────────────

    def _update_grid_vbo(self, d_min, d_max):
        key = (round(d_min, 2), round(d_max, 2))
        if self._grid_key == key and self._n_grid > 0:
            return
        x_pts = np.arange(-GRID_X_RANGE, GRID_X_RANGE, GRID_DENSITY,
                          dtype=np.float32)
        z_pts = np.arange(-d_max, -d_min, GRID_DENSITY,
                          dtype=np.float32)

        # Horizontal lines: vectorised meshgrid replaces Python for-loop
        z_lines = np.arange(-d_max, -d_min, GRID_SPACING, dtype=np.float32)
        x_lines = np.arange(-GRID_X_RANGE, GRID_X_RANGE, GRID_SPACING,
                            dtype=np.float32)

        parts = []
        if len(z_lines) > 0 and len(x_pts) > 0:
            # All horizontal grid lines at once via repeat/tile
            n_h = len(x_pts) * len(z_lines)
            hx = np.tile(x_pts, len(z_lines))
            hz = np.repeat(z_lines, len(x_pts))
            hy = np.full(n_h, GRID_Y_FLOOR, dtype=np.float32)
            parts.append(np.column_stack([hx, hy, hz]))

        if len(x_lines) > 0 and len(z_pts) > 0:
            # All vertical grid lines at once
            n_v = len(z_pts) * len(x_lines)
            vx = np.repeat(x_lines, len(z_pts))
            vz = np.tile(z_pts, len(x_lines))
            vy = np.full(n_v, GRID_Y_FLOOR, dtype=np.float32)
            parts.append(np.column_stack([vx, vy, vz]))

        if not parts:
            self._n_grid = 0
            self._grid_key = key
            return
        grid_xyz = np.vstack(parts) if len(parts) > 1 else parts[0]
        n = len(grid_xyz)
        # Build interleaved pos+color buffer in one shot
        data = np.empty((n, 6), dtype=np.float32)
        data[:, :3] = grid_xyz
        data[:, 3:] = GRID_COLOR
        self._n_grid = n
        self._grid_vbo.orphan()
        self._grid_vbo.write(data.tobytes())
        self._grid_key = key

    def _update_noise_vbo(self, d_min, d_max):
        key = (round(d_min, 2), round(d_max, 2))
        if self._noise_key == key and self._n_noise > 0:
            return
        # Build interleaved pos+color in one pre-typed array
        data = np.empty((NOISE_COUNT, 6), dtype=np.float32)
        data[:, 0] = np.random.uniform(*NOISE_X_RANGE, NOISE_COUNT)
        data[:, 1] = np.random.uniform(*NOISE_Y_RANGE, NOISE_COUNT)
        data[:, 2] = np.random.uniform(-d_max, -d_min, NOISE_COUNT)
        ni = np.random.uniform(0.06, 0.45, NOISE_COUNT).astype(
            np.float32)
        data[:, 3] = ni * 0.05
        data[:, 4] = ni * 0.55
        data[:, 5] = ni
        self._n_noise = NOISE_COUNT
        self._noise_vbo.orphan()
        self._noise_vbo.write(data.tobytes())
        self._noise_key = key

    # ────────────────── Video / depth textures ──────────────────

    def update_video_texture(self, curr_color):
        """Upload current colour frame to the video feed texture."""
        if curr_color is None:
            return
        h, w = curr_color.shape[:2]
        if self._vid_rgb_buf.shape[0] == h and self._vid_rgb_buf.shape[1] == w:
            cv2.cvtColor(curr_color, cv2.COLOR_BGR2RGB,
                         dst=self._vid_rgb_buf)
            self.video_texture.write(self._vid_rgb_buf)

    def update_depth_texture(self, curr_depth):
        """Upload current depth frame to the depth feed texture."""
        if curr_depth is None:
            return
        d_min_mm = settings["depth_min_cm"] * 10
        d_max_mm = settings["depth_max_cm"] * 10
        d_rng = max(d_max_mm - d_min_mm, 1)
        np.copyto(self._depth_f32_buf, curr_depth, casting='unsafe')
        np.clip(self._depth_f32_buf, d_min_mm, d_max_mm,
                out=self._depth_f32_buf)
        self._depth_f32_buf -= d_min_mm
        self._depth_f32_buf *= (255.0 / d_rng)
        np.copyto(self._depth_u8_buf, self._depth_f32_buf,
                  casting='unsafe')
        cv2.applyColorMap(self._depth_u8_buf, cv2.COLORMAP_INFERNO,
                          dst=self._depth_rgb_buf)
        self.depth_feed_texture.write(self._depth_rgb_buf)

    # ────────────────── Scene rendering ──────────────────

    def render(self, w, h, camera, gpu, now, start_time):
        """Full render pass: 3D scene + post-processing."""
        self.ctx.viewport = (0, 0, w, h)
        bg = settings["bg_color"]

        has_temporal = (
            bool(settings["cyber_ghosts"])
            or bool(settings["cyber_drip_trails"]))
        has_postproc = (
            bool(settings["bloom_enabled"])
            or has_temporal
            or bool(settings["cyber_glitch_double"])
            or bool(settings["chromatic_aberration"])
            or bool(settings["edge_glow"])
            or bool(settings["pixelate"]))

        # Clear accumulation when temporal effects are off
        if (not has_temporal and self._postproc_size[0] > 0
                and self._accum_fbo[0] is not None):
            for i in range(2):
                self._accum_fbo[i].use()
                self.ctx.clear(bg[0], bg[1], bg[2])

        if has_postproc:
            self._ensure_postproc_fbos(w, h)
            self._scene_fbo.use()
        else:
            self.ctx.screen.use()

        self.ctx.clear(bg[0], bg[1], bg[2])
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # ── Shader uniforms ──
        elapsed = float(now - start_time)
        d_min_m = settings["depth_min_cm"] / 100.0
        d_max_m = settings["depth_max_cm"] / 100.0
        mvp = camera.get_mvp(w, h)

        p = self.prog
        p['mvp'].write(mvp.tobytes())
        p['render_mode'].value = 0

        base_pt = float(max(settings["point_size"], 1))
        if settings["cyber_glitch_bands"] and np.random.random() < 0.025:
            pt = max(1.0, base_pt + np.random.uniform(-2.0, 4.0))
        else:
            pt = base_pt
        p['point_size'].value = pt
        p['time'].value = elapsed

        p['cyber_enabled'].value = settings["cyberspace"]
        p['cyber_invert'].value = settings["cyber_invert"]
        p['cyber_jitter'].value = settings["cyber_jitter"]
        p['cyber_glitch_bands'].value = settings["cyber_glitch_bands"]
        p['cyber_glitch_color'].value = settings["cyber_glitch_color"]
        p['depth_min'].value = d_min_m
        p['depth_max'].value = d_max_m
        p['cluster_threshold'].value = float(
            _pc_module.cluster_threshold_t)
        p['point_shape'].value = settings["point_shape"]
        p['depth_scale_points'].value = settings["depth_scale_points"]
        p['depth_scale_factor'].value = float(
            settings["depth_scale_factor"])
        p['ssao_enabled'].value = settings["ssao_enabled"]
        p['ssao_strength'].value = float(settings["ssao_strength"])

        p['wave_distortion'].value = settings["wave_distortion"]
        p['wave_amplitude'].value = float(settings["wave_amplitude"])
        p['wave_frequency'].value = float(settings["wave_frequency"])
        p['voxelize'].value = settings["voxelize"]
        p['voxel_size'].value = float(settings["voxel_size"])
        p['pulse'].value = settings["pulse"]
        p['pulse_speed'].value = float(settings["pulse_speed"])
        p['color_palette'].value = settings["color_palette"]

        # ── Draw point cloud (indirect) ──
        self.pc_vao.render_indirect(
            gpu.indirect_buf, moderngl.POINTS, count=1)

        # ── Grid floor ──
        if settings["cyber_grid"]:
            self._update_grid_vbo(d_min_m, d_max_m)
            if self._n_grid > 0:
                p['render_mode'].value = 2
                self._grid_vao.render(
                    moderngl.POINTS, vertices=self._n_grid)

        # ── Noise particles ──
        if settings["cyber_noise"]:
            self._update_noise_vbo(d_min_m, d_max_m)
            if self._n_noise > 0:
                p['render_mode'].value = 1
                self._noise_vao.render(
                    moderngl.POINTS, vertices=self._n_noise)

        # ── Post-processing pipeline ──
        if has_postproc:
            self.ctx.disable(moderngl.DEPTH_TEST)

            if has_temporal:
                self._accum_fbo[self._accum_write].use()
                self.ctx.viewport = (0, 0, w, h)
                self.ctx.clear(0, 0, 0)
                self._scene_color_tex.use(0)
                self._accum_color_tex[self._accum_read].use(1)
                ap = self.accum_prog
                ap['current_scene'].value = 0
                ap['prev_accum'].value = 1
                ap['ghost_enabled'].value = int(
                    settings["cyber_ghosts"])
                ap['drip_enabled'].value = int(
                    settings["cyber_drip_trails"])
                ap['ghost_decay'].value = GHOST_DECAY
                ap['drip_speed'].value = DRIP_SPEED
                ap['bg_color'].value = tuple(bg)
                self._accum_vao.render(moderngl.TRIANGLES)
                source_tex = self._accum_color_tex[self._accum_write]
                self._accum_read, self._accum_write = (
                    self._accum_write, self._accum_read)
            else:
                source_tex = self._scene_color_tex

            # Composite pass → screen
            self.ctx.screen.use()
            self.ctx.viewport = (0, 0, w, h)
            source_tex.use(0)
            cp = self.composite_prog
            cp['scene_tex'].value = 0
            cp['bloom_enabled'].value = int(settings["bloom_enabled"])
            cp['bloom_strength'].value = float(
                settings["bloom_strength"])
            cp['texel_size'].value = (
                1.0 / max(w, 1), 1.0 / max(h, 1))
            cp['double_glitch_enabled'].value = int(
                settings["cyber_glitch_double"])
            cp['time'].value = elapsed
            cp['chromatic_aberration'].value = int(
                settings["chromatic_aberration"])
            cp['chromatic_strength'].value = float(
                settings["chromatic_strength"])
            cp['edge_glow'].value = int(settings["edge_glow"])
            cp['edge_glow_strength'].value = float(
                settings["edge_glow_strength"])
            cp['pixelate_enabled'].value = int(settings["pixelate"])
            cp['pixelate_size'].value = float(
                settings["pixelate_size"])
            self._composite_vao.render(moderngl.TRIANGLES)

    # ────────────────── Screenshot ──────────────────

    def capture_screenshot(self, window):
        """Capture framebuffer pixels (call BEFORE imgui render)."""
        w, h = glfw.get_framebuffer_size(window)
        data = self.ctx.screen.read(
            viewport=(0, 0, w, h), components=3, alignment=1)
        expected = w * h * 3
        if len(data) != expected:
            log.error("Screenshot: data size %d != %dx%dx3=%d",
                      len(data), w, h, expected)
            return None
        img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        img = np.flipud(img)
        # cvtColor always returns a new contiguous array — no .copy() needed
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def gl_reset_state():
        """Full GL state reset after ctx.screen.read() so
        pyopengl-imgui doesn't hit stale bindings."""
        from OpenGL import GL as gl

        # 1. Drain ALL accumulated GL errors (from moderngl read)
        try:
            _raw = gl.glGetError.wrappedOperation
        except AttributeError:
            _raw = gl.glGetError
        for _ in range(32):
            if _raw() == 0:
                break

        # 2. Reset bindings
        for fn, *args in [
            (gl.glBindFramebuffer, gl.GL_FRAMEBUFFER, 0),
            (gl.glBindTexture, gl.GL_TEXTURE_2D, 0),
            (gl.glBindVertexArray, 0),
            (gl.glBindBuffer, gl.GL_ARRAY_BUFFER, 0),
            (gl.glUseProgram, 0),
        ]:
            try:
                fn(*args)
            except Exception:
                pass

        # 3. Final drain
        for _ in range(32):
            if _raw() == 0:
                break

    @staticmethod
    def save_screenshot_to_disk(img, screenshots_dir):
        """Save captured pixels to PNG (call outside GL context)."""
        if img is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(screenshots_dir, f"kinect_{timestamp}.png")
        cv2.imwrite(path, img)
        log.info("Screenshot saved: %s (%dx%d)",
                 path, img.shape[1], img.shape[0])

    # ────────────────── Cleanup ──────────────────

    def release(self):
        """Release all GPU resources."""
        if self._scene_fbo is not None:
            self._scene_fbo.release()
            self._scene_color_tex.release()
            self._scene_depth_rb.release()
        for i in range(2):
            if self._accum_fbo[i] is not None:
                self._accum_fbo[i].release()
                self._accum_color_tex[i].release()
        if self.pc_vao is not None:
            self.pc_vao.release()
