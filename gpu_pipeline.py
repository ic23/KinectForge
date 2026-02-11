# =============================
# GPUPipeline — compute shaders, SSBOs, point cloud build
# =============================
"""
Owns the OpenGL 4.3+ compute pipeline that builds the point cloud
entirely on the GPU:

    depth texture → compute shader → SSBO → indirect draw

Zero CPU↔GPU synchronisation on the hot path.

Also manages GPU-driven visual effects:
  - Ghost Particles (double-buffered SSBO history)
  - Particle Trails  (double-buffered SSBO history)

Both effect shaders now use a ``phase`` uniform and are dispatched
in two passes with a ``glMemoryBarrier`` in between, fixing the
data-race that the original single-dispatch approach had when the
build shader's actual output was smaller than the CPU estimate
passed via ``current_point_count``.
"""

import logging
import struct
import os
import ctypes
from datetime import datetime

import numpy as np
import cv2
import moderngl

import config
from config import (
    MAX_POINTS, settings,
    GHOST_TTL, GHOST_SAMPLE,
    TEMPORAL_SNAP_THRESHOLD, FRUSTUM_MARGIN,
    TRAIL_FREEZE_CHANCE, TRAIL_FREEZE_N_RANGE, TRAIL_FREEZE_LIFE_RANGE,
)
from shaders import (
    POINTCLOUD_BUILD_COMPUTE_SHADER,
    GHOST_PARTICLES_COMPUTE_SHADER,
    PARTICLE_TRAILS_COMPUTE_SHADER,
)
import pointcloud as _pc_module

log = logging.getLogger(__name__)

# Cached constant struct packs (avoid repeated packing every frame)
_INDIRECT_RESET = struct.pack('4I', 0, 1, 0, 0)
_ZERO_U32 = struct.pack('I', 0)

MAX_GHOST_GPU = 200_000   # max ghost history entries
MAX_TRAIL_GPU = 30_000    # max trail base points


class GPUPipeline:
    """GPU compute pipeline for point cloud generation and effects."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx

        # ── Build compute shader ──
        try:
            self._build_compute = ctx.compute_shader(
                POINTCLOUD_BUILD_COMPUTE_SHADER)
        except Exception as e:
            log.critical("GPU pipeline requires OpenGL 4.3+: %s", e)
            raise SystemExit(1)

        # ── Point cloud SSBO + indirect draw buffer ──
        self.pc_ssbo = ctx.buffer(reserve=MAX_POINTS * 6 * 4)
        self.indirect_buf = ctx.buffer(data=_INDIRECT_RESET)

        # ── GPU textures for compute input (double-buffered) ──
        self._depth_tex_curr = None
        self._depth_tex_prev = None
        self._color_tex_curr = None
        self._color_tex_prev = None
        self._temporal_depth_tex = None
        self._has_prev_frame = False
        self._frame_ready = False
        self._interp_alpha = 0.0
        self._tex_size = (0, 0)
        self._prev_draw_count = 0  # Phase 1 dispatch bound (from previous frame)
        self._cluster_next_time = 0.0  # throttle cluster computation to ~5 Hz

        # Targeted memory barrier (SSBO only, not ALL bits)
        self._ssbo_barrier = (
            0x00002000    # GL_SHADER_STORAGE_BARRIER_BIT
            | 0x00000100  # GL_COMMAND_BARRIER_BIT (indirect draw)
        )

        # glFlush — push pending commands to driver without CPU stall
        self._glFlush = ctypes.windll.opengl32.glFlush

        # Uniform cache — avoid redundant ctypes calls
        self._uniform_cache = {}

        # Backward-compat aliases
        self._depth_tex = None
        self._color_tex = None

        # ── Ghost Particles ──
        self._ghost_compute = None
        self._ghost_ssbo = [None, None]
        self._ghost_count_buf = None
        self._ghost_in_count = 0
        self._ghost_read = 0
        self._ghost_write = 1

        # ── Particle Trails ──
        self._trail_compute = None
        self._trail_ssbo = [None, None]
        self._trail_count_buf = None
        self._trail_in_count = 0
        self._trail_read = 0
        self._trail_write = 1

        # Try to create ghost particles compute shader
        try:
            self._ghost_compute = ctx.compute_shader(
                GHOST_PARTICLES_COMPUTE_SHADER)
            self._ghost_ssbo[0] = ctx.buffer(
                data=bytes(MAX_GHOST_GPU * 7 * 4))
            self._ghost_ssbo[1] = ctx.buffer(
                data=bytes(MAX_GHOST_GPU * 7 * 4))
            self._ghost_count_buf = ctx.buffer(data=_ZERO_U32)
            log.info("GPU ghost particles: OK")
        except Exception as e:
            log.warning("GPU ghost particles: not available (%s)", e)
            self._ghost_compute = None

        # Try to create particle trails compute shader
        try:
            self._trail_compute = ctx.compute_shader(
                PARTICLE_TRAILS_COMPUTE_SHADER)
            self._trail_ssbo[0] = ctx.buffer(
                data=bytes(MAX_TRAIL_GPU * 8 * 4))
            self._trail_ssbo[1] = ctx.buffer(
                data=bytes(MAX_TRAIL_GPU * 8 * 4))
            self._trail_count_buf = ctx.buffer(data=_ZERO_U32)
            log.info("GPU particle trails: OK")
        except Exception as e:
            log.warning("GPU particle trails: not available (%s)", e)
            self._trail_compute = None

        # ── Pre-allocated work buffers (avoid per-frame allocations) ──
        self._interp_a = None       # float32 depth for interpolation
        self._interp_b = None       # float32 depth for interpolation
        self._dm_buf = None         # scratch depth-in-metres
        self._tsnap_buf = None      # boolean temporal snap mask
        self._cr_buf = None         # BGR→RGB colour buffer

        log.info("GPU pipeline: OK (OpenGL 4.3+)")

    # ────────────────── Texture management ──────────────────

    def ensure_textures(self, state):
        """Recreate GPU textures if point cloud resolution changed."""
        dw, dh = state.down_w, state.down_h
        if self._tex_size == (dw, dh):
            return
        # Release old textures
        for tex in (self._depth_tex_curr, self._depth_tex_prev,
                    self._color_tex_curr, self._color_tex_prev,
                    self._temporal_depth_tex):
            if tex is not None:
                tex.release()

        # Current + previous depth textures (R32F, mm)
        self._depth_tex_curr = self.ctx.texture((dw, dh), 1, dtype='f4')
        self._depth_tex_curr.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._depth_tex_prev = self.ctx.texture((dw, dh), 1, dtype='f4')
        self._depth_tex_prev.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Current + previous color textures (RGB8 — data uploaded as BGR)
        self._color_tex_curr = self.ctx.texture((dw, dh), 3)
        self._color_tex_curr.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._color_tex_prev = self.ctx.texture((dw, dh), 3)
        self._color_tex_prev.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Persistent temporal depth state (R32F image, read-write in shader)
        self._temporal_depth_tex = self.ctx.texture((dw, dh), 1, dtype='f4')
        self._temporal_depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        # Clear temporal state
        self._temporal_depth_tex.write(
            np.zeros((dh, dw), dtype=np.float32))

        self._tex_size = (dw, dh)
        self._has_prev_frame = False

        # Backward-compat aliases (used in ensure_textures callers)
        self._depth_tex = self._depth_tex_curr
        self._color_tex = self._color_tex_curr

        # Pre-allocate work buffers for this resolution
        self._interp_a = np.empty((dh, dw), dtype=np.float32)
        self._interp_b = np.empty((dh, dw), dtype=np.float32)
        self._dm_buf = np.empty((dh, dw), dtype=np.float32)
        self._tsnap_buf = np.empty((dh, dw), dtype=np.bool_)
        self._snap3d_buf = np.empty((dh, dw, 1), dtype=np.bool_)
        self._cr_buf = np.empty((dh, dw, 3), dtype=np.uint8)
        self._interp_color_buf = np.empty((dh, dw, 3), dtype=np.uint8)
        self._cluster_mask_buf = np.empty((dh, dw), dtype=np.bool_)

        # Pre-allocated resize destination buffers
        self._resize_depth_buf = np.empty((dh, dw), dtype=np.uint16)
        self._resize_color_buf = np.empty((dh, dw, 3), dtype=np.uint8)

        # Reset GPU effects state
        self._ghost_in_count = 0
        self._trail_in_count = 0

    # ────────────────── Frame preprocessing ──────────────────

    def preprocess_frame(self, state, depth_array, color_array, bilateral):
        """Downsample and optionally filter depth/color for the pipeline.

        Uses pre-allocated buffers to avoid per-frame allocations.
        Returns (depth_down, color_down) ready for upload.
        """
        dw, dh = state.down_w, state.down_h
        self.ensure_textures(state)

        src_h, src_w = depth_array.shape[:2]
        if (src_w, src_h) != (dw, dh):
            dd = cv2.resize(depth_array, (dw, dh),
                            dst=self._resize_depth_buf,
                            interpolation=cv2.INTER_NEAREST)
            dc = cv2.resize(color_array, (dw, dh),
                            dst=self._resize_color_buf)
        else:
            dd = depth_array
            dc = color_array

        if bilateral:
            # _interp_a is pre-allocated float32 for the input.
            # Note: cv2.bilateralFilter always allocates a new output array.
            np.copyto(self._interp_a, dd, casting='unsafe')
            dd = cv2.bilateralFilter(self._interp_a, 5, 80, 5)

        return dd, dc

    # ────────────────── Frame processing ──────────────────

    def upload_frame(self, state, new_kinect_frame):
        """Upload textures and prepare data for compute dispatch.

        Separated from dispatch_compute() so CPU upload cost can be
        profiled independently from compute shader submissions.
        """
        if state.curr_depth_down is None:
            self._frame_ready = False
            return

        self.ensure_textures(state)

        if new_kinect_frame:
            # Swap GPU textures: current → previous
            self._depth_tex_prev, self._depth_tex_curr = (
                self._depth_tex_curr, self._depth_tex_prev)
            self._color_tex_prev, self._color_tex_curr = (
                self._color_tex_curr, self._color_tex_prev)
            # Update backward-compat aliases
            self._depth_tex = self._depth_tex_curr
            self._color_tex = self._color_tex_curr

            # Upload depth: uint16 → float32 into pre-allocated buffer
            np.copyto(self._interp_a, state.curr_depth_down,
                      casting='unsafe')
            self._depth_tex_curr.write(self._interp_a)

            # Upload BGR color directly (shader swizzles BGR→RGB)
            self._color_tex_curr.write(state.curr_color_down)

            self._has_prev_frame = True
            self._interp_alpha = 0.0
            self._frame_ready = True
        else:
            # Interpolation frame — compute alpha from elapsed time
            if not self._has_prev_frame or state.prev_depth_down is None:
                self._frame_ready = False
                return
            elapsed = state.last_kinect_time  # will be used with now in dispatch
            self._interp_alpha = -1.0  # sentinel: compute in dispatch
            self._frame_ready = True

    def dispatch_compute(self, state, camera, now, new_kinect_frame):
        """Run build + ghost + trail compute shaders.

        Must be called after upload_frame(). Returns True if the point
        cloud was rebuilt this frame.
        """
        if not self._frame_ready:
            return False

        dw, dh = state.down_w, state.down_h

        # Resolve interpolation alpha
        if self._interp_alpha < 0.0:
            elapsed = now - state.last_kinect_time
            interp_alpha = min(elapsed / (1.0 / 30.0), 1.0)
            if interp_alpha >= 0.999:
                return False
        else:
            interp_alpha = self._interp_alpha

        # ── Cluster threshold (throttled to ~5 Hz, downsampled) ──
        d_min = settings["depth_min_cm"] / 100.0
        d_max = settings["depth_max_cm"] / 100.0
        if (new_kinect_frame
                and now >= self._cluster_next_time
                and (settings["cluster_coloring"]
                     or settings["cyberspace"])):
            self._cluster_next_time = now + 0.2  # ~5 Hz
            # _interp_a has float32 depth from upload; subsample every 4th pixel
            sub = self._interp_a.ravel()[::4]
            np.multiply(sub, 0.001, out=sub)
            valid = sub[(sub > d_min) & (sub < d_max)]
            if len(valid) > 50:
                _, ct = _pc_module._find_cluster_threshold(
                    valid, d_min, d_max)
                _pc_module.cluster_threshold_t = ct

        # ── Intrinsics scaled to cloud resolution ──
        sx = dw / 640.0
        sy = dh / 480.0

        # ── Reset indirect draw buffer (count=0, instanceCount=1) ──
        self.indirect_buf.write(_INDIRECT_RESET)

        # ── Bind textures + SSBOs ──
        self._depth_tex_curr.use(location=2)
        self._color_tex_curr.use(location=3)
        self._depth_tex_prev.use(location=4)
        self._color_tex_prev.use(location=5)
        # Bind temporal depth as image for read-write in shader
        self._temporal_depth_tex.bind_to_image(0, read=True, write=True)
        self.pc_ssbo.bind_to_storage_buffer(0)
        self.indirect_buf.bind_to_storage_buffer(1)

        # ── Set compute uniforms (cached — skip unchanged values) ──
        c = self._build_compute
        uc = self._uniform_cache

        def _su(name, val):
            """Set uniform only if changed."""
            if uc.get(name) != val:
                c[name].value = val
                uc[name] = val

        _su('depth_tex', 2)
        _su('color_tex', 3)
        _su('prev_depth_tex', 4)
        _su('prev_color_tex', 5)
        _su('cloud_w', dw)
        _su('cloud_h', dh)
        _su('fx_d', float(config.FX_DEPTH * sx))
        _su('fy_d', float(config.FY_DEPTH * sy))
        _su('cx_d', float(config.CX_DEPTH * sx))
        _su('cy_d', float(config.CY_DEPTH * sy))
        _su('fx_c', float(config.FX_COLOR * sx))
        _su('fy_c', float(config.FY_COLOR * sy))
        _su('cx_c', float(config.CX_COLOR * sx))
        _su('cy_c', float(config.CY_COLOR * sy))
        _su('depth_min', d_min)
        _su('depth_max', d_max)
        _su('depth_color_align', int(
            settings.get("depth_color_align", 0)))
        _su('baseline_x', float(
            settings.get("baseline_x_mm", 25.0)) / 1000.0)
        _su('baseline_y', float(
            settings.get("baseline_y_mm", 0.0)) / 1000.0)
        _su('max_points', MAX_POINTS)

        # Interpolation (changes every frame)
        c['interp_alpha'].value = float(interp_alpha)
        c['interp_has_prev'].value = int(
            self._has_prev_frame and interp_alpha > 0.0)
        _su('interp_snap_threshold_mm', 200.0)

        # Temporal smoothing
        _su('temporal_on', int(settings["temporal_smooth"]))
        _su('temporal_alpha', float(
            np.clip(settings["temporal_alpha"], 0.05, 1.0)))
        _su('temporal_snap_threshold_mm', float(
            TEMPORAL_SNAP_THRESHOLD))

        # Frustum culling
        fp = camera.get_frustum_planes()
        culling_on = int(settings["frustum_culling"] and fp is not None)
        _su('frustum_culling_on', culling_on)
        if fp is not None:
            c['frustum_planes'].write(fp.tobytes())
        _su('frustum_margin', FRUSTUM_MARGIN)

        # Edge filter
        _su('edge_filter_on', int(settings["edge_filter"]))
        _su('edge_filter_threshold', float(
            settings["edge_filter_threshold"]) / 1000.0)

        # Cluster colouring
        _su('cluster_coloring_on', int(settings["cluster_coloring"]))
        _su('cluster_blend', float(settings["cluster_blend"]))
        c['cluster_threshold'].value = float(
            _pc_module.cluster_threshold_t)

        # ── Dispatch: 16×16 workgroups ──
        self._build_compute.run((dw + 15) // 16, (dh + 15) // 16)
        self.ctx.memory_barrier(barriers=self._ssbo_barrier)

        # ── GPU Effects ──
        # Always use dw*dh as Phase 1 dispatch bound.
        # The shader reads draw_count from SSBO and early-exits;
        # extra threads cost ~0.02ms but avoids blocking indirect_buf.read().
        max_dispatch = dw * dh
        fx_ghost = (settings["cyber_ghost_particles"]
                    and self._ghost_compute is not None)
        fx_trail = (settings["cyber_particle_trails"]
                    and self._trail_compute is not None)

        if fx_ghost or fx_trail:
            state.need_gpu_reads = True

        # ── Ghost Particles (two-phase dispatch) ──
        if fx_ghost:
            self._dispatch_ghost_particles(max_dispatch, now, state)
        else:
            self._ghost_in_count = 0

        # ── Particle Trails (two-phase dispatch) ──
        if fx_trail:
            self._dispatch_particle_trails(max_dispatch, now, state)
        else:
            self._trail_in_count = 0

        # Flush GPU commands to pipeline — ensures driver starts executing
        # dispatches immediately so readback next frame won't stall.
        self._glFlush()

        return True

    # ────────────────── Ghost Particles ──────────────────

    def _dispatch_ghost_particles(self, max_dispatch, now, state):
        """Two-phase ghost dispatch with memory barrier between phases.

        Phase 0: Decay old ghosts (reads ghost_in, writes point_data + ghost_out)
        Phase 1: Sample new ghosts (reads point_data, writes ghost_out)

        The barrier ensures Phase 0's writes to point_data are globally
        visible before Phase 1 reads from point_data.

        ``max_dispatch`` is a conservative upper bound (dw*dh).  The shader
        reads ``draw_count`` directly from the indirect SSBO for bounds
        checking — no CPU readback needed.
        """
        self._ghost_count_buf.write(_ZERO_U32)

        self.pc_ssbo.bind_to_storage_buffer(0)
        self.indirect_buf.bind_to_storage_buffer(1)
        self._ghost_ssbo[self._ghost_read].bind_to_storage_buffer(2)
        self._ghost_ssbo[self._ghost_write].bind_to_storage_buffer(3)
        self._ghost_count_buf.bind_to_storage_buffer(4)

        gc = self._ghost_compute
        gc['ghost_in_count'].value = self._ghost_in_count
        gc['ghost_sample_rate'].value = float(GHOST_SAMPLE)
        gc['ghost_ttl'].value = GHOST_TTL
        gc['max_points'].value = MAX_POINTS
        gc['max_ghost_points'].value = MAX_GHOST_GPU
        gc['frame_seed'].value = float(now)
        gc['max_dispatch'].value = max_dispatch

        # Phase 0: decay old ghosts
        gc['phase'].value = 0
        if self._ghost_in_count > 0:
            self._ghost_compute.run(
                (self._ghost_in_count + 255) // 256)
            self.ctx.memory_barrier(barriers=self._ssbo_barrier)

        # Phase 1: sample new points from current frame
        gc['phase'].value = 1
        if max_dispatch > 0:
            self._ghost_compute.run(
                (max_dispatch + 255) // 256)
            self.ctx.memory_barrier(barriers=self._ssbo_barrier)

        state.pending_ghost_swap = True

    # ────────────────── Particle Trails ──────────────────

    def _dispatch_particle_trails(self, max_dispatch, now, state):
        """Two-phase trail dispatch with memory barrier between phases.

        Phase 0: Process existing trails (drip rendering)
        Phase 1: Freeze new trail points from current frame

        ``max_dispatch`` is a conservative upper bound.  The shader reads
        ``draw_count`` directly from the indirect SSBO.
        """
        self._trail_count_buf.write(_ZERO_U32)

        self.pc_ssbo.bind_to_storage_buffer(0)
        self.indirect_buf.bind_to_storage_buffer(1)
        self._trail_ssbo[self._trail_read].bind_to_storage_buffer(2)
        self._trail_ssbo[self._trail_write].bind_to_storage_buffer(3)
        self._trail_count_buf.bind_to_storage_buffer(4)

        tc = self._trail_compute
        tc['trail_in_count'].value = self._trail_in_count
        tc['current_time'].value = float(now)
        tc['max_points'].value = MAX_POINTS
        tc['max_trail_points'].value = MAX_TRAIL_GPU
        tc['frame_seed'].value = float(now)
        tc['max_dispatch'].value = max_dispatch

        # Freeze effect (stochastic — CPU decides each frame)
        if np.random.random() < TRAIL_FREEZE_CHANCE:
            tc['freeze_this_frame'].value = 1
            tc['n_freeze'].value = int(
                np.random.randint(*TRAIL_FREEZE_N_RANGE))
            tc['freeze_lifetime'].value = float(
                np.random.uniform(*TRAIL_FREEZE_LIFE_RANGE))
        else:
            tc['freeze_this_frame'].value = 0
            tc['n_freeze'].value = 0
            tc['freeze_lifetime'].value = 1.0

        # Phase 0: process existing trails
        tc['phase'].value = 0
        if self._trail_in_count > 0:
            self._trail_compute.run(
                (self._trail_in_count + 255) // 256)
            self.ctx.memory_barrier(barriers=self._ssbo_barrier)

        # Phase 1: freeze new trail points
        tc['phase'].value = 1
        if max_dispatch > 0:
            self._trail_compute.run(
                (max_dispatch + 255) // 256)
            self.ctx.memory_barrier(barriers=self._ssbo_barrier)

        state.pending_trail_swap = True

    # ────────────────── Deferred readback ──────────────────

    def deferred_reads(self, state):
        """Swap ghost/trail double buffers.

        No GPU readback — dispatch uses MAX bounds, shader early-exits.
        Ghost/trail counts are managed entirely on the GPU via atomic
        counters that reset each frame.
        """
        if state.pending_ghost_swap:
            # Swap read/write SSBOs; use MAX as upper bound for next dispatch
            self._ghost_in_count = MAX_GHOST_GPU
            self._ghost_read, self._ghost_write = (
                self._ghost_write, self._ghost_read)
            state.pending_ghost_swap = False

        if state.pending_trail_swap:
            self._trail_in_count = MAX_TRAIL_GPU
            self._trail_read, self._trail_write = (
                self._trail_write, self._trail_read)
            state.pending_trail_swap = False

        state.need_gpu_reads = False

    # ────────────────── PLY export ──────────────────

    def export_ply(self, exports_dir, plugin_mgr=None, app_ctx=None):
        """Read the current GPU SSBO and save as binary PLY.

        Uses ``ctx.finish()`` which causes a **full pipeline stall**.
        This is acceptable for a one-shot manual export.

        .. note:: Serial export improvement

           For batch/serial export (e.g. recording sequences), replace
           ``ctx.finish()`` with a **fence + PBO** approach:

           1. Insert a fence after the compute dispatch::

                sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

           2. Before reading, wait on the fence::

                glClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, timeout_ns)

           3. Use a PBO for async readback to avoid blocking the render
              thread entirely.

           moderngl does not expose fences natively; use ``PyOpenGL``
           for the raw GL calls, or batch exports on a separate thread.
        """
        try:
            self.ctx.finish()   # full stall — acceptable for one-shot export

            indirect_data = self.indirect_buf.read(16)
            n_points = struct.unpack('I', indirect_data[:4])[0]
            if n_points == 0:
                log.warning("PLY export: 0 points")
                return None, None, None
            n_points = min(n_points, MAX_POINTS)

            raw = self.pc_ssbo.read(n_points * 6 * 4)
            data = np.frombuffer(raw, dtype=np.float32).copy().reshape(
                (n_points, 6))
            xyz = data[:, :3]
            rgb_f = data[:, 3:]
            rgb = np.clip(rgb_f * 255.0, 0, 255).astype(np.uint8)

            log.info("PLY color stats: R[%d..%d] G[%d..%d] B[%d..%d]",
                     rgb[:, 0].min(), rgb[:, 0].max(),
                     rgb[:, 1].min(), rgb[:, 1].max(),
                     rgb[:, 2].min(), rgb[:, 2].max())

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(exports_dir, f"kinect_{timestamp}.ply")

            ply_dt = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ])
            vertices = np.empty(n_points, dtype=ply_dt)
            vertices['x'] = xyz[:, 0]
            vertices['y'] = xyz[:, 1]
            vertices['z'] = xyz[:, 2]
            vertices['red'] = rgb[:, 0]
            vertices['green'] = rgb[:, 1]
            vertices['blue'] = rgb[:, 2]

            with open(path, "wb") as f:
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    f"element vertex {n_points}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    "end_header\n"
                )
                f.write(header.encode("ascii"))
                f.write(vertices.tobytes())

            log.info("PLY exported: %s (%d pts)", path, n_points)

            # Plugin: on_export hook
            if plugin_mgr is not None and app_ctx is not None:
                plugin_mgr.call_export(app_ctx, xyz, rgb, path)

            return xyz, rgb, path

        except Exception as e:
            log.error("PLY export error: %s", e)
            return None, None, None

    # ────────────────── Cleanup ──────────────────

    def release(self):
        """Release all GPU resources."""
        for buf in [self.pc_ssbo, self.indirect_buf]:
            if buf is not None:
                buf.release()
        for tex in (self._depth_tex_curr, self._depth_tex_prev,
                    self._color_tex_curr, self._color_tex_prev,
                    self._temporal_depth_tex):
            if tex is not None:
                tex.release()
        for buf in self._ghost_ssbo:
            if buf is not None:
                buf.release()
        if self._ghost_count_buf is not None:
            self._ghost_count_buf.release()
        for buf in self._trail_ssbo:
            if buf is not None:
                buf.release()
        if self._trail_count_buf is not None:
            self._trail_count_buf.release()
