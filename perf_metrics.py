# =============================
# PerfMetrics — lightweight performance profiler
# =============================
"""
Collects CPU and GPU timing metrics for real-time analysis.

Usage::

    metrics = PerfMetrics(ctx)        # once at init
    metrics.begin_frame()             # top of main loop
    metrics.begin("kinect")           # before section
    ...                               # work
    metrics.end("kinect")             # after section
    metrics.begin_gpu("compute")      # before GPU dispatch
    ...                               # dispatch
    metrics.end_gpu("compute")        # after GPU dispatch
    metrics.end_frame()               # bottom of main loop

CPU sections use ``time.perf_counter()``; GPU sections use
OpenGL timer queries (``GL_TIME_ELAPSED``).  GPU results are
read back with a 1-frame delay to avoid pipeline stalls.

All data is exposed as rolling averages for ImGui display.
"""

import time
from collections import defaultdict
import numpy as np

import moderngl

# Rolling history length (frames)
_HISTORY = 120
_GPU_QUERY_POOL = 3  # ring-buffer depth to avoid stalls


class _CpuSection:
    __slots__ = ('history', 'idx', 'start', 'avg_ms')

    def __init__(self):
        self.history = np.zeros(_HISTORY, dtype=np.float64)
        self.idx = 0
        self.start = 0.0
        self.avg_ms = 0.0


class _GpuSection:
    __slots__ = ('queries', 'ring_idx', 'avg_ms', 'history', 'hist_idx')

    def __init__(self, ctx):
        # Ring buffer of query pairs to read with 1-frame delay
        self.queries = []
        for _ in range(_GPU_QUERY_POOL):
            self.queries.append(ctx.query(time=True))
        self.ring_idx = 0
        self.avg_ms = 0.0
        self.history = np.zeros(_HISTORY, dtype=np.float64)
        self.hist_idx = 0


class PerfMetrics:
    """Lightweight CPU + GPU profiler with rolling averages."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.enabled = False  # toggled from UI

        # CPU sections
        self._cpu: dict[str, _CpuSection] = {}
        self._cpu_order: list[str] = []

        # GPU sections
        self._gpu: dict[str, _GpuSection] = {}
        self._gpu_order: list[str] = []

        # Frame-level timing
        self._frame_start = 0.0
        self._frame_section = _CpuSection()

        # Totals for sparkline
        self.frame_history = np.zeros(_HISTORY, dtype=np.float32)
        self._frame_hist_idx = 0
        self.frame_avg_ms = 0.0
        self.frame_max_ms = 0.0

        # For 1% low FPS
        self._fps_history = np.zeros(_HISTORY, dtype=np.float32)
        self._fps_hist_idx = 0
        self.fps_1pct_low = 0.0

    # ────────────── Frame boundary ──────────────

    def begin_frame(self):
        if not self.enabled:
            return
        self._frame_start = time.perf_counter()

    def end_frame(self):
        if not self.enabled:
            return
        elapsed = (time.perf_counter() - self._frame_start) * 1000.0

        # Frame time history
        idx = self._frame_hist_idx % _HISTORY
        self.frame_history[idx] = elapsed
        self._frame_hist_idx += 1

        n = min(self._frame_hist_idx, _HISTORY)
        self.frame_avg_ms = float(np.mean(self.frame_history[:n]))
        self.frame_max_ms = float(np.max(self.frame_history[:n]))

        # 1% low FPS
        fps = 1000.0 / max(elapsed, 0.01)
        fidx = self._fps_hist_idx % _HISTORY
        self._fps_history[fidx] = fps
        self._fps_hist_idx += 1
        fn = min(self._fps_hist_idx, _HISTORY)
        sorted_fps = np.sort(self._fps_history[:fn])
        pct1_count = max(int(fn * 0.01), 1)
        self.fps_1pct_low = float(np.mean(sorted_fps[:pct1_count]))

        # Read back GPU queries (1-frame-delayed results)
        self._collect_gpu_results()

    # ────────────── CPU sections ──────────────

    def begin(self, name: str):
        if not self.enabled:
            return
        sec = self._cpu.get(name)
        if sec is None:
            sec = _CpuSection()
            self._cpu[name] = sec
            self._cpu_order.append(name)
        sec.start = time.perf_counter()

    def end(self, name: str):
        if not self.enabled:
            return
        sec = self._cpu.get(name)
        if sec is None:
            return
        elapsed = (time.perf_counter() - sec.start) * 1000.0
        idx = sec.idx % _HISTORY
        sec.history[idx] = elapsed
        sec.idx += 1
        n = min(sec.idx, _HISTORY)
        sec.avg_ms = float(np.mean(sec.history[:n]))

    # ────────────── GPU sections ──────────────

    def begin_gpu(self, name: str):
        if not self.enabled:
            return
        sec = self._gpu.get(name)
        if sec is None:
            sec = _GpuSection(self.ctx)
            self._gpu[name] = sec
            self._gpu_order.append(name)
        q = sec.queries[sec.ring_idx % _GPU_QUERY_POOL]
        q.__enter__()

    def end_gpu(self, name: str):
        if not self.enabled:
            return
        sec = self._gpu.get(name)
        if sec is None:
            return
        q = sec.queries[sec.ring_idx % _GPU_QUERY_POOL]
        q.__exit__(None, None, None)
        sec.ring_idx += 1

    def _collect_gpu_results(self):
        """Read GPU timer results from queries that are ≥1 frame old."""
        for name in self._gpu_order:
            sec = self._gpu[name]
            if sec.ring_idx < 2:
                continue  # need at least 1. frame delay
            read_idx = (sec.ring_idx - 1) % _GPU_QUERY_POOL
            q = sec.queries[read_idx]
            elapsed_ns = q.elapsed
            elapsed_ms = elapsed_ns / 1_000_000.0

            hidx = sec.hist_idx % _HISTORY
            sec.history[hidx] = elapsed_ms
            sec.hist_idx += 1
            n = min(sec.hist_idx, _HISTORY)
            sec.avg_ms = float(np.mean(sec.history[:n]))

    # ────────────── Data access for UI ──────────────

    def get_cpu_sections(self):
        """Returns list of (name, avg_ms, history_array)."""
        return [
            (name, self._cpu[name].avg_ms, self._cpu[name].history)
            for name in self._cpu_order
        ]

    def get_gpu_sections(self):
        """Returns list of (name, avg_ms, history_array)."""
        return [
            (name, self._gpu[name].avg_ms, self._gpu[name].history)
            for name in self._gpu_order
        ]

    def reset(self):
        """Clear all history."""
        self._cpu.clear()
        self._cpu_order.clear()
        self._gpu.clear()
        self._gpu_order.clear()
        self.frame_history[:] = 0
        self._frame_hist_idx = 0
        self._fps_history[:] = 0
        self._fps_hist_idx = 0
