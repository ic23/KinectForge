<div align="center">

# Kinect360Viewer

**Real-time 3D point cloud viewer for Xbox Kinect 360**

GPU-accelerated rendering · cyberspace effects · plugin system · built-in profiler

[![Python 3.10](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![OpenGL 4.3+](https://img.shields.io/badge/OpenGL-4.3%2B-5586A4?logo=opengl&logoColor=white)](#requirements)
[![Platform](https://img.shields.io/badge/platform-Windows-0078D4?logo=windows&logoColor=white)](#requirements)
[![Kinect SDK 1.8](https://img.shields.io/badge/Kinect%20SDK-v1.8-107C10?logo=xbox&logoColor=white)](https://www.microsoft.com/en-us/download/details.aspx?id=40278)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

</div>

---

## Overview

Kinect360Viewer transforms an Xbox Kinect 360 sensor into a real-time 3D point cloud viewer. The entire depth-to-point-cloud pipeline runs on the GPU via OpenGL 4.3 compute shaders — no CPU-side geometry building, no VBO upload stalls, no blocking GPU readbacks.

The viewer includes 20+ visual effects (cyberspace mode, ghost particles, particle trails, bloom, chromatic aberration), depth filters, color palettes, a preset system, a built-in performance profiler, and an extensible plugin architecture.

**Key highlights:**

- **Zero-CPU point cloud** — depth texture → compute shader → SSBO → `glDrawArraysIndirect`
- **144 FPS interpolation** — Kinect's 30 FPS depth smoothly interpolated on GPU with depth-edge snapping
- **GPU temporal smoothing** — EMA-based depth stabilization computed per-pixel in compute shader
- **Batched atomics** — workgroup-level shared memory reservation reduces global atomic contention ~256×
- **20+ real-time effects** — cyberspace mode, ghost particles, particle trails, bloom, glitch, CRT, and more
- **Built-in profiler** — CPU + GPU section timings, 1% low FPS, sparkline graphs
- **Plugin system** — extend with Python scripts, hot-reloadable at runtime
- **One-click setup** — `install.bat` → `run.bat`

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Controls](#controls)
- [Features](#features)
  - [Rendering Pipeline](#rendering-pipeline)
  - [Depth Filters](#depth-filters)
  - [Cyberspace Mode](#cyberspace-mode)
  - [Post-Processing](#post-processing)
  - [Vertex Effects](#vertex-effects)
  - [Color Palettes](#color-palettes)
  - [Export](#export)
  - [Presets](#presets)
  - [Performance Profiler](#performance-profiler)
- [Plugin API](#plugin-api)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Requirements

| Component | Details |
|-----------|---------|
| **OS** | Windows 10 / 11 |
| **Python** | 3.10 (other versions not tested) |
| **Kinect SDK** | [Microsoft Kinect SDK v1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278) |
| **Sensor** | Xbox Kinect 360 (v1) with USB adapter + power supply |
| **GPU** | OpenGL 4.3+ capable (compute shaders required) |

---

## Installation

### Step 1 — Install Kinect SDK

Download and install [Microsoft Kinect SDK v1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278). Connect your Kinect sensor (USB + power supply).

### Step 2 — Set up the environment

Double-click **`install.bat`**. The script will:

1. Locate your Python 3.10 installation
2. Create a virtual environment (`kinect_env/`)
3. Install all dependencies:

```
numpy  opencv-python  moderngl  glfw  imgui[glfw]  pythonnet  PyOpenGL  PyOpenGL-accelerate
```

### Step 3 — Launch

Double-click **`run.bat`**.

> The script activates the virtual environment and runs `main.py`. On first launch, the app will initialize the Kinect sensor and open a 1280×720 window.

---

## Usage

The application opens an interactive 3D viewport with a control panel on the left. All settings are adjustable in real time via the ImGui interface. Changes can be saved to presets and exported.

---

## Controls

| Input | Action |
|:------|:-------|
| <kbd>LMB</kbd> + drag | Orbit camera |
| <kbd>RMB</kbd> + drag | Pan camera |
| Scroll wheel | Zoom in/out |
| <kbd>Home</kbd> | Reset camera to default |
| <kbd>F11</kbd> | Toggle fullscreen |
| <kbd>F12</kbd> | Capture screenshot (PNG, no UI overlay) |
| <kbd>P</kbd> | Export point cloud to PLY |
| <kbd>ESC</kbd> | Quit |

---

## Features

### Rendering Pipeline

| Feature | Description |
|---------|------------|
| **GPU Compute Pipeline** | Depth + color textures → compute shader (16×16 workgroups) → SSBO → `glDrawArraysIndirect`. Zero CPU geometry. Batched atomic reservation via shared memory (~256× less global atomic contention). |
| **GPU Frame Interpolation** | 30 → 144 FPS via weighted depth/color blending with 200 mm snap threshold at depth edges. Runs entirely in compute shader using double-buffered textures (current + previous Kinect frame). |
| **GPU Temporal Smoothing** | Exponential moving average per-pixel in compute shader via persistent `image2D` texture. Snap threshold prevents smearing at motion edges. |
| **Point Size & Shape** | Adjustable size (1–10), circle or square shape. |
| **Depth Range** | Configurable min/max depth in cm. |
| **Depth-Color Alignment** | Perspective correction for IR ↔ RGB camera offset with adjustable baseline (X/Y mm). |
| **Frustum Culling** | Gribb-Hartmann 6-plane extraction, evaluated per-point in compute shader. |
| **Depth-Scaled Points** | Near points rendered larger (configurable factor). |
| **Point Shading (SSAO)** | Per-point hemisphere ambient occlusion with adjustable strength. |

### Depth Filters

| Filter | Description |
|--------|------------|
| **Smooth Depth** | Bilateral filter (`cv2.bilateralFilter`) — smooths noise while preserving edges. |
| **Edge Filter** | Removes flying pixels at depth discontinuities (threshold: 10–200 mm). Runs in compute shader using `texelFetch` neighbor sampling. |
| **Temporal Smoothing** | GPU-side EMA with snap threshold — stabilizes surfaces without smearing motion. Persistent state in `r32f` image texture. |

### Cyberspace Mode

A full netrunner-aesthetic rendering mode with depth-based coloring and numerous glitch effects:

| Effect | Description |
|--------|------------|
| **Cluster Coloring** | Automatic near/far cluster detection via depth histogram gap (~5 Hz, downsampled). Cyan near, magenta far. |
| **Invert Colors** | Swap near/far cluster palette. |
| **Color Glitch** | Random per-fragment color replacement (cyan / pink / white / channel swap). |
| **Glitch Bands** | Horizontal displacement bands, stochastically triggered. |
| **Double Glitch** | Screen-space duplication with random offset (composite pass). |
| **Ghost Render** | Temporal screen accumulation with configurable decay (ping-pong FBOs). |
| **Drip Trails** | Downward UV shift creating vertical drip effect. |
| **Ghost Particles** | Sampled points persist as fading ghosts. GPU double-buffered SSBO, two-phase dispatch with `glMemoryBarrier`. Up to 200K entries. |
| **Particle Trails** | Randomly frozen points with downward drip animation. GPU compute, two-phase dispatch. Up to 30K base points. |
| **Grid Floor** | Procedural dot grid at floor level. |
| **Noise Particles** | Floating ambient particles, shader-animated. |
| **Jitter** | Per-vertex position noise. |

### Post-Processing

| Effect | Description |
|--------|------------|
| **Bloom** | Multi-sample gaussian blur with brightness threshold. Adjustable strength. |
| **Chromatic Aberration** | Radial R/G/B channel offset from screen center. |
| **Edge Glow** | Neighbor color-difference detection with white-cyan glow. |
| **Pixelate** | UV grid snapping, adjustable pixel size (1–40). |

### Vertex Effects

| Effect | Description |
|--------|------------|
| **Wave Distortion** | Sinusoidal position displacement. Configurable amplitude and frequency. |
| **Voxelize** | Snaps vertex positions to a 3D grid. |
| **Pulse** | Rhythmic point size oscillation. |

### Color Palettes

| Palette | Style |
|---------|-------|
| **Original** | Kinect RGB colors |
| **Thermal** | Blue → Cyan → Green → Yellow → Red → White |
| **Night Vision** | Green phosphor with noise grain |
| **Retro Amber** | Warm amber monochrome |

### Export

| Format | Details |
|--------|---------|
| **Screenshot** (<kbd>F12</kbd>) | PNG capture of the viewport without UI overlay. Saved to `screenshots/`. |
| **PLY** (<kbd>P</kbd>) | Binary little-endian PLY with XYZ (float32) + RGB (uint8). Read directly from GPU SSBO via `ctx.finish()`. Up to 1M points. Saved to `exports/`. |

### Presets

Save, load, and delete named presets via the UI panel. All settings (including plugin settings) are stored as JSON files in `presets/`. Presets are portable — drop a `.json` file into the folder and it appears in the list.

### Performance Profiler

Built-in CPU + GPU profiler accessible from the **Performance** section in the UI panel.

| Feature | Description |
|---------|-------------|
| **CPU Sections** | Per-section timing (`perf_counter`) for: swap, events, readback, kinect, gpu_upload, compute, tex_upload, render, imgui |
| **GPU Sections** | OpenGL timer queries (`GL_TIME_ELAPSED`) with 1-frame-delayed readback (ring buffer of 3 queries) for: compute, render |
| **Frame Stats** | Average/max frame time, 1% low FPS (worst 1% of frames over 120-frame window) |
| **Sparkline** | 120-frame rolling graph of frame times |
| **Zero overhead** | Disabled by default; no performance cost when profiler is off |

---

## Plugin API

Plugins are Python files placed in the `plugins/` directory. Each file must define a class inheriting from `KinectPlugin`. Plugins are auto-discovered at startup and can be hot-reloaded from the UI.

### Minimal Example

```python
from plugin_api import KinectPlugin

class MyPlugin(KinectPlugin):
    name = "My Plugin"
    version = "1.0"
    author = "Author"
    description = "Does something cool"

    def on_init(self, app):
        """Called once after OpenGL initialization."""
        pass

    def on_kinect_frame(self, app, depth, color):
        """Called on each new Kinect frame.
        depth: np.ndarray uint16 (mm), color: np.ndarray uint8 (BGR)"""
        pass

    def on_draw_ui(self, app):
        """Draw custom ImGui panels."""
        pass
```

### Lifecycle Hooks

| Hook | When |
|:-----|:-----|
| `on_init(app)` | After GL init, before main loop |
| `on_cleanup(app)` | Application shutdown |
| `on_frame_start(app, dt)` | Start of each frame |
| `on_kinect_frame(app, depth, color)` | New Kinect depth + color data received |
| `on_pre_render(app)` | Before 3D scene render (after compute dispatch) |
| `on_post_render(app)` | After composite pass, before ImGui |
| `on_draw_ui(app)` | Inside ImGui frame |
| `on_export(app, xyz, rgb, path)` | After PLY export completes |
| `on_settings_changed(app, key, old, new)` | Any setting modified via UI |
| `on_preset_save(app, name)` | Preset saved |
| `on_preset_load(app, name)` | Preset loaded |

### AppContext

The `app` argument passed to all hooks exposes:

| Field | Type | Description |
|:------|:-----|:------------|
| `ctx` | `moderngl.Context` | OpenGL context |
| `window` | GLFW window | Window handle |
| `camera` | `OrbitCamera` | Camera (get MVP, frustum planes) |
| `settings` | `dict` | All application settings |
| `sensor` | `KinectSensor` | .NET Kinect sensor object |
| `pc_ssbo` | `moderngl.Buffer` | Point cloud SSBO (XYZ+RGB, float32×6 per point) |
| `indirect_buf` | `moderngl.Buffer` | Indirect draw buffer (first uint = point count) |
| `width`, `height` | `int` | Viewport size |
| `time` | `float` | Current time (seconds) |
| `dt` | `float` | Frame delta time |
| `fps` | `int` | Current FPS |

### Plugin Settings Persistence

Override `get_settings()` → `dict` and `set_settings(data)` to save/restore plugin-specific settings with presets.

### Included Plugins

| Plugin | Description |
|--------|-------------|
| **Auto Orbit** | Cinematic auto-rotation with vertical bob, zoom breathe, pause on mouse drag |
| **CRT Monitor** | Post-processing CRT effect: scanlines, vignette, barrel distortion, flicker, RGB shift, film noise |
| **FPS Overlay** | Transparent overlay with FPS and point count (throttled GPU readback at 2 Hz) |
| **Point Explosion** | GPU compute shader that explodes cloud from center with gravity, spin, and fade |
| **Smoke Dissolve** | Top-down dissolve sweep into rising turbulent smoke with configurable color |

---

## Architecture

```
Kinect 360 (30 FPS)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  C# Interop (pythonnet)                          │
│  DepthStream → int16[] → numpy (bit-shift >> 3)  │
│  ColorStream → byte[] → numpy (BGRA→BGR)         │
│  Zero-copy double-buffered output arrays         │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  CPU Pre-processing                              │
│  Downsample (cv2.resize, pre-allocated dst) →    │
│  Optional bilateral filter                       │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  GPU Upload (< 0.15 ms)                          │
│  Depth: uint16→float32 → R32F texture            │
│  Color: BGR direct upload (shader swizzles)      │
│  Double-buffered: curr ↔ prev textures           │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  GPU Compute Pipeline (OpenGL 4.3)               │
│                                                  │
│  Build Shader (16×16 workgroups)                 │
│  ├─ Interpolation (prev↔curr depth+color blend)  │
│  ├─ Temporal smoothing (EMA via image2D state)   │
│  ├─ Range filter, edge filter (texelFetch)       │
│  ├─ Unproject (IR intrinsics → 3D)               │
│  ├─ Color lookup (RGB intrinsics + baseline)     │
│  ├─ BGR→RGB swizzle in shader                    │
│  ├─ Frustum culling (6-plane test)               │
│  ├─ Cluster coloring (depth histogram gap)       │
│  └─ Atomic append → SSBO + indirect draw count   │
│       │                                          │
│       ├── Ghost Particles Compute (2-phase)      │
│       │   Shared-memory batched atomics           │
│       │   Double-buffered SSBO, up to 200K       │
│       │                                          │
│       └── Particle Trails Compute (2-phase)      │
│           Shared-memory batched atomics           │
│           Double-buffered SSBO, up to 30K        │
│                                                  │
│  glFlush() → pipeline GPU commands immediately   │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Render                                          │
│  glDrawArraysIndirect (points from SSBO)         │
│  + grid floor + noise particles                  │
│       │                                          │
│       ▼                                          │
│  Post-Processing FBO chain                       │
│  Scene → Temporal Accumulation (ping-pong)       │
│       → Composite (bloom, chroma, edge, pixel)   │
│       → Plugin post-effects (CRT, etc.)          │
│       → Screen                                   │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Frame Timing                                    │
│  VSync (swap_interval) OR                        │
│  Frame limiter (hybrid sleep + spin-wait, 1 ms   │
│  Windows timer resolution via timeBeginPeriod)   │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
Kinect360Viewer/
│
├── main.py              Application class, GLFW window, main loop
├── app_state.py         AppState dataclass (replaces 30+ globals)
├── config.py            Settings TypedDict, constants, presets
├── kinect_capture.py    Kinect v1 sensor I/O via .NET interop
├── gpu_pipeline.py      Compute shaders, SSBOs, point cloud build
├── renderer.py          OpenGL programs, FBOs, draw calls
├── shaders.py           All GLSL shaders (vertex, fragment, compute ×4)
├── camera.py            OrbitCamera with MVP and frustum extraction
├── pointcloud.py        Cluster threshold detection (histogram gap)
├── ui_manager.py        ImGui panels, settings UI, preset management
├── perf_metrics.py      CPU + GPU performance profiler
├── plugin_api.py        KinectPlugin base class, PluginManager, AppContext
│
├── install.bat          One-click environment setup (venv + pip)
├── run.bat              One-click launch (activate venv → python main.py)
├── LICENSE              MIT License
│
├── presets/              Saved setting presets (.json)
├── screenshots/          Captured screenshots (.png)
├── exports/              Exported point clouds (.ply)
└── plugins/              User plugins (.py)
    ├── auto_orbit.py        Auto-rotation with camera animation
    ├── crt_monitor.py       CRT monitor post-effect
    ├── fps_overlay.py       FPS + point count overlay
    ├── point_explosion.py   GPU compute explosion effect
    └── smoke_mug.py         Smoke dissolve effect
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Python не найден"** | Install Python 3.10 to the default path (`AppData\Local\Programs\Python\Python310\`). |
| **"Kinect не обнаружен"** | Check USB + power supply connections. Verify Kinect appears in Device Manager. |
| **"Near mode не поддерживается"** | Normal for standard Kinect 360. Near mode requires Kinect for Windows. The app continues in Default mode. |
| **"GPU pipeline requires OpenGL 4.3+"** | Your GPU does not support compute shaders. Update drivers or use a newer GPU. |
| **Kinect disconnects mid-session** | The app auto-reconnects within ~3 seconds. Check power supply stability. |
| **Low FPS** | Lower the Point Cloud resolution preset (e.g., 320×240). Disable Bloom, Ghost Particles, Particle Trails. Enable the **Performance** profiler to identify the bottleneck. |
| **Crash on minimize** | Fixed — the app skips rendering when the framebuffer size is 0×0. |
| **"invalid color attachment"** | Usually means the window was minimized or resized to 0. Should be handled automatically. |

---

## License

This project is licensed under the [MIT License](LICENSE).
