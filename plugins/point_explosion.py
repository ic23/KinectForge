# =============================
# Plugin: Point Explosion
# =============================
"""
"Взрывает" облако точек — по нажатию кнопки все точки
разлетаются от центра взрыва, вращаясь и затухая.

Использует GPU compute shader для модификации SSBO in-place
(до рендера, после build compute).
"""
import time
import math
import imgui
import moderngl
import numpy as np

from plugin_api import KinectPlugin

_EXPLOSION_COMPUTE = """
#version 430
layout(local_size_x = 256) in;

// Point cloud SSBO: interleaved (x, y, z, r, g, b) per point
layout(std430, binding = 0) buffer Points {
    float data[];
};

layout(std430, binding = 1) buffer DrawIndirect {
    uint draw_count;
    uint draw_instance_count;
    uint draw_first;
    uint draw_base_instance;
};

uniform vec3 center;        // центр взрыва
uniform float force;        // сила (0..1+)
uniform float elapsed;      // время с момента взрыва
uniform float gravity;      // гравитация
uniform float spin;         // вращение
uniform float spread;       // разброс направления
uniform float fade;         // затухание цвета
uniform float max_radius;   // макс. радиус разлёта
uniform uint max_pt;

// PCG hash
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float hashf(uint v) { return float(pcg(v)) / 4294967295.0; }

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= min(draw_count, max_pt)) return;

    uint base = idx * 6u;
    vec3 pos = vec3(data[base], data[base+1u], data[base+2u]);
    vec3 col = vec3(data[base+3u], data[base+4u], data[base+5u]);

    // Направление от центра взрыва
    vec3 dir = pos - center;
    float dist = length(dir);
    if (dist < 0.001) {
        // Случайное направление для точек в центре
        float h1 = hashf(idx * 3u + 1u) * 6.2832;
        float h2 = hashf(idx * 3u + 2u) * 3.1416 - 1.5708;
        dir = vec3(cos(h2)*cos(h1), sin(h2), cos(h2)*sin(h1));
        dist = 0.001;
    } else {
        dir = dir / dist;
    }

    // Случайный разброс направления (уникальный для каждой точки)
    float r1 = (hashf(idx * 7u + 3u) - 0.5) * spread;
    float r2 = (hashf(idx * 7u + 5u) - 0.5) * spread;
    float r3 = (hashf(idx * 7u + 7u) - 0.5) * spread;
    dir = normalize(dir + vec3(r1, r2, r3));

    // Скорость зависит от расстояния до центра (ближние летят быстрее)
    float speed_mult = 1.0 + (1.0 / max(dist, 0.1)) * 0.3;
    float t = elapsed;

    // Смещение = сила × время × скорость
    float displacement = force * t * speed_mult;
    displacement = min(displacement, max_radius);

    // Гравитация (точки падают вниз со временем)
    float grav_offset = -0.5 * gravity * t * t;

    // Вращение вокруг Y-оси
    float angle = spin * t * (hashf(idx * 11u + 13u) * 0.5 + 0.75);
    float cs = cos(angle);
    float sn = sin(angle);

    vec3 offset = dir * displacement;
    // Применяем вращение к горизонтальной компоненте
    float ox = offset.x * cs - offset.z * sn;
    float oz = offset.x * sn + offset.z * cs;
    offset.x = ox;
    offset.z = oz;
    offset.y += grav_offset;

    pos += offset;

    // Затухание цвета (яркость падает)
    float brightness = max(1.0 - fade * t, 0.0);
    // Тёплый оттенок при взрыве (огненный)
    float heat = max(1.0 - t * 0.5, 0.0) * force * 0.3;
    col.r = min(col.r + heat, 1.0);
    col.g *= brightness;
    col.b *= brightness * brightness; // синий гаснет быстрее

    data[base]     = pos.x;
    data[base+1u]  = pos.y;
    data[base+2u]  = pos.z;
    data[base+3u]  = col.r * brightness + heat * 0.5;
    data[base+4u]  = col.g;
    data[base+5u]  = col.b;
}
"""


class PointExplosionPlugin(KinectPlugin):
    name = "Point Explosion"
    version = "1.0"
    author = "Kinect 360"
    description = "Взрывает облако точек — частицы разлетаются от центра"

    def __init__(self):
        super().__init__()
        self._compute = None
        self._exploding = False
        self._explosion_time = 0.0
        self._duration = 3.0       # длительность анимации (сек)
        self._force = 1.5          # сила взрыва
        self._gravity = 1.0        # гравитация
        self._spin = 2.0           # вращение
        self._spread = 0.6         # разброс направления
        self._fade = 0.4           # скорость затухания
        self._max_radius = 5.0     # макс. радиус разлёта
        self._center = [0.0, 0.0, -2.0]  # центр взрыва
        self._auto_center = True   # авто-центр по облаку
        self._loop = False         # зацикливание

    def on_init(self, app):
        try:
            self._compute = app.ctx.compute_shader(_EXPLOSION_COMPUTE)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Explosion plugin shader error: %s", e)
            self._compute = None

    def _trigger_explosion(self):
        """Запускает взрыв."""
        self._exploding = True
        self._explosion_time = 0.0

    def on_pre_render(self, app):
        if not self._exploding or self._compute is None:
            return

        self._explosion_time += app.dt

        # Завершение анимации
        if self._explosion_time > self._duration:
            if self._loop:
                self._explosion_time = 0.0
            else:
                self._exploding = False
                return

        ctx = app.ctx

        # Привязываем буферы
        app.pc_ssbo.bind_to_storage_buffer(0)
        app.indirect_buf.bind_to_storage_buffer(1)

        # Dispatch over MAX_POINTS — shader reads draw_count from
        # SSBO and early-exits, avoiding indirect_buf.read() stall.
        from config import MAX_POINTS

        c = self._compute
        c['center'].value = tuple(self._center)
        c['force'].value = float(self._force)
        c['elapsed'].value = float(self._explosion_time)
        c['gravity'].value = float(self._gravity)
        c['spin'].value = float(self._spin)
        c['spread'].value = float(self._spread)
        c['fade'].value = float(self._fade)
        c['max_radius'].value = float(self._max_radius)
        c['max_pt'].value = MAX_POINTS

        c.run((MAX_POINTS + 255) // 256)
        ctx.memory_barrier()

    def on_draw_ui(self, app):
        if imgui.collapsing_header("Point Explosion")[0]:
            # Кнопка взрыва
            if self._exploding:
                if imgui.button("Stop##expl"):
                    self._exploding = False
                imgui.same_line()
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.5, 0.1, 1.0)
                bar = self._explosion_time / max(self._duration, 0.01)
                imgui.text(f"BOOM! {bar*100:.0f}%%")
                imgui.pop_style_color()
            else:
                if imgui.button("EXPLODE!##expl"):
                    # Авто-центр
                    if self._auto_center:
                        self._center = list(app.camera.target)
                    self._trigger_explosion()

            _, self._loop = imgui.checkbox("Loop##expl", self._loop)

            imgui.separator()

            _, self._force = imgui.slider_float(
                "Force", self._force, 0.2, 5.0, "%.1f")
            _, self._duration = imgui.slider_float(
                "Duration (s)", self._duration, 0.5, 10.0, "%.1f")
            _, self._gravity = imgui.slider_float(
                "Gravity", self._gravity, 0.0, 5.0, "%.1f")
            _, self._spin = imgui.slider_float(
                "Spin", self._spin, 0.0, 10.0, "%.1f")
            _, self._spread = imgui.slider_float(
                "Spread", self._spread, 0.0, 2.0, "%.2f")
            _, self._fade = imgui.slider_float(
                "Fade", self._fade, 0.0, 2.0, "%.2f")
            _, self._max_radius = imgui.slider_float(
                "Max Radius", self._max_radius, 1.0, 20.0, "%.1f")

            imgui.separator()
            _, self._auto_center = imgui.checkbox(
                "Auto Center (camera target)", self._auto_center)
            if not self._auto_center:
                _, self._center = imgui.input_float3(
                    "Center##expl", *self._center)

            if self._compute is None:
                imgui.push_style_color(imgui.COLOR_TEXT, 0.95, 0.3, 0.2, 1.0)
                imgui.text("Shader compilation failed!")
                imgui.pop_style_color()

    def get_settings(self):
        return {
            "force": self._force,
            "duration": self._duration,
            "gravity": self._gravity,
            "spin": self._spin,
            "spread": self._spread,
            "fade": self._fade,
            "max_radius": self._max_radius,
            "auto_center": self._auto_center,
            "loop": self._loop,
        }

    def set_settings(self, data):
        self._force = data.get("force", 1.5)
        self._duration = data.get("duration", 3.0)
        self._gravity = data.get("gravity", 1.0)
        self._spin = data.get("spin", 2.0)
        self._spread = data.get("spread", 0.6)
        self._fade = data.get("fade", 0.4)
        self._max_radius = data.get("max_radius", 5.0)
        self._auto_center = data.get("auto_center", True)
        self._loop = data.get("loop", False)
