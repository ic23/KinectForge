# =============================
# Plugin: Smoke Dissolve (Частицы в дым)
# =============================
"""
Превращает ВСЕ частицы (точки/кружки) облака точек Kinect
в дым, двигаясь сверху вниз.

Фронт растворения опускается по всей сцене.
Частицы выше фронта → превращаются в дым (поднимаются вверх
с турбулентностью, завихрениями, расширяются и затухают).
Частицы ниже фронта → остаются на месте.

GPU compute shader модифицирует SSBO in-place.
"""
import imgui

from plugin_api import KinectPlugin

_SMOKE_COMPUTE = """
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Points {
    float data[];   // interleaved: x, y, z, r, g, b  per point
};

layout(std430, binding = 1) buffer DrawIndirect {
    uint draw_count;
    uint draw_instance_count;
    uint draw_first;
    uint draw_base_instance;
};

// ── Фронт растворения ──
uniform float front_y;          // текущая Y фронта (опускается)
uniform float front_soft;       // мягкость края фронта (м)
uniform float sweep_top;        // верх диапазона (Y, откуда начинался фронт)
uniform float sweep_range;      // полный диапазон Y (top - bottom)
uniform float sweep_duration;   // время полного прохода фронта (сек)
uniform float total_elapsed;    // общее время с начала эффекта (сек)

// ── Дым ──
uniform float rise_speed;       // скорость подъёма
uniform float turbulence;       // сила турбулентности
uniform float expand;           // расширение в стороны
uniform float fade_speed;       // скорость затухания
uniform float swirl;            // завихрение
uniform float smoke_density;    // доля затронутых точек [0..1]
uniform vec3  smoke_color;      // целевой цвет дыма
uniform float color_mix;        // сила подмешивания цвета
uniform uint  max_pt;

// ── PCG hash ──
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

    // Пропускаем уже убранные точки
    if (pos.x > 9000.0) return;

    // ── Фронт растворения (сверху вниз) ──
    float above = pos.y - front_y;
    if (above < -front_soft) return;  // ниже фронта — не трогаем

    // Мягкий переход на границе (0 = твёрдая, 1 = полностью дым)
    float dissolve_t = clamp(above / max(front_soft, 0.001) * 0.5 + 0.5, 0.0, 1.0);

    // Контроль плотности
    float density_check = hashf(idx * 53u + 17u);
    if (density_check > smoke_density) return;

    // ── Вычисляем, сколько ВРЕМЕНИ прошло с момента растворения этой точки ──
    // Фронт прошёл через pos.y в момент: t_dissolve = (sweep_top - pos.y) / sweep_range * sweep_duration
    float point_progress = clamp((sweep_top - pos.y) / max(sweep_range, 0.001), 0.0, 1.0);
    float time_dissolved = point_progress * sweep_duration;
    float time_in_smoke = max(total_elapsed - time_dissolved, 0.0);

    // «Личные» параметры каждой частицы
    float life_hash  = hashf(idx * 31u + 7u);
    float speed_var  = 0.6 + hashf(idx * 37u + 11u) * 0.8;
    float turb_phase = hashf(idx * 43u + 13u) * 6.2832;

    float t = time_in_smoke * speed_var;
    if (t <= 0.001) return;

    // ─── Подъём (ускоряющийся как настоящий дым) ───
    float rise = t * rise_speed * (1.0 + t * 0.3);

    // ─── Турбулентность ───
    float turb_x = sin(turb_phase + t * 2.7 + life_hash * 3.0) * turbulence;
    float turb_z = cos(turb_phase * 1.3 + t * 3.1 + life_hash * 5.0) * turbulence;
    float turb_grow = 1.0 + t * 0.5;
    turb_x *= turb_grow;
    turb_z *= turb_grow;

    // Медленная низкочастотная турбулентность
    float slow_turb = sin(t * 0.7 + life_hash * 6.28) * turbulence * 0.6;

    // ─── Расширение ───
    float angle_spread = hashf(idx * 67u + 1u) * 6.2832;
    float expand_r = t * expand * (0.6 + life_hash * 0.8);
    float spread_x = cos(angle_spread) * expand_r;
    float spread_z = sin(angle_spread) * expand_r;

    // ─── Завихрение ───
    float swirl_angle = swirl * t * (0.6 + hashf(idx * 79u + 19u) * 0.8);
    float cs = cos(swirl_angle);
    float sn = sin(swirl_angle);

    vec3 offset;
    offset.x = turb_x + spread_x + slow_turb;
    offset.z = turb_z + spread_z;
    float ox = offset.x * cs - offset.z * sn;
    float oz = offset.x * sn + offset.z * cs;
    offset.x = ox;
    offset.z = oz;
    offset.y = rise + abs(slow_turb) * 0.2;

    // Применяем смещение с учётом dissolve_t (на границе — частичное)
    pos += offset * dissolve_t;

    // ─── Затухание ───
    float alpha = 1.0 - clamp(t * fade_speed, 0.0, 1.0);
    alpha = alpha * alpha;
    alpha = mix(1.0, alpha, dissolve_t);

    // ─── Цвет дыма ───
    float cmix = clamp(color_mix * dissolve_t * (0.2 + t * 0.5), 0.0, 1.0);
    col = mix(col, smoke_color, cmix);

    // Тёплый оттенок в начале
    float heat = max(1.0 - t * 1.5, 0.0) * dissolve_t * 0.15;
    col.r = min(col.r + heat, 1.0);
    col.g = min(col.g + heat * 0.2, 1.0);

    col *= alpha;

    // Полностью затухшие → убираем
    if (alpha < 0.01) {
        pos = vec3(9999.0, 9999.0, 9999.0);
        col = vec3(0.0);
    }

    data[base]     = pos.x;
    data[base+1u]  = pos.y;
    data[base+2u]  = pos.z;
    data[base+3u]  = col.r;
    data[base+4u]  = col.g;
    data[base+5u]  = col.b;
}
"""


class SmokeMugPlugin(KinectPlugin):
    name = "Smoke Dissolve"
    version = "3.0"
    author = "Kinect 360"
    description = "Растворяет все частицы облака точек в дым сверху вниз"

    def __init__(self):
        super().__init__()
        self._compute = None
        self._active = False
        self._elapsed = 0.0

        # Диапазон Y сцены
        self._y_top = 1.0          # верхняя граница (м)
        self._y_bottom = -1.0      # нижняя граница (м)

        # Фронт растворения
        self._front_soft = 0.08    # мягкость края фронта (м)

        # Параметры дыма
        self._rise_speed = 0.35    # скорость подъёма
        self._turbulence = 0.05    # сила турбулентности
        self._expand = 0.06        # расширение в стороны
        self._fade_speed = 0.3     # скорость затухания
        self._swirl = 1.0          # завихрение
        self._density = 1.0        # доля точек
        self._smoke_color = [0.6, 0.6, 0.65]  # цвет дыма
        self._color_mix = 0.5      # подмешивание цвета
        self._duration = 6.0       # длительность (сек)
        self._loop = False

    def on_init(self, app):
        try:
            self._compute = app.ctx.compute_shader(_SMOKE_COMPUTE)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Smoke Dissolve plugin shader error: %s", e)
            self._compute = None

    def _start_effect(self):
        self._active = True
        self._elapsed = 0.0

    def _stop_effect(self):
        self._active = False
        self._elapsed = 0.0

    def on_pre_render(self, app):
        if not self._active or self._compute is None:
            return

        self._elapsed += app.dt

        if self._elapsed > self._duration:
            if self._loop:
                self._elapsed = 0.0
            else:
                self._active = False
                return

        app.pc_ssbo.bind_to_storage_buffer(0)
        app.indirect_buf.bind_to_storage_buffer(1)

        # Dispatch over MAX_POINTS — shader reads draw_count from
        # SSBO and early-exits, avoiding indirect_buf.read() stall.
        from config import MAX_POINTS
        n_pts = MAX_POINTS

        y_top = self._y_top
        y_bot = self._y_bottom
        y_range = max(y_top - y_bot, 0.01)

        # Время полного прохода фронта = 70% длительности
        sweep_duration = self._duration * 0.7

        # Фронт опускается от y_top до y_bottom
        progress = min(self._elapsed / max(sweep_duration, 0.01), 1.0)
        front_y = y_top - progress * y_range

        c = self._compute
        c['front_y'].value = float(front_y)
        c['front_soft'].value = float(self._front_soft)
        c['sweep_top'].value = float(y_top)
        c['sweep_range'].value = float(y_range)
        c['sweep_duration'].value = float(sweep_duration)
        c['total_elapsed'].value = float(self._elapsed)
        c['rise_speed'].value = float(self._rise_speed)
        c['turbulence'].value = float(self._turbulence)
        c['expand'].value = float(self._expand)
        c['fade_speed'].value = float(self._fade_speed)
        c['swirl'].value = float(self._swirl)
        c['smoke_density'].value = float(self._density)
        c['smoke_color'].value = tuple(self._smoke_color)
        c['color_mix'].value = float(self._color_mix)
        c['max_pt'].value = n_pts

        c.run((n_pts + 255) // 256)
        app.ctx.memory_barrier()

    def on_draw_ui(self, app):
        expanded, _ = imgui.collapsing_header("Smoke Dissolve (Частицы в дым)")
        if not expanded:
            return

        # ── Кнопка запуска ──
        if self._active:
            if imgui.button("Stop##smoke"):
                self._stop_effect()
            imgui.same_line()
            imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.9, 1.0)
            bar = self._elapsed / max(self._duration, 0.01)
            imgui.text(f"~ dissolving ~ {bar*100:.0f}%%")
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.3, 0.5, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.4, 0.4, 0.65, 1.0)
            if imgui.button("Dissolve to Smoke!##smoke"):
                self._start_effect()
            imgui.pop_style_color(2)

        _, self._loop = imgui.checkbox("Loop##smoke", self._loop)

        imgui.separator()
        imgui.text("Scene Y Bounds")

        _, self._y_top = imgui.slider_float(
            "Y Top##smoke", self._y_top, -2.0, 3.0, "%.2f m")
        _, self._y_bottom = imgui.slider_float(
            "Y Bottom##smoke", self._y_bottom, -3.0, 2.0, "%.2f m")

        imgui.separator()
        imgui.text("Dissolve Front")

        _, self._front_soft = imgui.slider_float(
            "Edge Softness##smoke", self._front_soft, 0.01, 0.5, "%.3f m")
        _, self._duration = imgui.slider_float(
            "Duration (s)##smoke", self._duration, 1.0, 20.0, "%.1f")

        imgui.separator()
        imgui.text("Smoke")

        _, self._rise_speed = imgui.slider_float(
            "Rise Speed##smoke", self._rise_speed, 0.05, 2.0, "%.2f")
        _, self._turbulence = imgui.slider_float(
            "Turbulence##smoke", self._turbulence, 0.0, 0.3, "%.3f")
        _, self._expand = imgui.slider_float(
            "Expand##smoke", self._expand, 0.0, 0.5, "%.3f")
        _, self._swirl = imgui.slider_float(
            "Swirl##smoke", self._swirl, 0.0, 5.0, "%.1f")
        _, self._fade_speed = imgui.slider_float(
            "Fade Speed##smoke", self._fade_speed, 0.05, 2.0, "%.2f")
        _, self._density = imgui.slider_float(
            "Density##smoke", self._density, 0.1, 1.0, "%.2f")

        imgui.separator()
        imgui.text("Color")

        _, self._smoke_color = imgui.color_edit3(
            "Smoke Color##smoke", *self._smoke_color)
        self._smoke_color = list(self._smoke_color)
        _, self._color_mix = imgui.slider_float(
            "Color Mix##smoke", self._color_mix, 0.0, 1.0, "%.2f")

        if self._compute is None:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.95, 0.3, 0.2, 1.0)
            imgui.text("Shader compilation failed!")
            imgui.pop_style_color()

    def get_settings(self):
        return {
            "y_top": self._y_top,
            "y_bottom": self._y_bottom,
            "front_soft": self._front_soft,
            "rise_speed": self._rise_speed,
            "turbulence": self._turbulence,
            "expand": self._expand,
            "fade_speed": self._fade_speed,
            "swirl": self._swirl,
            "density": self._density,
            "smoke_color": self._smoke_color,
            "color_mix": self._color_mix,
            "duration": self._duration,
            "loop": self._loop,
        }

    def set_settings(self, data):
        self._y_top = data.get("y_top", 1.0)
        self._y_bottom = data.get("y_bottom", -1.0)
        self._front_soft = data.get("front_soft", 0.08)
        self._rise_speed = data.get("rise_speed", 0.35)
        self._turbulence = data.get("turbulence", 0.05)
        self._expand = data.get("expand", 0.06)
        self._fade_speed = data.get("fade_speed", 0.3)
        self._swirl = data.get("swirl", 1.0)
        self._density = data.get("density", 1.0)
        self._smoke_color = data.get("smoke_color", [0.6, 0.6, 0.65])
        self._color_mix = data.get("color_mix", 0.5)
        self._duration = data.get("duration", 6.0)
        self._loop = data.get("loop", False)

    def on_cleanup(self, app):
        self._compute = None
