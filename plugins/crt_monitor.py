# =============================
# Plugin: CRT Monitor Effect
# =============================
"""
Пост-процессинг эффект CRT-монитора.
Применяет к экрану: scanlines, виньетка, RGB-смещение,
искривление экрана (barrel distortion), мерцание.

Демонстрирует создание кастомных шейдеров через плагин.
"""
import struct
import time
import imgui
import moderngl

from plugin_api import KinectPlugin

# Fullscreen triangle (covers screen without quad seam)
_FULLSCREEN_VERT = """
#version 330
out vec2 uv;
void main() {
    // fullscreen triangle trick: 3 vertices, no VBO needed
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
"""

_CRT_FRAG = """
#version 330
in vec2 uv;
out vec4 fragColor;

uniform sampler2D screen_tex;
uniform float time;
uniform vec2 resolution;

// Toggles & params
uniform int scanlines_on;
uniform float scanline_intensity;
uniform float scanline_count;

uniform int vignette_on;
uniform float vignette_strength;

uniform int barrel_on;
uniform float barrel_amount;

uniform int flicker_on;
uniform float flicker_strength;

uniform int rgb_shift_on;
uniform float rgb_shift_amount;

uniform int noise_on;
uniform float noise_strength;

// PCG hash
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float hash(vec2 p, float t) {
    uint h = pcg(floatBitsToUint(p.x) ^ pcg(floatBitsToUint(p.y) ^ pcg(floatBitsToUint(t))));
    return float(h) / 4294967295.0;
}

vec2 barrel_distort(vec2 coord, float amt) {
    vec2 cc = coord - 0.5;
    float r2 = dot(cc, cc);
    return coord + cc * r2 * amt;
}

void main() {
    vec2 tc = uv;

    // Barrel distortion (CRT curvature)
    if (barrel_on == 1) {
        tc = barrel_distort(tc, barrel_amount);
        // Black outside screen
        if (tc.x < 0.0 || tc.x > 1.0 || tc.y < 0.0 || tc.y > 1.0) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
    }

    vec3 color;

    // RGB subpixel shift (chromatic aberration)
    if (rgb_shift_on == 1) {
        float shift = rgb_shift_amount / resolution.x;
        color.r = texture(screen_tex, tc + vec2(shift, 0.0)).r;
        color.g = texture(screen_tex, tc).g;
        color.b = texture(screen_tex, tc - vec2(shift, 0.0)).b;
    } else {
        color = texture(screen_tex, tc).rgb;
    }

    // Scanlines
    if (scanlines_on == 1) {
        float line = sin(tc.y * scanline_count * 3.14159) * 0.5 + 0.5;
        color *= 1.0 - scanline_intensity * (1.0 - line);
    }

    // Film noise
    if (noise_on == 1) {
        float n = hash(gl_FragCoord.xy, time) * 2.0 - 1.0;
        color += vec3(n * noise_strength);
    }

    // Flicker
    if (flicker_on == 1) {
        float flick = 1.0 - flicker_strength * 0.5 * (sin(time * 60.0) * 0.5 + 0.5);
        color *= flick;
    }

    // Vignette
    if (vignette_on == 1) {
        vec2 vc = tc - 0.5;
        float vign = 1.0 - dot(vc, vc) * vignette_strength;
        color *= clamp(vign, 0.0, 1.0);
    }

    // Subtle green phosphor tint
    color *= vec3(0.95, 1.0, 0.95);

    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""


class CRTEffectPlugin(KinectPlugin):
    name = "CRT Monitor"
    version = "1.0"
    author = "Kinect 360"
    description = "Пост-эффект CRT-монитора: scanlines, виньетка, искривление"

    def __init__(self):
        super().__init__()
        self._active = False
        # Sub-effects
        self._scanlines = True
        self._scanline_intensity = 0.3
        self._scanline_count = 400.0
        self._vignette = True
        self._vignette_strength = 2.5
        self._barrel = True
        self._barrel_amount = 0.15
        self._flicker = True
        self._flicker_strength = 0.03
        self._rgb_shift = True
        self._rgb_shift_amount = 1.5
        self._noise = True
        self._noise_strength = 0.04
        # GL resources
        self._prog = None
        self._vao = None
        self._fbo = None
        self._tex = None
        self._fbo_size = (0, 0)
        self._start_time = 0.0

    def on_init(self, app):
        try:
            self._prog = app.ctx.program(
                vertex_shader=_FULLSCREEN_VERT,
                fragment_shader=_CRT_FRAG,
            )
            # Empty VAO for fullscreen triangle (no VBO needed)
            self._vao = app.ctx.vertex_array(self._prog, [])
            self._start_time = time.perf_counter()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("CRT plugin shader error: %s", e)
            self._prog = None

    def _ensure_fbo(self, ctx, w, h):
        """Пересоздаёт FBO при изменении размера окна."""
        if self._fbo_size == (w, h) and self._fbo is not None:
            return
        if self._fbo is not None:
            self._fbo.release()
            self._tex.release()
        self._tex = ctx.texture((w, h), 3)
        self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._fbo = ctx.framebuffer(color_attachments=[self._tex])
        self._fbo_size = (w, h)

    def on_post_render(self, app):
        if not self._active or self._prog is None:
            return

        w, h = app.width, app.height
        if w < 1 or h < 1:
            return

        ctx = app.ctx
        self._ensure_fbo(ctx, w, h)

        # Копируем экран → наш FBO
        ctx.copy_framebuffer(self._fbo, ctx.screen)

        # Рендерим обратно на экран с CRT-шейдером
        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)
        ctx.disable(moderngl.DEPTH_TEST)

        self._tex.use(location=0)
        p = self._prog
        p['screen_tex'].value = 0
        p['time'].value = float(time.perf_counter() - self._start_time)
        p['resolution'].value = (float(w), float(h))

        p['scanlines_on'].value = int(self._scanlines)
        p['scanline_intensity'].value = float(self._scanline_intensity)
        p['scanline_count'].value = float(self._scanline_count)
        p['vignette_on'].value = int(self._vignette)
        p['vignette_strength'].value = float(self._vignette_strength)
        p['barrel_on'].value = int(self._barrel)
        p['barrel_amount'].value = float(self._barrel_amount)
        p['flicker_on'].value = int(self._flicker)
        p['flicker_strength'].value = float(self._flicker_strength)
        p['rgb_shift_on'].value = int(self._rgb_shift)
        p['rgb_shift_amount'].value = float(self._rgb_shift_amount)
        p['noise_on'].value = int(self._noise)
        p['noise_strength'].value = float(self._noise_strength)

        self._vao.render(moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)

    def on_draw_ui(self, app):
        if imgui.collapsing_header("CRT Monitor")[0]:
            changed, val = imgui.checkbox("Active##crt", self._active)
            if changed:
                self._active = val

            if self._active:
                _, self._scanlines = imgui.checkbox("Scanlines", self._scanlines)
                if self._scanlines:
                    _, self._scanline_intensity = imgui.slider_float(
                        "Line Intensity", self._scanline_intensity, 0.05, 1.0, "%.2f")
                    _, self._scanline_count = imgui.slider_float(
                        "Line Count", self._scanline_count, 100, 1200, "%.0f")

                _, self._vignette = imgui.checkbox("Vignette", self._vignette)
                if self._vignette:
                    _, self._vignette_strength = imgui.slider_float(
                        "Vignette##str", self._vignette_strength, 0.5, 6.0, "%.1f")

                _, self._barrel = imgui.checkbox("Barrel Distortion", self._barrel)
                if self._barrel:
                    _, self._barrel_amount = imgui.slider_float(
                        "Curvature", self._barrel_amount, 0.02, 0.5, "%.2f")

                _, self._rgb_shift = imgui.checkbox("RGB Shift", self._rgb_shift)
                if self._rgb_shift:
                    _, self._rgb_shift_amount = imgui.slider_float(
                        "Shift##rgb", self._rgb_shift_amount, 0.5, 5.0, "%.1f")

                _, self._flicker = imgui.checkbox("Flicker", self._flicker)
                if self._flicker:
                    _, self._flicker_strength = imgui.slider_float(
                        "Flicker##str", self._flicker_strength, 0.01, 0.15, "%.2f")

                _, self._noise = imgui.checkbox("Film Noise", self._noise)
                if self._noise:
                    _, self._noise_strength = imgui.slider_float(
                        "Noise##str", self._noise_strength, 0.01, 0.15, "%.2f")

                if self._prog is None:
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.95, 0.3, 0.2, 1.0)
                    imgui.text("Shader compilation failed!")
                    imgui.pop_style_color()

    def on_cleanup(self, app):
        if self._fbo is not None:
            self._fbo.release()
        if self._tex is not None:
            self._tex.release()
        # prog и vao — moderngl освободит при ctx.release()

    def get_settings(self):
        return {
            "active": self._active,
            "scanlines": self._scanlines,
            "scanline_intensity": self._scanline_intensity,
            "scanline_count": self._scanline_count,
            "vignette": self._vignette,
            "vignette_strength": self._vignette_strength,
            "barrel": self._barrel,
            "barrel_amount": self._barrel_amount,
            "flicker": self._flicker,
            "flicker_strength": self._flicker_strength,
            "rgb_shift": self._rgb_shift,
            "rgb_shift_amount": self._rgb_shift_amount,
            "noise": self._noise,
            "noise_strength": self._noise_strength,
        }

    def set_settings(self, data):
        self._active = data.get("active", False)
        self._scanlines = data.get("scanlines", True)
        self._scanline_intensity = data.get("scanline_intensity", 0.3)
        self._scanline_count = data.get("scanline_count", 400.0)
        self._vignette = data.get("vignette", True)
        self._vignette_strength = data.get("vignette_strength", 2.5)
        self._barrel = data.get("barrel", True)
        self._barrel_amount = data.get("barrel_amount", 0.15)
        self._flicker = data.get("flicker", True)
        self._flicker_strength = data.get("flicker_strength", 0.03)
        self._rgb_shift = data.get("rgb_shift", True)
        self._rgb_shift_amount = data.get("rgb_shift_amount", 1.5)
        self._noise = data.get("noise", True)
        self._noise_strength = data.get("noise_strength", 0.04)
