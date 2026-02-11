# =============================
# Plugin: Auto Orbit
# =============================
"""
Автоматическое вращение камеры вокруг облака точек.
Создаёт кинематографический эффект для демонстраций/инсталляций.

Управление через ImGui-панель:
  - Скорость и направление вращения
  - Вертикальное покачивание (bob)
  - Авто-зум (дыхание)
  - Пауза по клику мыши
"""
import math
import imgui

from plugin_api import KinectPlugin


class AutoOrbitPlugin(KinectPlugin):
    name = "Auto Orbit"
    version = "1.0"
    author = "Kinect 360"
    description = "Кинематографическое авто-вращение камеры вокруг сцены"

    def __init__(self):
        super().__init__()
        self._active = False
        self._speed = 0.3          # оборотов в минуту
        self._direction = 1        # 1 = по часовой, -1 = против
        self._bob_enabled = True   # вертикальное покачивание
        self._bob_amp = 0.15       # амплитуда покачивания (рад)
        self._bob_speed = 0.4      # скорость покачивания
        self._breathe = True       # авто-зум "дыхание"
        self._breathe_amp = 0.3    # амплитуда зума
        self._breathe_speed = 0.25 # скорость зума
        self._time = 0.0
        self._paused_by_mouse = False
        self._base_distance = None
        self._base_pitch = None

    def on_frame_start(self, app, dt):
        if not self._active:
            return

        import glfw
        # Пауза при зажатой мышке (ручное управление)
        lmb = glfw.get_mouse_button(app.window, glfw.MOUSE_BUTTON_LEFT)
        rmb = glfw.get_mouse_button(app.window, glfw.MOUSE_BUTTON_RIGHT)
        if lmb == glfw.PRESS or rmb == glfw.PRESS:
            self._paused_by_mouse = True
            return
        elif self._paused_by_mouse:
            # Мышь отпущена — обновляем базовые значения
            self._paused_by_mouse = False
            self._base_distance = app.camera.distance
            self._base_pitch = app.camera.pitch

        self._time += dt

        # Сохраняем базовые значения при первом кадре
        if self._base_distance is None:
            self._base_distance = app.camera.distance
        if self._base_pitch is None:
            self._base_pitch = app.camera.pitch

        # Вращение (yaw)
        rps = self._speed / 60.0  # об/мин → об/сек
        app.camera.yaw += self._direction * rps * 2.0 * math.pi * dt

        # Вертикальное покачивание (pitch)
        if self._bob_enabled:
            target_pitch = self._base_pitch + math.sin(
                self._time * self._bob_speed * 2.0 * math.pi
            ) * self._bob_amp
            target_pitch = max(-math.pi / 2 + 0.05,
                               min(math.pi / 2 - 0.05, target_pitch))
            app.camera.pitch = target_pitch

        # Дыхание (zoom)
        if self._breathe:
            offset = math.sin(
                self._time * self._breathe_speed * 2.0 * math.pi
            ) * self._breathe_amp
            app.camera.distance = max(0.5, self._base_distance + offset)

    def on_draw_ui(self, app):
        if imgui.collapsing_header("Auto Orbit")[0]:
            changed, val = imgui.checkbox("Active##orbit", self._active)
            if changed:
                self._active = val
                if val:
                    # Запоминаем текущую позицию камеры
                    self._base_distance = app.camera.distance
                    self._base_pitch = app.camera.pitch
                    self._time = 0.0

            if self._active:
                _, self._speed = imgui.slider_float(
                    "Speed (rpm)", self._speed, 0.05, 3.0, "%.2f")

                if imgui.button("CW##orb"):
                    self._direction = 1
                imgui.same_line()
                if imgui.button("CCW##orb"):
                    self._direction = -1
                imgui.same_line()
                imgui.text("CW" if self._direction == 1 else "CCW")

                imgui.separator()

                _, self._bob_enabled = imgui.checkbox("Vertical Bob", self._bob_enabled)
                if self._bob_enabled:
                    _, self._bob_amp = imgui.slider_float(
                        "Bob Amplitude", self._bob_amp, 0.02, 0.5, "%.2f")
                    _, self._bob_speed = imgui.slider_float(
                        "Bob Speed", self._bob_speed, 0.05, 2.0, "%.2f")

                _, self._breathe = imgui.checkbox("Zoom Breathe", self._breathe)
                if self._breathe:
                    _, self._breathe_amp = imgui.slider_float(
                        "Breathe Amp", self._breathe_amp, 0.1, 2.0, "%.2f")
                    _, self._breathe_speed = imgui.slider_float(
                        "Breathe Speed", self._breathe_speed, 0.05, 1.0, "%.2f")

                if self._paused_by_mouse:
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.95, 0.75, 0.2, 1.0)
                    imgui.text("Paused (mouse)")
                    imgui.pop_style_color()

    def get_settings(self):
        return {
            "active": self._active,
            "speed": self._speed,
            "direction": self._direction,
            "bob_enabled": self._bob_enabled,
            "bob_amp": self._bob_amp,
            "bob_speed": self._bob_speed,
            "breathe": self._breathe,
            "breathe_amp": self._breathe_amp,
            "breathe_speed": self._breathe_speed,
        }

    def set_settings(self, data):
        self._active = data.get("active", False)
        self._speed = data.get("speed", 0.3)
        self._direction = data.get("direction", 1)
        self._bob_enabled = data.get("bob_enabled", True)
        self._bob_amp = data.get("bob_amp", 0.15)
        self._bob_speed = data.get("bob_speed", 0.4)
        self._breathe = data.get("breathe", True)
        self._breathe_amp = data.get("breathe_amp", 0.3)
        self._breathe_speed = data.get("breathe_speed", 0.25)
