# Kinect360Viewer — Plugins

Плагины автоматически загружаются из этой папки при запуске.

## Встроенные плагины

| Плагин | Версия | Описание |
|--------|--------|----------|
| **Auto Orbit** | 1.0 | Кинематографическое авто-вращение камеры вокруг облака точек. Скорость, направление, вертикальная качка, zoom breathe, пауза при перетаскивании мышью. |
| **CRT Monitor** | 1.0 | Post-effect CRT монитора: scanlines, vignette, barrel distortion, flicker, RGB shift, film noise. Создаёт собственный shader program и FBO. |
| **FPS Overlay** | 1.0 | Прозрачный оверлей с FPS и количеством точек. GPU readback ограничен до 2 раз/сек. Настраиваемая позиция (угол экрана). |
| **Point Explosion** | 1.0 | GPU compute shader разлёт облака из центра. Гравитация, вращение, разброс, затухание, макс. радиус. Модифицирует SSBO in-place. |
| **Smoke Dissolve** | 3.0 | Растворение облака в восходящий дым с moving front сверху вниз. GPU compute: турбулентность, вихри, расширение, затухание, настраиваемый цвет дыма. |

## Как создать плагин

1. Создайте `.py` файл в этой папке (имена с `_` в начале игнорируются)
2. Импортируйте базовый класс:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plugin_api import KinectPlugin
```

3. Создайте класс-наследник `KinectPlugin`:

```python
class MyPlugin(KinectPlugin):
    name = "My Plugin"
    version = "1.0"
    author = "Your Name"
    description = "What it does"

    def on_init(self, app):
        # Инициализация (GL контекст доступен через app.ctx)
        pass

    def on_draw_ui(self, app):
        # Нарисовать ImGui элементы
        import imgui
        if imgui.collapsing_header("My Plugin")[0]:
            imgui.text("Hello from plugin!")

    def on_cleanup(self, app):
        # Освобождение ресурсов
        pass
```

## Доступные хуки (все опциональны)

| Хук | Когда вызывается |
|-----|-----------------|
| `on_init(app)` | После инициализации GL, перед главным циклом |
| `on_cleanup(app)` | При завершении программы |
| `on_frame_start(app, dt)` | Начало каждого кадра |
| `on_kinect_frame(app, depth, color)` | Новый кадр Kinect (`depth`: uint16 mm, `color`: BGR uint8) |
| `on_pre_render(app)` | Перед 3D-рендером (после compute dispatch) |
| `on_post_render(app)` | После composite pass, перед ImGui |
| `on_draw_ui(app)` | Внутри ImGui (рисуйте свои панели) |
| `on_export(app, xyz, rgb, path)` | После экспорта PLY |
| `on_settings_changed(app, key, old, new)` | При изменении настройки через UI |
| `on_preset_save(app, name)` | При сохранении пресета |
| `on_preset_load(app, name)` | После загрузки пресета |

## Объект `app` (AppContext)

| Поле | Тип | Описание |
|------|-----|----------|
| `app.ctx` | `moderngl.Context` | OpenGL контекст |
| `app.window` | GLFW window | Окно приложения |
| `app.camera` | `OrbitCamera` | Камера (orbit/pan/zoom, MVP, frustum planes) |
| `app.settings` | `dict` | Все настройки (config.settings) |
| `app.sensor` | `KinectSensor` | Kinect сенсор (.NET объект) |
| `app.width` | `int` | Ширина framebuffer |
| `app.height` | `int` | Высота framebuffer |
| `app.time` | `float` | Текущее время (`perf_counter`) |
| `app.dt` | `float` | Delta-time в секундах |
| `app.fps` | `int` | Текущий FPS |
| `app.pc_ssbo` | `moderngl.Buffer` | GPU SSBO облака точек (xyz+rgb, 6 float per point) |
| `app.indirect_buf` | `moderngl.Buffer` | Indirect draw buffer (first uint = point count) |

### Работа с GPU данными

```python
# Чтение количества точек (ОСТОРОЖНО — это GPU readback, вызывает stall!)
# Ограничивайте чтение до ≤ 2 раз/сек
import struct
count = struct.unpack('I', app.indirect_buf.read(4))[0]

# Чтение облака точек (полный stall — только для экспорта)
data = app.pc_ssbo.read(count * 6 * 4)
```

> **Важно:** Любой `buffer.read()` с GPU вызывает pipeline stall. Для per-frame данных передавайте значения через CPU-переменные. Для GPU-модификации точек используйте compute shader (см. `point_explosion.py` и `smoke_mug.py`).

## Сохранение настроек плагина в пресеты

Переопределите `get_settings()` и `set_settings(data)` — ваши данные
автоматически сохранятся/загрузятся вместе с пресетами.

```python
def get_settings(self):
    return {"speed": self.speed, "enabled": self.enabled}

def set_settings(self, data):
    self.speed = data.get("speed", 1.0)
    self.enabled = data.get("enabled", False)
```

## Горячая перезагрузка

Нажмите **"Reload Plugins"** в панели Plugins в UI — все плагины
будут выгружены (`on_cleanup`) и загружены заново (`on_init`) без перезапуска приложения.

## Примеры

| Пример | Что демонстрирует |
|--------|-------------------|
| `fps_overlay.py` | Простой UI оверлей, throttled GPU readback |
| `auto_orbit.py` | Анимация камеры, `get_settings`/`set_settings` |
| `crt_monitor.py` | Собственный shader + FBO в post-render |
| `point_explosion.py` | GPU compute shader, модификация SSBO |
| `smoke_mug.py` | Сложный GPU compute эффект с настройками |
