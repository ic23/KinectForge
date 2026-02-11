# =============================
# Kinect 360 — Plugin API
# =============================
"""
Базовый класс плагина и менеджер плагинов для Kinect 360 3D Viewer.

Плагины — это Python-модули в папке plugins/, содержащие класс,
унаследованный от KinectPlugin. Менеджер автоматически загружает
все плагины при старте и вызывает lifecycle-хуки в нужные моменты.

Lifecycle-хуки (все опциональны):
    on_init(app)              — после инициализации GL/GPU, перед main loop
    on_cleanup(app)           — при завершении, освобождение ресурсов
    on_frame_start(app, dt)   — начало кадра (после poll_events)
    on_kinect_frame(app, depth, color)
                              — новый кадр Kinect (до GPU-обработки)
    on_pre_render(app)        — перед 3D-рендером (после compute pipeline)
    on_post_render(app)       — после composite pass, перед imgui
    on_draw_ui(app)           — внутри ImGui-кадра (свои панели/настройки)
    on_export(app, xyz, rgb, path)
                              — после сохранения PLY (доп. форматы / обработка)
    on_settings_changed(app, key, old_value, new_value)
                              — при изменении settings через UI
    on_preset_save(app, name) — при сохранении пресета
    on_preset_load(app, name) — после загрузки пресета

Объект `app` предоставляет доступ к:
    app.ctx         — moderngl context
    app.window      — GLFW window
    app.camera      — OrbitCamera
    app.settings    — словарь настроек (config.settings)
    app.sensor      — KinectSensor
    app.width       — ширина framebuffer
    app.height      — высота framebuffer
    app.time        — time.perf_counter() текущего кадра
    app.dt          — delta-time (секунды)
    app.fps         — текущий FPS
    app.pc_ssbo     — GPU SSBO облака точек (moderngl Buffer)
    app.indirect_buf — GPU indirect draw buffer
"""

import os
import sys
import importlib
import importlib.util
import logging
import traceback
from typing import List, Optional, Dict, Any

log = logging.getLogger(__name__)


class KinectPlugin:
    """Базовый класс плагина. Переопределите нужные методы."""

    # Метаданные (переопределите в своём плагине)
    name: str = "Unnamed Plugin"
    version: str = "1.0"
    author: str = ""
    description: str = ""

    # Управление (менеджер ставит)
    enabled: bool = True

    def on_init(self, app):
        """Вызывается после инициализации GL, перед главным циклом."""
        pass

    def on_cleanup(self, app):
        """Вызывается при завершении. Освободите свои GPU-ресурсы."""
        pass

    def on_frame_start(self, app, dt: float):
        """Начало кадра. dt — время с прошлого кадра в секундах."""
        pass

    def on_kinect_frame(self, app, depth, color):
        """Новый кадр Kinect. depth: np.uint16 (mm), color: np.uint8 BGR."""
        pass

    def on_pre_render(self, app):
        """Перед 3D-рендером (после GPU compute pipeline)."""
        pass

    def on_post_render(self, app):
        """После composite pass, перед imgui."""
        pass

    def on_draw_ui(self, app):
        """Рисуйте свои ImGui-элементы здесь."""
        pass

    def on_export(self, app, xyz, rgb, path: str):
        """После экспорта PLY. xyz: float32 (N,3), rgb: uint8 (N,3)."""
        pass

    def on_settings_changed(self, app, key: str, old_value, new_value):
        """Настройка settings[key] изменилась."""
        pass

    def on_preset_save(self, app, name: str):
        """Пресет сохранён."""
        pass

    def on_preset_load(self, app, name: str):
        """Пресет загружен."""
        pass

    def get_settings(self) -> Dict[str, Any]:
        """Верните dict собственных настроек для сохранения в пресеты.
        Возвращённые ключи будут сохранены с префиксом 'plugin.<name>.'"""
        return {}

    def set_settings(self, data: Dict[str, Any]):
        """Получите dict настроек при загрузке пресета.
        Ключи те же, что вернули из get_settings()."""
        pass


class AppContext:
    """Объект-контейнер, передаваемый плагинам.
    Предоставляет безопасный доступ к состоянию приложения."""

    def __init__(self):
        self.ctx = None          # moderngl.Context
        self.window = None       # GLFW window handle
        self.camera = None       # OrbitCamera
        self.settings = None     # config.settings dict
        self.sensor = None       # KinectSensor
        self.width: int = 0
        self.height: int = 0
        self.time: float = 0.0
        self.dt: float = 0.0
        self.fps: int = 0
        self.pc_ssbo = None      # GPU point cloud SSBO
        self.indirect_buf = None  # GPU indirect draw buffer


class PluginManager:
    """Загружает, управляет и вызывает lifecycle-хуки плагинов."""

    def __init__(self, plugins_dir: str):
        self.plugins_dir = plugins_dir
        self.plugins: List[KinectPlugin] = []
        self._modules: Dict[str, Any] = {}  # filename → module
        self._plugin_map: Dict[str, KinectPlugin] = {}  # name → plugin
        self._load_errors: List[str] = []
        os.makedirs(plugins_dir, exist_ok=True)

    # ── Загрузка ──

    def discover_and_load(self):
        """Сканирует plugins/ и загружает все .py файлы с KinectPlugin."""
        self.plugins.clear()
        self._modules.clear()
        self._plugin_map.clear()
        self._load_errors.clear()

        if not os.path.isdir(self.plugins_dir):
            return

        for fname in sorted(os.listdir(self.plugins_dir)):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            self._load_plugin_file(fname)

        log.info("Plugins loaded: %d (%s)",
                 len(self.plugins),
                 ", ".join(p.name for p in self.plugins) or "none")

    def _load_plugin_file(self, fname: str):
        """Загружает один файл плагина."""
        fpath = os.path.join(self.plugins_dir, fname)
        mod_name = f"_kinect_plugin_{fname[:-3]}"

        # Ensure the project root is importable so plugins can do
        # ``from plugin_api import KinectPlugin`` without sys.path hacks.
        project_root = os.path.dirname(self.plugins_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            spec = importlib.util.spec_from_file_location(mod_name, fpath)
            if spec is None or spec.loader is None:
                self._load_errors.append(f"{fname}: не удалось создать spec")
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)

            # Ищем все классы-наследники KinectPlugin
            found = False
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type)
                        and issubclass(attr, KinectPlugin)
                        and attr is not KinectPlugin):
                    plugin = attr()
                    self.plugins.append(plugin)
                    self._plugin_map[plugin.name] = plugin
                    found = True
                    log.info("  Plugin: %s v%s (%s)",
                             plugin.name, plugin.version, fname)

            if not found:
                self._load_errors.append(
                    f"{fname}: нет классов KinectPlugin")

            self._modules[fname] = module

        except Exception as e:
            tb = traceback.format_exc()
            self._load_errors.append(f"{fname}: {e}")
            log.error("Plugin load error (%s): %s\n%s", fname, e, tb)

    def reload_plugin(self, fname: str):
        """Горячая перезагрузка плагина по имени файла."""
        # Удаляем старые плагины из этого файла
        if fname in self._modules:
            old_module = self._modules[fname]
            self.plugins = [
                p for p in self.plugins
                if type(p).__module__ != old_module.__name__
            ]
            # Удаляем из map
            self._plugin_map = {
                n: p for n, p in self._plugin_map.items()
                if type(p).__module__ != old_module.__name__
            }
            mod_name = old_module.__name__
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            del self._modules[fname]

        # Очищаем ошибки для этого файла
        self._load_errors = [
            e for e in self._load_errors if not e.startswith(fname)
        ]

        self._load_plugin_file(fname)

    def reload_all(self, app: AppContext):
        """Перезагружает все плагины. Вызывает cleanup→discover→init."""
        # Cleanup
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_cleanup", app)

        self.discover_and_load()

        # Init
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_init", app)

    # ── Lifecycle dispatch ──

    def _safe_call(self, plugin: KinectPlugin, method: str, *args, **kwargs):
        """Безопасный вызов метода плагина с обработкой ошибок."""
        try:
            fn = getattr(plugin, method, None)
            if fn is not None:
                fn(*args, **kwargs)
        except Exception as e:
            log.error("Plugin '%s'.%s() error: %s", plugin.name, method, e)
            log.debug(traceback.format_exc())

    def call_init(self, app: AppContext):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_init", app)

    def call_cleanup(self, app: AppContext):
        for p in self.plugins:
            self._safe_call(p, "on_cleanup", app)

    def call_frame_start(self, app: AppContext, dt: float):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_frame_start", app, dt)

    def call_kinect_frame(self, app: AppContext, depth, color):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_kinect_frame", app, depth, color)

    def call_pre_render(self, app: AppContext):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_pre_render", app)

    def call_post_render(self, app: AppContext):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_post_render", app)

    def call_draw_ui(self, app: AppContext):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_draw_ui", app)

    def call_export(self, app: AppContext, xyz, rgb, path: str):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_export", app, xyz, rgb, path)

    def call_settings_changed(self, app: AppContext,
                               key: str, old_val, new_val):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_settings_changed",
                                app, key, old_val, new_val)

    def call_preset_save(self, app: AppContext, name: str):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_preset_save", app, name)

    def call_preset_load(self, app: AppContext, name: str):
        for p in self.plugins:
            if p.enabled:
                self._safe_call(p, "on_preset_load", app, name)

    # ── Preset integration ──

    def collect_plugin_settings(self) -> Dict[str, Any]:
        """Собирает настройки всех плагинов для сохранения в пресет."""
        result = {}
        for p in self.plugins:
            try:
                data = p.get_settings()
                if data:
                    for k, v in data.items():
                        result[f"plugin.{p.name}.{k}"] = v
            except Exception as e:
                log.error("Plugin '%s'.get_settings() error: %s",
                          p.name, e)
        return result

    def apply_plugin_settings(self, all_data: Dict[str, Any]):
        """Раздаёт настройки плагинам при загрузке пресета."""
        # Группируем по имени плагина
        per_plugin: Dict[str, Dict[str, Any]] = {}
        prefix = "plugin."
        for k, v in all_data.items():
            if k.startswith(prefix):
                rest = k[len(prefix):]
                dot = rest.find(".")
                if dot > 0:
                    pname = rest[:dot]
                    pkey = rest[dot + 1:]
                    per_plugin.setdefault(pname, {})[pkey] = v

        for p in self.plugins:
            if p.name in per_plugin:
                try:
                    p.set_settings(per_plugin[p.name])
                except Exception as e:
                    log.error("Plugin '%s'.set_settings() error: %s",
                              p.name, e)

    # ── UI ──

    def get_plugin_list(self) -> List[Dict[str, Any]]:
        """Возвращает список плагинов для отображения в UI."""
        return [{
            "name": p.name,
            "version": p.version,
            "author": p.author,
            "description": p.description,
            "enabled": p.enabled,
        } for p in self.plugins]

    def set_enabled(self, name: str, enabled: bool):
        """Включает/выключает плагин по имени."""
        if name in self._plugin_map:
            self._plugin_map[name].enabled = enabled

    def get_load_errors(self) -> List[str]:
        return list(self._load_errors)

    @property
    def count(self) -> int:
        return len(self.plugins)

    @property
    def active_count(self) -> int:
        return sum(1 for p in self.plugins if p.enabled)
