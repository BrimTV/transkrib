"""Окно приложения: pywebview + мост Python↔JS. Никаких портов и серверов —
JS зовёт методы Api напрямую, Python толкает события через evaluate_js."""
import errno
import functools
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
import urllib.parse
import webbrowser

from . import __version__, engine, export, media


def _ensure_utf8_console():
    """Консоль Windows часто не UTF-8 (cp1252 и т.п.): печать кириллицы валит
    процесс UnicodeEncodeError на первой же русской строке — так падал
    --check-env в CI на строке `webview2=неизвестно...`. Одно место для всех
    неоконных режимов, зовём в самом начале main() до любой печати.
    sys.stdout/stderr в windowed-сборке бывают None или уже подменены (см.
    run.py) — у таких объектов reconfigure может не быть."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _ui_path():
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "app", "ui", "index.html")


def _ui_url():
    """Адрес страницы интерфейса.

    На macOS окно кеширует страницы file:// по адресу, а адрес после обновления
    программы прежний: человек получал старый интерфейс поверх нового движка, и
    на своей машине этого не видно, пока не почистишь кеш системы. Метка версии
    в адресе делает страницу заведомо свежей.

    На Windows такую метку добавлять нельзя: WebView2 с ней не открывает файл
    вовсе — окно остаётся пустым. Проверено в CI, поймано проверкой --smoke.
    Из исходников версия меняется редко, поэтому там берём время правки файла."""
    path = _ui_path()
    if sys.platform != "darwin":
        return path
    stamp = __version__
    if not getattr(sys, "frozen", False):
        try:
            stamp += "-" + str(int(os.path.getmtime(path)))
        except OSError:
            pass
    return pathlib.Path(path).as_uri() + "?v=" + urllib.parse.quote(stamp)


def _settings_path():
    return os.path.join(engine.data_dir(), "settings.json")


def _webview_storage_path():
    """Профиль WebView2/куки в своей папке данных, а не в роуминговом
    %APPDATA%\\pywebview (А7): private_mode=False без storage_path создаёт
    именно там cache_dir (winforms.init_storage) — десятки МБ, которые в
    доменных окружениях ездят по сети вместе с профилем пользователя. Наш
    интерфейс не использует ни куки, ни localStorage (всё состояние — в
    settings.json), так что менять поведение private_mode не нужно: достаточно
    указать некочующую папку явно, и на Windows, и на macOS (там роуминга нет,
    но лишний профиль пусть тоже живёт рядом с остальными данными приложения)."""
    return os.path.join(engine.data_dir(), "webview")


# ── версия и обновления (В5) ─────────────────────────────────────────────────
# Разрешаем open_url только на страницы своего репозитория — метод торчит в JS,
# и открывать по нему произвольные адреса незачем.
_ALLOWED_URL_PREFIX = "https://github.com/BrimTV/transkrib"


def _build_info():
    """Блок build из bundled_models/variant.json (версия, короткий хеш коммита,
    дата, вариант сборки lite/medium) — кладёт build/fetch_model.py. При запуске
    из исходников (variant.json нет) отдаём безопасную заглушку для шапки UI."""
    try:
        with open(os.path.join(engine.bundled_root(), "variant.json"), encoding="utf-8") as f:
            data = json.load(f)
        build = data.get("build") or {}
        if build:
            return build
    except Exception:
        pass
    return dict(version=__version__, commit="-", date="-", variant="исходники")


DEFAULT_SETTINGS = dict(model=engine.DEFAULT_MODEL, language="ru", prefer_gpu=True,
                        autosave=True, autosave_formats=["txt", "srt"],
                        ts_mode="paragraph", ts_interval=60, ts_style="brackets_short",
                        diarize=False, num_speakers=0, pick_dir="")


def load_settings():
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(s):
    with open(_settings_path(), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


# ── автосохранение без перезаписи и с фолбэком (В1) ─────────────────────────
def _documents_dir():
    """~/Documents/Transkrib (на Windows — %USERPROFILE%\\Documents\\Transkrib):
    первый каталог-фолбэк, когда папка с исходником недоступна для записи."""
    if sys.platform == "win32":
        base = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    else:
        base = os.path.expanduser("~")
    return os.path.join(base, "Documents", "Transkrib")


# Коды ошибок, при которых имеет смысл переключиться на другой каталог, а не
# считать это фатальной ошибкой. ENOSPC сюда нарочно не входит — «нет места»
# не лечится сменой папки, это отдельная понятная ошибка для пользователя.
_WRITE_FALLBACK_ERRNOS = {errno.EACCES, errno.EROFS, errno.EPERM, errno.ENOENT}


def _is_write_permission_error(e):
    """os.access() тут не годится: он врёт на сетевых дисках и в OneDrive (см. ТЗ).
    Надёжнее пробовать писать и ловить настоящую ошибку доступа."""
    if isinstance(e, PermissionError):
        return True
    return isinstance(e, OSError) and e.errno in _WRITE_FALLBACK_ERRNOS


def _autosave_index_path():
    return os.path.join(engine.data_dir(), "autosave_index.json")


def _load_autosave_index():
    try:
        with open(_autosave_index_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_autosave_index(idx):
    try:
        with open(_autosave_index_path(), "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # реестр — вспомогательная штука для распознавания «наших» файлов; потерять не страшно


def _is_ours(idx, path):
    """Файл в реестре и его размер/время изменения совпадают с тем, что мы сами
    записали — значит после нас его никто не трогал и перезаписать можно. Иначе
    (чужой файл или наш, но изменённый человеком) — только под новым именем."""
    rec = idx.get(path)
    if not rec:
        return False
    try:
        st = os.stat(path)
    except OSError:
        return False
    return rec.get("size") == st.st_size and rec.get("mtime") == st.st_mtime


def _autosave_target(dest_dir, base, fmt, idx):
    """<base>.<fmt>, если файла нет или это наш; иначе <base>.transkrib.<fmt>,
    при новой коллизии <base>.transkrib-2.<fmt> и так далее — плееры подхватывают
    video.transkrib.srt так же, как video.srt."""
    direct = os.path.join(dest_dir, f"{base}.{fmt}")
    if not os.path.exists(direct) or _is_ours(idx, direct):
        return direct
    out = os.path.join(dest_dir, f"{base}.transkrib.{fmt}")
    n = 2
    while os.path.exists(out) and not _is_ours(idx, out):
        out = os.path.join(dest_dir, f"{base}.transkrib-{n}.{fmt}")
        n += 1
    return out


def _write_atomic(out, content):
    """Во временный файл рядом, затем переименование — не оставить получателя
    (например, плеер, следящий за файлом) с половиной записи."""
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, out)


def _autosave_to_dir(dest_dir, base, formats, render, idx):
    """Пишет все форматы в dest_dir и регистрирует их в реестре. Бросает исключение
    на первой же неудачной записи — вызывающий код решает, пробовать ли другой каталог."""
    os.makedirs(dest_dir, exist_ok=True)
    written = []
    for fmt in formats:
        out = _autosave_target(dest_dir, base, fmt, idx)
        _write_atomic(out, render(fmt))
        st = os.stat(out)
        idx[out] = dict(size=st.st_size, mtime=st.st_mtime)
        written.append(out)
    return written


class Api:
    def __init__(self):
        # Имя с подчёркиванием обязательно. Перед открытием окна pywebview
        # рекурсивно обходит этот объект, собирая методы для JS, и пропускает
        # только имена с подчёркивания (webview/util.py: get_functions). Пока
        # окно лежало здесь под открытым именем, обход уходил внутрь него: на
        # macOS это лишняя работа, а на Windows — тысячи ошибок COM и упор в
        # предел рекурсии, после которого приложение не закрывалось вовсе.
        self._window = None
        self._cancel = None
        self._busy = False
        self._thread = None
        self._last_push = {}

    # ── события в окно ───────────────────────────────────────────────────
    def _emit(self, event, job_id=None):
        # download/extract/progress летят часто — не чаще 10 раз в секунду на тип
        t = event.get("type")
        if t in ("download", "extract", "progress", "diar_progress"):
            now = time.time()
            if now - self._last_push.get(t, 0) < 0.1:
                return
            self._last_push[t] = now
        if job_id is not None:
            event["job"] = job_id
        payload = json.dumps(event, ensure_ascii=False)
        try:
            self._window.evaluate_js(f"window.onEngineEvent({payload})")
        except Exception:
            pass

    # ── перетаскивание файлов ────────────────────────────────────────────
    def _wire_dnd(self):
        """pywebviewFullPath подставляется в событие drop только если на элемент
        подписан Python-обработчик (webview/util.py: без него drag-n-drop от
        cocoa и edgechromium никогда не узнают настоящий путь к файлу — только
        имя). get_element дёргает evaluate_js, поэтому элементы должны уже быть
        в DOM — подписываемся из window.events.loaded, не раньше."""
        for sel, kind in (("#dropTr", "transcribe"), ("#dropCv", "convert")):
            el = self._window.dom.get_element(sel)
            if el:
                el.events.drop += functools.partial(self._on_drop, kind)

    def _on_drop(self, kind, e):
        files = (e.get("dataTransfer") or {}).get("files") or []
        # Список забираем один раз и целиком: он нужен и для восстановления, и для
        # записи в лог, а второе чтение вернуло бы уже пустой.
        collected = self._window_dnd_paths(consume=True)
        paths, missing = [], []
        for f in files:
            p = f.get("pywebviewFullPath")
            if p:
                paths.append(p)
            else:
                missing.append(f)

        lost = 0
        if missing:
            # Пофайлово, а не «всё или ничего»: библиотека сопоставляет имена по
            # одному, и в одном перетаскивании часть путей может дойти, а часть
            # (кириллица в разложенной форме) — потеряться.
            recovered = self._recover_dropped_paths(missing, collected)
            paths.extend(recovered)
            lost = len(missing) - len(recovered)
            engine.log(f"dnd: {kind} путей нет у {len(missing)} из {len(files)}, "
                       f"восстановлено {len(recovered)}; "
                       f"имена из события={[f.get('name') for f in missing]}; "
                       f"собрано окном={[n for n, _ in collected]}")
        else:
            engine.log(f"dnd: {kind} paths={len(paths)}")

        paths, skipped, truncated = self._expand_dropped(paths)
        self._emit(dict(type="dropped", target=kind, paths=paths, skipped=skipped,
                        lost=lost, truncated=truncated, limit=self._DROP_LIMIT))

    # Столько файлов берём из перетащенных папок за раз. Больше — это уже не
    # «закинул записи», а случайно брошенный Рабочий стол: очередь из тысячи
    # карточек человеку не поможет.
    _DROP_LIMIT = 200
    # Папки, которые macOS показывает одним объектом: внутрь лезть незачем.
    _BUNDLE_DIR_EXT = frozenset({".app", ".bundle", ".framework", ".photoslibrary",
                                ".fcpbundle", ".imovielibrary", ".tvlibrary",
                                ".logicx", ".band", ".sparsebundle", ".rtfd"})

    @classmethod
    def _expand_dropped(cls, paths):
        """Папку разворачиваем в лежащие внутри записи, а не отвечаем ошибкой:
        люди перетаскивают папку целиком, это нормальный способ. Всё, что не
        аудио и не видео, откладываем и сообщаем числом — молча проглатывать
        нельзя, человек будет искать пропавший файл.

        Возвращает (пути, сколько пропущено, упёрлись ли в предел)."""
        out, skipped = [], 0
        for p in paths:
            if os.path.isdir(p):
                for root, dirs, names in os.walk(p):
                    dirs.sort()
                    # Скрытые папки и пакеты (.app, фотобиблиотека, проект
                    # монтажа) внутрь не разворачиваем: записей там нет, зато
                    # бывают десятки тысяч файлов. Точка в имени сама по себе
                    # ничего не значит — «эфир 12.05» это обычная папка.
                    dirs[:] = [d for d in dirs if not d.startswith(".")
                               and os.path.splitext(d)[1].lower() not in cls._BUNDLE_DIR_EXT]
                    for n in sorted(names):
                        full = os.path.join(root, n)
                        if media.is_media(full):
                            out.append(full)
                        if len(out) >= cls._DROP_LIMIT:
                            engine.log(f"dnd: взято первых {cls._DROP_LIMIT} файлов")
                            return out, skipped, True
            elif media.is_media(p):
                out.append(p)
            else:
                skipped += 1
        return out, skipped, False

    @staticmethod
    def _window_dnd_paths(consume=False):
        """Список (имя, путь), который библиотека окна собрала при перетаскивании.
        consume=True забирает его себе и очищает.

        Чистить обязательно: библиотека удаляет из этого списка только то, что
        сама сопоставила с файлом, а всё несопоставленное остаётся там навсегда
        и копится от перетаскивания к перетаскиванию. Если этим не управлять,
        запасное сопоставление ниже однажды подставит человеку файл из прошлого
        раза — молча и с виду правдоподобно.

        Приватная деталь pywebview, поэтому под try: её пропажа не должна ронять
        обработчик — просто не сработает запасной путь."""
        try:
            from webview.dom import _dnd_state
            paths = list(_dnd_state.get("paths") or [])
            if consume:
                del _dnd_state["paths"][:]
            return paths
        except Exception as exc:
            engine.log(f"dnd: список окна недоступен: {exc}")
            return []

    def _recover_dropped_paths(self, files, collected):
        """Запасное сопоставление, когда pywebviewFullPath пуст.

        Библиотека сравнивает имена дословно, а macOS отдаёт кириллицу в
        разложенной форме (и+знак), браузер же — в собранной: строки разные,
        файл «не найден», путь теряется. Сравниваем нормализованно.

        Берём только хвост списка длиной в число перетащенных файлов: свежие
        записи library добавляет в конец, а всё, что осталось от прошлых
        перетаскиваний, лежит перед ними."""
        if not collected or len(collected) < len(files):
            return []
        tail = collected[-len(files):]

        def norm(x):
            return unicodedata.normalize("NFC", os.path.basename(x or "")).casefold()

        by_name = {}
        for name, path in tail:
            by_name.setdefault(norm(urllib.parse.unquote(name)), []).append(path)
        out, matched = [], True
        for f in files:
            bucket = by_name.get(norm(f.get("name")))
            if bucket:
                out.append(bucket.pop(0))
            else:
                matched = False
        if matched and out:
            return out
        # Имена не сошлись совсем: событие и хвост списка — из одного
        # перетаскивания, значит порядок тот же.
        return [path for _, path in tail]

    # ── справочная информация ────────────────────────────────────────────
    def info(self):
        hw = engine.hardware_info()
        models = [dict(key="auto", label=engine.auto_label(), size_mb=0, cached=True,
                       resolved=engine.resolve_model("auto", hw["backend"]))]
        for key, m in engine.MODELS.items():
            models.append(dict(key=key, label=m["label"], size_mb=m["size_mb"], resolved=key,
                               cached=engine.model_is_cached(key, hw["backend"])))
        from . import diarize
        return dict(version=__version__, build=_build_info(), hardware=hw,
                    backend_label=engine.backend_label(hw["backend"]),
                    diarize_available=diarize.available(), diarize_models=bool(diarize.model_paths()),
                    models=models, languages=engine.LANGUAGES, settings=load_settings(),
                    models_dir=engine.models_dir(), convert_formats=list(media.CONVERT_PRESETS))

    def set_settings(self, s):
        cur = load_settings()
        cur.update(s or {})
        save_settings(cur)
        return cur

    # ── диалоги ──────────────────────────────────────────────────────────
    def pick_files(self):
        import webview
        exts = sorted(e.lstrip(".") for e in media.AUDIO_EXT | media.VIDEO_EXT)
        types = ("Аудио и видео (" + ";".join("*." + e for e in exts) + ")", "Все файлы (*.*)")
        try:
            for t in types:
                webview.util.parse_file_type(t)
        except ValueError as e:
            # Если кто-то поправит строку описания и сломает формат (слэш, запятая,
            # точка вне списка расширений), create_file_dialog бросит исключение
            # ДО открытия диалога, и клик по области выбора файла будет молча
            # ничего не делать. Проверяем сами и открываем диалог без фильтра типов.
            engine.log(f"pick_files: некорректное описание типов файлов ({e}), открываю без фильтра")
            types = ()
        s = load_settings()
        # winforms при пустом directory лезет в переменную окружения HOMEPATH,
        # которой может не быть (падение с KeyError) — всегда передаём
        # существующий каталог явно: последний, из которого брали файл.
        last_dir = s.get("pick_dir") or ""
        if not (last_dir and os.path.isdir(last_dir)):
            last_dir = os.path.expanduser("~")
        res = self._window.create_file_dialog(webview.OPEN_DIALOG, directory=last_dir,
                                             allow_multiple=True, file_types=types)
        paths = [p for p in (res or []) if os.path.isfile(p)]
        if paths:
            s["pick_dir"] = os.path.dirname(paths[0])
            save_settings(s)
        return paths

    def open_log(self):
        return self.open_folder(engine.data_dir())

    def open_folder(self, path):
        """Открыть папку в системном файловом менеджере, по возможности выделив
        в ней файл. os.startfile бросал OSError при сломанной ассоциации типов —
        исключение улетало в отклонённое promise JS и терялось (А7). Теперь
        никогда не бросает: всегда словарь ok/error, ошибку JS показывает тостом."""
        try:
            is_file = os.path.isfile(path)
            folder = path if os.path.isdir(path) else (os.path.dirname(path) or ".")
            if not (is_file or os.path.isdir(folder)):
                # open/xdg-open/explorer запускаются через Popen и не бросают
                # исключение сами по себе даже на несуществующий путь (процесс
                # стартует и падает уже сам, асинхронно) — проверяем заранее.
                return dict(ok=False, error=f"Папка не найдена: {folder}")
            if sys.platform == "win32":
                if is_file:
                    # /select открывает Проводник с выделенным файлом; не трогает
                    # ассоциацию типов файла (в отличие от os.startfile на самом
                    # файле), поэтому не падает на сломанной ассоциации.
                    subprocess.Popen(["explorer", f"/select,{path}"])
                else:
                    os.startfile(folder)  # noqa
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path] if is_file else ["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
            return dict(ok=True)
        except Exception as e:
            engine.log(f"open_folder: не удалось открыть {path} ({e})")
            return dict(ok=False, error=f"Не удалось открыть папку: {e}")

    def open_url(self, url):
        """Открывает страницу релизов в системном браузере (В5). Разрешены
        только адреса собственного репозитория — метод доступен из JS, и
        открывать по нему произвольные ссылки не нужно."""
        if not isinstance(url, str) or not url.startswith(_ALLOWED_URL_PREFIX):
            return dict(ok=False, error="адрес не разрешён")
        try:
            webbrowser.open(url)
            return dict(ok=True)
        except Exception as e:
            return dict(ok=False, error=f"Не удалось открыть браузер: {e}")

    # ── расшифровка ──────────────────────────────────────────────────────
    def start(self, job_id, path, model, language, prefer_gpu, diarize=False, num_speakers=0):
        try:
            if self._busy:
                # событие done уходит из engine раньше, чем finally потока ниже
                # сбросит _busy — JS успевает вызвать start следующего файла и
                # получить отказ, хотя предыдущая задача уже фактически кончилась.
                # Ждём недолго вместо мгновенного отказа.
                t = self._thread
                if t is not None and t.is_alive():
                    t.join(3.0)
            if self._busy:
                return dict(ok=False, error="уже идёт задача")
            if not os.path.isfile(path):
                return dict(ok=False, error="файл не найден")
            cancel = threading.Event()
            self._cancel = cancel

            def run():
                try:
                    engine.transcribe_file(path, model, language,
                                           lambda e: self._emit(e, job_id), cancel, prefer_gpu,
                                           diarize=bool(diarize), num_speakers=int(num_speakers or 0))
                except Exception as e:
                    self._emit(dict(type="error", msg=f"{e}\n{traceback.format_exc()[-400:]}"), job_id)
                finally:
                    self._busy = False

            thread = threading.Thread(target=run, daemon=True)
            self._thread = thread
            # _busy выставляем перед стартом, а не после: при мгновенно
            # завершающейся задаче поток может успеть добежать до finally
            # раньше, чем управление вернётся сюда, и тогда запись True
            # поверх его False зависла бы навсегда. Если start() всё же
            # бросит исключение, его поймает except ниже и сбросит busy.
            self._busy = True
            thread.start()
            return dict(ok=True)
        except Exception as e:
            self._busy = False
            return dict(ok=False, error=f"{e}")

    def cancel(self):
        if self._cancel:
            self._cancel.set()
        return True

    @staticmethod
    def _opts():
        s = load_settings()
        return dict(ts_mode=s.get("ts_mode"), ts_interval=s.get("ts_interval"), ts_style=s.get("ts_style"))

    def render(self, segments, fmt, title=None):
        return export.render(segments, fmt, title, self._opts())

    def autosave(self, path, segments, formats):
        """Положить результат рядом с исходником, не затирая чужие файлы (В1).
        Если папка недоступна для записи — переключаемся на Документы/Transkrib,
        а если и туда нельзя — на рабочую папку программы. Возвращает
        dict(written=[...], fallback=None | {dir, reason, label})."""
        base = os.path.splitext(os.path.basename(path))[0]
        idx = _load_autosave_index()
        opts = self._opts()

        def render(fmt):
            return export.render(segments, fmt, os.path.basename(path), opts)

        is_win = sys.platform == "win32"
        attempts = [
            (os.path.dirname(path) or ".", None, None),
            (_documents_dir(), "нет доступа к папке с файлом",
             r"Документы\Transkrib" if is_win else "Документы/Transkrib"),
            (os.path.join(engine.data_dir(), "output"), "нет доступа и к папке Документы",
             "рабочую папку программы"),
        ]
        last_err = None
        for dest_dir, reason, label in attempts:
            try:
                written = _autosave_to_dir(dest_dir, base, formats, render, idx)
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    raise engine.UserError(
                        f"На диске нет места для сохранения результата (папка {dest_dir})") from e
                if not _is_write_permission_error(e):
                    raise
                last_err = e
                engine.log(f"autosave: {dest_dir} недоступна для записи ({e}), пробую другой каталог")
                continue
            _save_autosave_index(idx)
            fallback = None
            if reason:
                fallback = dict(dir=dest_dir, reason=reason, label=label)
                engine.log(f"autosave: сохранил в {dest_dir} вместо папки с файлом ({reason})")
            return dict(written=written, fallback=fallback)
        raise engine.UserError(
            "Не удалось сохранить результат: ни папка с файлом, ни Документы, ни рабочая папка "
            f"программы недоступны для записи ({last_err})")

    def save_as(self, segments, fmt, suggested):
        import webview
        res = self._window.create_file_dialog(webview.SAVE_DIALOG, save_filename=f"{suggested}.{fmt}")
        if not res:
            return None
        out = res[0] if isinstance(res, (list, tuple)) else res
        with open(out, "w", encoding="utf-8") as f:
            f.write(export.render(segments, fmt, suggested, self._opts()))
        return out

    def memory(self):
        return dict(available_gb=engine.memory_available_gb(), loaded=engine.models_loaded())

    # ── фоновый уход за памятью ──────────────────────────────────────────
    def start_housekeeping(self):
        def loop():
            while True:
                time.sleep(30)
                try:
                    reason = engine.maybe_unload(self._busy)
                    if reason:
                        self._emit(dict(type="info", msg=reason.capitalize()))
                except Exception:
                    pass
        threading.Thread(target=loop, daemon=True).start()

    # ── конвертер ────────────────────────────────────────────────────────
    def convert(self, job_id, path, fmt):
        if not os.path.isfile(path):
            return dict(ok=False, error="файл не найден")
        base, ext = os.path.splitext(path)
        out = f"{base}.{fmt}"
        if ext.lower() == f".{fmt}":
            out = f"{base}_converted.{fmt}"
        n = 1
        while os.path.exists(out):
            out = f"{base}_{n}.{fmt}"; n += 1

        def run():
            try:
                self._emit(dict(type="convert_progress", progress=0.0), job_id)
                media.convert(path, out, fmt,
                              on_progress=lambda p: self._emit(dict(type="convert_progress", progress=p), job_id))
                self._emit(dict(type="convert_done", out=out), job_id)
            except Exception as e:
                self._emit(dict(type="convert_error", msg=str(e)), job_id)

        threading.Thread(target=run, daemon=True).start()
        return dict(ok=True, out=out)


def _webview2_version():
    """Версия WebView2 Runtime или None. Настоящая проверка реестра — в app/winprep.py;
    крючок TRANSKRIB_FAKE_NO_WEBVIEW2 нужен CI, чтобы проверить отрицательную ветку
    на раннере, где WebView2 как раз установлен."""
    from . import winprep
    return winprep.webview2_version()


def _dotnet_release():
    """Release-номер .NET Framework (4.7.2 = 461808) или None."""
    from . import winprep
    return winprep.dotnet_release()


def _run_check_env():
    """Печатает факты окружения для диагностики и складывает их же в transkrib.log —
    консольный exe (A6) отдаёт их пользователю без плясок со Start-Process."""
    import ctypes
    from . import diarize

    facts = []

    def rec(k, v):
        facts.append(f"{k}={v}")

    rec("version", __version__)
    rec("platform", sys.platform)
    rec("frozen", bool(getattr(sys, "frozen", False)))
    meipass = getattr(sys, "_MEIPASS", None)
    rec("meipass", meipass or "-")
    rec("meipass_len", len(meipass) if meipass else 0)

    code = 0
    if sys.platform == "win32":
        try:
            acp = ctypes.windll.kernel32.GetACP()
        except Exception as e:
            acp = f"ошибка: {e}"
        rec("ACP", acp)
        wv2 = _webview2_version()
        rec("webview2", wv2 or "не найден")
        rec("dotnet_release", _dotnet_release() or "не найден")
        if wv2 is None:
            code = 3

    exe = media.ffmpeg_exe()
    rec("ffmpeg", exe)
    rec("ffmpeg_exists", os.path.exists(exe))
    rec("bundled_model", engine.bundled_model_key() or "-")
    rec("models_dir", engine.models_dir())
    rec("data_dir", engine.data_dir())
    rec("diarize_available", diarize.available())
    rec("diarize_model_paths", diarize.model_paths() or "-")

    if meipass and len(meipass) > 170:
        # Проводник в Windows не всегда включает длинные пути (LongPathsEnabled=0
        # у большинства); честнее сказать пользователю сразу, а не после падения.
        rec("warning", f"путь сборки длиннее 170 символов ({len(meipass)})")
        code = 4

    for line in facts:
        print(line)
    engine.log("check-env: " + "; ".join(facts))
    return code


def _run_smoke(argv):
    """Для CI: создаёт настоящее окно pywebview, дожидается загрузки страницы,
    зовёт pywebview.api.info() как это делает JS, и закрывается само. Это же
    проверка, что мост Python↔JS вообще жив в собранном приложении."""
    import webview

    hold = 0.0
    if "--hold" in argv:
        try:
            hold = float(argv[argv.index("--hold") + 1])
        except (ValueError, IndexError):
            hold = 0.0

    limit = 90.0 + hold
    log_path = os.path.join(engine.data_dir(), "smoke.log")
    lines = []
    ok = threading.Event()          # info() реально ответил
    loaded_flag = threading.Event()  # событие loaded вообще наступило
    finished = threading.Event()     # штатный выход уже начался

    def write_log():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    api = Api()

    def finish(code):
        # window.destroy() из фонового потока на некоторых бэкендах не будит
        # системный цикл событий (замечено на macOS: process.run() виснет
        # навсегда, хотя окно уже закрыто) — даём штатному выходу короткий
        # срок и подстраховываемся принудительным os._exit, иначе CI зависнет.
        finished.set()
        write_log()
        try:
            window.destroy()
        except Exception:
            pass
        t = threading.Timer(3.0, os._exit, args=(code,))
        t.daemon = True
        t.start()

    def on_loaded():
        loaded_flag.set()
        lines.append("READY")
        write_log()
        got_info = threading.Event()

        def on_info(result):
            lines.append("OK")
            ok.set()
            got_info.set()

        try:
            window.evaluate_js("pywebview.api.info()", callback=on_info)
            # evaluate_js с колбэком не блокирует до resolve промиса — ждём здесь.
            if not got_info.wait(30):
                lines.append("FAIL: pywebview.api.info() не ответил за 30 с")
        except Exception as e:
            lines.append(f"FAIL: {e}")
        try:
            # Мост библиотеки жив и без нашего кода: ошибка в index.html не мешает
            # pywebview.api.info() ответить, и до этой проверки сломанный интерфейс
            # доехал бы до людей. Спрашиваем именно наши функции.
            # Без callback: он вызывается только для промисов, а простое выражение
            # pywebview возвращает значением (на промисе мы бы ждали впустую).
            alive = window.evaluate_js(
                "(typeof onDropped === 'function' && typeof window.onEngineEvent === 'function'"
                " && !!document.querySelector('#dropTr'))")
            lines.append("UI" if alive else f"FAIL: интерфейс не загрузился ({alive!r})")
        except Exception as e:
            lines.append(f"FAIL: проверка интерфейса {e}")
        try:
            # Синтетический drop без файлов: проверяет, что подписка _wire_dnd
            # жива и _on_drop реально вызывается (см. A0) — путей не будет,
            # это нормально, дословный путь проверяется только руками (Г4).
            window.evaluate_js(
                "document.querySelector('#dropTr').dispatchEvent("
                "new DragEvent('drop', {dataTransfer: new DataTransfer(), bubbles: true}))"
            )
            lines.append("DND_DISPATCHED")
        except Exception as e:
            lines.append(f"FAIL: dnd dispatch {e}")
        if hold > 0:
            time.sleep(hold)
        finish(0 if ok.is_set() and "UI" in lines else 2)

    window = webview.create_window(f"Transkrib {__version__}", _ui_url(), js_api=api,
                                   width=1100, height=740, background_color="#111418")
    api._window = window
    window.events.loaded += api._wire_dnd
    window.events.loaded += on_loaded

    def watchdog():
        # Сторож абсолютный, а не «пока не наступил loaded»: зависнуть можно и
        # после загрузки страницы — на evaluate_js, на закрытии окна, на чужом
        # цикле событий. Проверка окна в CI однажды провисела шесть часов, потому
        # что ждали только loaded. Любой исход, кроме штатного выхода, — os._exit.
        if finished.wait(limit):
            return
        lines.append(f"FAIL: сторожевой таймер {limit:.0f} с "
                     f"(loaded={'да' if loaded_flag.is_set() else 'нет'}, "
                     f"мост={'да' if ok.is_set() else 'нет'})")
        write_log()
        os._exit(2)

    threading.Thread(target=watchdog, daemon=True).start()
    webview.start(private_mode=False, storage_path=_webview_storage_path())
    return 0 if ok.is_set() and "UI" in lines else 2


def main():
    _ensure_utf8_console()

    if "--diarize-worker" in sys.argv:
        # Отдельный убиваемый процесс для диаризации (A3): sherpa-onnx не даёт
        # прервать сегментацию и кластеризацию колбэком, поэтому единственная
        # надёжная отмена — kill этого процесса из родителя (diarize.run_in_worker).
        # Ветка до import webview — воркеру окно не нужно и не должно открываться.
        from . import diarize
        args = sys.argv[sys.argv.index("--diarize-worker") + 1:]
        sys.exit(diarize.worker_main(args))

    if "--check-env" in sys.argv:
        sys.exit(_run_check_env())

    if "--smoke" in sys.argv:
        sys.exit(_run_smoke(sys.argv))

    if "--selftest" in sys.argv:
        # Для CI: без окна, результат ещё и в файл. Папка exe бывает только для
        # чтения — пишем в data_dir(), а не в cwd (было раньше).
        log_path = os.path.join(engine.data_dir(), "selftest.log")
        print(f"selftest.log: {log_path}")
        with open(log_path, "w", encoding="utf-8") as log:
            class Tee:
                def write(self, s):
                    log.write(s); log.flush()
                    try:
                        sys.__stdout__.write(s); sys.__stdout__.flush()
                    except Exception:
                        pass
                def flush(self):
                    pass
            sys.stdout = sys.stderr = Tee()
            sys.exit(engine.selftest())

    if "--cli" in sys.argv:
        # Transkrib --cli файл [--speakers] [--model small] [--lang ru] — без окна, события в stdout
        args = sys.argv[sys.argv.index("--cli") + 1:]
        path = next((a for a in args if not a.startswith("--")), None)
        if not path:
            sys.exit("укажите файл")
        opt = lambda k, d: args[args.index(k) + 1] if k in args else d  # noqa: E731
        # Кодировка stdout/stderr уже переключена на UTF-8 в _ensure_utf8_console() выше.
        segs = []

        def show(e):
            if e["type"] == "segment":
                segs.append(e); print(f"[{e['start']:7.1f}] {e['text']}", flush=True)
            elif e["type"] == "speakers":
                from . import diarize
                diarize.assign(segs, e["turns"]); print(f"говорящих: {e['count']}", flush=True)
            elif e["type"] in ("stage", "done", "error", "cancelled"):
                print("::", e["msg"], flush=True)
        engine.transcribe_file(path, opt("--model", "auto"), opt("--lang", "ru"), show,
                               diarize="--speakers" in args)
        print(export.render(segs, "txt", opts=dict(ts_mode="paragraph")))
        sys.exit(0)

    # До создания окна: без WebView2 pywebview молча откатится на движок Internet
    # Explorer и человек увидит пустое окно без единого объяснения. Проверяем и
    # говорим прямо, с предложением скачать. На macOS функция ничего не делает.
    from . import winprep
    winprep.require_ui_runtime()

    # Две копии грузят по полтора гигабайта модели и пишут один файл настроек.
    # Берём блокировку только в оконном режиме: служебные режимы и воркер диаризации
    # запускаются тем же бинарником и не должны на неё натыкаться.
    if not winprep.acquire_instance_lock(window_title=f"Transkrib {__version__}"):
        engine.log("второй экземпляр: окно уже открыто, выходим")
        sys.exit(0)

    # Уборка в фоне: прерванная или упавшая задача оставляет wav на сотни мегабайт,
    # и у человека это копится незаметно, пока не кончится место (у нас за день
    # набежало 1.4 ГБ). Порог в час защищает файлы второго запущенного экземпляра.
    threading.Thread(target=engine.cleanup_temp, daemon=True).start()

    import webview
    api = Api()
    window = webview.create_window(f"Transkrib {__version__}", _ui_url(), js_api=api,
                                   width=1100, height=740, min_size=(820, 560),
                                   background_color="#111418")
    api._window = window
    window.events.loaded += api._wire_dnd
    api.start_housekeeping()
    webview.start(private_mode=False, storage_path=_webview_storage_path(), debug="--debug" in sys.argv)


if __name__ == "__main__":
    main()
