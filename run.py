"""Точка входа для PyInstaller и для `python run.py`."""
import multiprocessing
import os
import sys
import threading

# До любых импортов app: huggingface_hub читает эти переменные при первом импорте.
# В windowed-сборке (console=False) sys.stdout/sys.stderr равны None, а прогресс-бар
# tqdm внутри huggingface_hub при скачивании модели падает с AttributeError
# ('NoneType' object has no attribute 'write'). Прогресс скачивания в интерфейсе и
# так есть (событие download), отдельный вывод в консоль не нужен.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")


class _LogStream:
    """Подменяет sys.stdout/sys.stderr, когда они None (windowed-сборка): без этого
    любой print или чужой tqdm падает с AttributeError вместо тихой записи в лог.
    Импорт engine отложен до первой записи — на момент создания объекта app ещё
    не должен быть импортирован (см. _unblock_mark_of_the_web)."""
    def write(self, s):
        s = s.rstrip("\n")
        if s:
            from app.engine import log
            log(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def _fix_frozen_streams():
    if sys.stdout is None:
        sys.stdout = _LogStream()
    if sys.stderr is None:
        sys.stderr = _LogStream()


def _unblock_mark_of_the_web():
    """Windows, только frozen-сборка. Проводник при распаковке zip ставит на файлы
    поток Zone.Identifier (Mark-of-the-Web); .NET отказывается грузить такие сборки
    (Python.Runtime.dll, WebView2) с ошибкой 0x80131515 — окно просто не открывается.
    Снимаем поток один раз при первом запуске, до импорта app (и pythonnet вместе
    с ним) — читаем только os/sys, чтобы порядок точно ничего не сломал."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return None
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    marker = os.path.join(meipass, "webview", "lib", "Python.Runtime.dll") + ":Zone.Identifier"
    if not os.path.exists(marker):
        return None
    cleaned = failed = 0
    for root, _, files in os.walk(meipass):
        for name in files:
            if os.path.splitext(name)[1].lower() not in (".dll", ".pyd", ".exe"):
                continue
            try:
                os.remove(os.path.join(root, name) + ":Zone.Identifier")
                cleaned += 1
            except OSError:
                failed += 1
    return cleaned, failed


def _log_exception(exc_type, exc_value, exc_tb, thread_name=None):
    import traceback
    from app.engine import log
    where = f" [{thread_name}]" if thread_name else ""
    log(f"необработанное исключение{where}:\n" +
        "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))


def _install_excepthooks():
    """Сейчас исключение фонового потока в windowed-сборке пропадает бесследно —
    ни трейсбека, ни следа в логе."""
    sys.excepthook = _log_exception

    def _thread_hook(args):
        _log_exception(args.exc_type, args.exc_value, args.exc_traceback,
                       args.thread.name if args.thread else None)
    threading.excepthook = _thread_hook


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _motw_result = _unblock_mark_of_the_web()
    _fix_frozen_streams()
    _install_excepthooks()
    from app.main import main
    if _motw_result is not None:
        from app.engine import log
        log(f"Mark-of-the-Web: поток снят с {_motw_result[0]} файлов, ошибок {_motw_result[1]}")
    main()
