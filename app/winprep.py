"""Проверка окружения Windows до создания окна pywebview: WebView2 и .NET Framework.

Без WebView2 pywebview молча откатывается на MSHTML (Internet Explorer) — окно
открывается пустым, без объяснений. Без .NET Framework 4.7.2 не стартует pythonnet
(мост в winforms-бэкенд pywebview). Модуль сознательно не импортирует ничего
тяжёлого (numpy, ctranslate2, webview...) и вызывается из main() до `import webview`.

На не-Windows все функции возвращают безопасные заглушки — модуль должен молча
импортироваться и на macOS (там эта проверка не имеет смысла: там cocoa, не WebView2).
"""
import os
import sys

IS_WIN = sys.platform == "win32"

# CLSID совпадает с тем, что проверяет сам pywebview (webview/platforms/winforms.py,
# _is_chromium) — те же три ключа реестра, тот же порядок проверки.
_WEBVIEW2_CLSID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
_WEBVIEW2_DOWNLOAD_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
_DOTNET_472 = 461808  # Release для .NET Framework 4.7.2 (минимум для pythonnet)
_LONG_PATH_LIMIT = 170  # A7: за этой длиной longPathAware не спасает без LongPathsEnabled=1


def webview2_version():
    """Версия установленного WebView2 Runtime (Evergreen) или None, если не найден.

    Проверяет реестр в трёх местах — как pywebview: HKLM\\...\\WOW6432Node\\...,
    тот же путь без WOW6432Node, и HKCU. Установленным считаем непустое значение
    `pv`, не равное "0.0.0.0" (так помечен ключ Evergreen до первой установки).
    Переменная TRANSKRIB_FAKE_NO_WEBVIEW2=1 заставляет функцию считать, что
    WebView2 не найден — без неё отрицательную ветку не проверить ни в CI, ни на
    маке разработчиков."""
    if os.environ.get("TRANSKRIB_FAKE_NO_WEBVIEW2") == "1":
        return None
    if not IS_WIN:
        return None

    import winreg

    candidates = [
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLSID}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLSID}"),
        (winreg.HKEY_CURRENT_USER, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLSID}"),
    ]
    for hive, path in candidates:
        try:
            with winreg.OpenKey(hive, path) as key:
                value, _ = winreg.QueryValueEx(key, "pv")
        except OSError:
            continue
        value = str(value).strip()
        if value and value != "0.0.0.0":
            return value
    return None


def dotnet_release():
    """Release-номер установленного .NET Framework (ветка v4\\Full) или None, если
    ключ не найден. 461808 — версия 4.7.2."""
    if not IS_WIN:
        return None

    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full"
        ) as key:
            value, _ = winreg.QueryValueEx(key, "Release")
        return int(value)
    except (OSError, ValueError, TypeError):
        return None


def require_ui_runtime(interactive=True):
    """Проверка перед созданием окна. Возвращает True, если WebView2 на месте.

    interactive=True (обычный запуск): при отсутствии WebView2 показывает MessageBoxW
    с предложением открыть страницу загрузки Evergreen-рантайма и завершает процесс
    кодом 2 — продолжать без WebView2 бессмысленно, окно всё равно будет пустым.
    interactive=False (нужен CI и --check-env): окно не показывает и не завершает
    процесс, просто возвращает результат проверки."""
    if not IS_WIN:
        return True  # проверка не имеет смысла вне Windows

    if webview2_version() is not None:
        return True
    if not interactive:
        return False

    import ctypes

    MB_ICONERROR = 0x10
    MB_YESNO = 0x4
    MB_SETFOREGROUND = 0x10000
    IDYES = 6
    text = "Для работы нужен компонент Microsoft WebView2. Открыть страницу загрузки?"
    try:
        choice = ctypes.windll.user32.MessageBoxW(0, text, "Transkrib", MB_ICONERROR | MB_YESNO | MB_SETFOREGROUND)
    except Exception:
        choice = 0
    if choice == IDYES:
        try:
            os.startfile(_WEBVIEW2_DOWNLOAD_URL)
        except OSError:
            pass
    sys.exit(2)


def check_install_path_length():
    """A7: длинные пути. `longPathAware=true` в манифесте exe работает только
    когда в реестре включена системная настройка `LongPathsEnabled=1` — у
    большинства пользователей она выключена, и распакованная в глубокую папку
    сборка (например, кириллический путь пользователя плюс вложенные подпапки
    архива) откроет файлы модели с ошибкой, которую человеку не объяснить.
    Возвращает текст для человека или None, если всё в порядке; сама решение
    «что делать» (MessageBox, exit) не принимает — это дело main(), как и с
    require_ui_runtime()."""
    if not IS_WIN:
        return None
    base = getattr(sys, "_MEIPASS", None)
    if not base or len(base) <= _LONG_PATH_LIMIT:
        return None
    return "Программа лежит слишком глубоко. Перенесите папку ближе к корню диска, например в C:\\Transkrib"


# ── единственный экземпляр (A7) ─────────────────────────────────────────────
# Дескриптор держим модульной переменной, а не локальной: блокировка снимается,
# как только файл закрывается (сборщиком мусора в том числе), поэтому ссылку
# нужно держать живой до конца процесса — ОС сама снимет lock при его смерти.
_instance_lock_fh = None


def _bring_existing_window_to_front(title):
    """Windows-only: второй запуск не открывает окно заново, а поднимает уже
    работающее — по тому же заголовку, что `create_window` ставит в
    app/main.py (`f"Transkrib {__version__}"`). Тихо ничего не делает, если
    окно не нашлось (например, оно ещё не успело создаться) — второй процесс
    в этом случае просто завершится, а не покажет ложную ошибку."""
    import ctypes
    SW_RESTORE = 9
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def acquire_instance_lock(window_title=None):
    """Блокировка на файле `<data_dir>/instance.lock`: не даёт открыть окно
    дважды разом — иначе две копии грузят по 1.5 ГБ модели одновременно и
    гоняются за одним settings.json. Дескриптор держим открытым до конца
    процесса и не закрываем сами — ОС снимает блокировку в момент смерти
    процесса, в том числе аварийной, сама.

    Брать эту блокировку нужно только в оконном режиме (вызывающий решает
    когда): служебные режимы — самопроверка, --cli, и особенно
    --diarize-worker (см. app/diarize.py, A3: воркер — тот же exe,
    перезапущенный параллельно с работающим окном) — не должны в неё
    упираться, иначе воркер диаризации не смог бы стартовать вовсе.

    Возвращает True, если блокировка захвачена этим процессом (можно
    открывать окно), False — уже есть другой экземпляр (на Windows при False
    дополнительно пытаемся вывести его окно на передний план по window_title)."""
    from . import engine
    global _instance_lock_fh
    path = os.path.join(engine.data_dir(), "instance.lock")
    try:
        fh = open(path, "a+")
    except OSError:
        return True  # не смогли открыть файл лока — не блокируем запуск из-за диагностики

    if IS_WIN:
        import msvcrt
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            fh.close()
            if window_title:
                _bring_existing_window_to_front(window_title)
            return False
    else:
        import fcntl
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False

    _instance_lock_fh = fh
    return True
