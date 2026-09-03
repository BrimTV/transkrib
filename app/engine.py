"""Движок расшифровки: выбор железа, скачивание модели, потоковая выдача сегментов.

Три пути, по убыванию предпочтения:
  mlx  — Apple Silicon, GPU через mlx-whisper (перенесено из VK-Vibe unpack/transcribe.py)
  cuda — Windows/Linux с NVIDIA, faster-whisper на CUDA (float16)
  cpu  — везде, faster-whisper int8 (тот же режим, что CPU-фолбэк в VK-Vibe)

Любой сбой GPU-пути → молча переходим на cpu, пользователь видит только строку статуса.
Все события уходят в emit(dict) — окно рисует из них прогресс и живой текст.
"""
import contextlib
import glob
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
import wave
from logging.handlers import RotatingFileHandler

APP_NAME = "Transkrib"


class UserError(Exception):
    """Исключение, чей текст показывается пользователю дословно, без обёртки
    «не удалось обработать файл» и без traceback в интерфейсе (traceback всё равно
    уходит в log())."""


from . import media  # noqa: E402 — после UserError: media.py импортирует его обратно


_logger = None
_logger_lock = threading.Lock()


def _get_logger():
    """logging.Logger с RotatingFileHandler (В3): без ротации transkrib.log рос
    без ограничений — на машине разработчика раздулся вместе с забытыми
    temp-папками, а обычный пользователь узнал бы об этом, только когда
    кончится место на диске. 2 МБ × 2 копии — этого с запасом хватает на историю
    последних сессий, но файл не растёт бесконечно."""
    global _logger
    if _logger is not None:
        return _logger
    with _logger_lock:
        if _logger is not None:
            return _logger
        logger = logging.getLogger("transkrib")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            handler = RotatingFileHandler(
                os.path.join(data_dir(), "transkrib.log"),
                maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
            logger.addHandler(handler)
        except Exception:
            pass
        _logger = logger
        # Строка-разделитель раз за процесс, при первом реальном обращении к логу:
        # без версии/платформы/железа по логу от пользователя не понять, какая у
        # него сборка и какой бэкенд она выбрала.
        try:
            from . import __version__
            logger.info(f"=== Transkrib {__version__} platform={sys.platform} "
                        f"machine={platform.machine()} {hardware_info()} ===")
        except Exception:
            pass
        return logger


def log(msg):
    """Строка в <data_dir>/transkrib.log — единственное место, где видно, почему
    GPU-путь упал: тост в окне живёт 4 секунды."""
    try:
        _get_logger().info(str(msg))
    except Exception:
        pass

# ── модели ───────────────────────────────────────────────────────────────────
# fw: имя репозитория CTranslate2 (faster-whisper). mlx: репозиторий mlx-community.
MODELS = {
    "small":          dict(label="Лёгкая — быстро даже на слабом ноутбуке",
                           fw="Systran/faster-whisper-small",
                           mlx="mlx-community/whisper-small-mlx", size_mb=480),
    "medium":         dict(label="Средняя — заметно точнее, в 2–3 раза медленнее",
                           fw="Systran/faster-whisper-medium",
                           mlx="mlx-community/whisper-medium-mlx", size_mb=1500),
    "large-v3-turbo": dict(label="Большая turbo — для видеокарты, догружается отдельно",
                           fw="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                           mlx="mlx-community/whisper-large-v3-turbo", size_mb=1600),
    "tiny":           dict(label="Крошечная — мгновенно, качество слабое",
                           fw="Systran/faster-whisper-tiny",
                           mlx="mlx-community/whisper-tiny-mlx", size_mb=75),
}
DEFAULT_MODEL = "auto"


# ── встроенная модель (кладётся в сборку build/fetch_model.py) ───────────────
def bundled_root():
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "bundled_models")


def bundled_model_key():
    """Ключ модели, вшитой в эту сборку, или None (запуск из исходников без bundled_models)."""
    try:
        import json
        with open(os.path.join(bundled_root(), "variant.json"), encoding="utf-8") as f:
            return json.load(f).get("model")
    except Exception:
        return None


def bundled_model_path(model_key, backend):
    kind = "mlx" if backend == "mlx" else "fw"
    p = os.path.join(bundled_root(), model_key, kind)
    return p if os.path.isfile(os.path.join(p, "config.json")) else None


def resolve_model(model_key, backend):
    """'auto' → встроенная в сборку модель; из исходников — small."""
    if model_key != "auto":
        return model_key
    return bundled_model_key() or "small"


def auto_label():
    return f"Встроенная — {resolve_model('auto', 'cpu')}, без скачивания"

LANGUAGES = [("ru", "Русский"), ("auto", "Определить автоматически"), ("en", "English"),
             ("uk", "Українська"), ("kk", "Қазақша"), ("de", "Deutsch"), ("fr", "Français"),
             ("es", "Español"), ("it", "Italiano"), ("pt", "Português"), ("tr", "Türkçe"),
             ("zh", "中文"), ("ja", "日本語")]


def data_dir():
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    d = os.path.join(base, APP_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def models_dir():
    d = os.path.join(data_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def temp_dir():
    """Свой временный каталог, а не системный (В3): на Windows системный temp
    бывает на маленьком диске и/или под кириллическим путём (см. A1), а тут —
    рядом с моделями, там же, где приложение и так пишет."""
    d = os.path.join(data_dir(), "tmp")
    os.makedirs(d, exist_ok=True)
    return d


def cleanup_temp(max_age_sec=3600):
    """Удалить папки transkrib_* старше max_age_sec — и в своём temp_dir(), и в
    системном tempfile.gettempdir() (там их оставляли версии до В3). Порог в
    час не занижать: он же защищает wav второго одновременно запущенного
    экземпляра приложения от удаления прямо во время его работы. Ошибки
    удаления (файл занят, нет прав) молча пропускаем — это уборка, а не
    критичная операция."""
    now = time.time()
    for base in (temp_dir(), tempfile.gettempdir()):
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for name in names:
            if not name.startswith("transkrib_"):
                continue
            p = os.path.join(base, name)
            try:
                if not os.path.isdir(p) or now - os.path.getmtime(p) < max_age_sec:
                    continue
                shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass


# ── безопасные пути для C++-библиотек на Windows (A1) ──────────────────────────
def _short_path_name(p):
    """GetShortPathNameW отдельной функцией: тестам проще подменить её саму, чем
    возиться с ctypes.windll на машине без Windows."""
    import ctypes
    buf = ctypes.create_unicode_buffer(260)
    n = ctypes.windll.kernel32.GetShortPathNameW(p, buf, 260)
    return buf.value if n else p


def _junction_or_copy(p):
    """Junction в C:\\Users\\Public\\Transkrib\\m-<sha1(p)[:8]> — stdlib, без прав
    администратора, ноль байт, работает между томами → копия, если и junction не
    удался (например, файловая система без их поддержки)."""
    import hashlib
    digest = hashlib.sha1(p.encode("utf-8")).hexdigest()[:8]
    link = os.path.join(r"C:\Users\Public\Transkrib", f"m-{digest}")
    if os.path.exists(link):
        return link
    try:
        os.makedirs(os.path.dirname(link), exist_ok=True)
        import _winapi
        _winapi.CreateJunction(p, link)
        return link
    except Exception:
        pass
    try:
        if os.path.isdir(p):
            shutil.copytree(p, link)
            return link
        os.makedirs(link, exist_ok=True)
        shutil.copy2(p, os.path.join(link, os.path.basename(p)))
        return os.path.join(link, os.path.basename(p))
    except Exception:
        return p  # ничего не вышло — отдаём как есть, дальше упадёт с понятной ошибкой открытия файла


def ascii_safe_path(p, acp=None):
    """ctranslate2 и sherpa-onnx открывают файлы через std::ifstream по узкой
    строке; на Windows это ANSI-кодовая страница процесса, и путь с кириллицей
    (типичная папка обычного пользователя) не откроется, если она не UTF-8.
    Три попытки по возрастанию инвазивности: короткое имя 8.3, junction, копия.
    acp — кодовая страница параметром ради теста без реальной Windows: обычно
    None, тогда берём настоящую GetACP()."""
    if sys.platform != "win32":
        return p
    if acp is None:
        import ctypes
        acp = ctypes.windll.kernel32.GetACP()
    if acp == 65001 or p.isascii():
        return p
    short = _short_path_name(p)
    if short.isascii():
        return short
    return _junction_or_copy(p)


# ── железо ───────────────────────────────────────────────────────────────────
_cuda_prepared = False


def _prepare_cuda_dlls():
    """Windows: cublas/cudnn лежат в pip-пакетах nvidia-*, а не в системе.
    ctranslate2 найдёт их только если папки добавлены в поиск DLL ДО импорта."""
    global _cuda_prepared
    if _cuda_prepared or sys.platform != "win32":
        return
    _cuda_prepared = True
    roots = [getattr(sys, "_MEIPASS", None), os.path.dirname(os.path.dirname(__file__))]
    try:
        import site
        roots += site.getsitepackages()
    except Exception:
        pass
    for root in filter(None, roots):
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _has_mlx():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper  # noqa: F401
        return True
    except Exception:
        return False


_cuda_diag_result = None
_cuda_diag_lock = threading.Lock()


def _nvidia_smi():
    """Вывод nvidia-smi — имя карты, версия драйвера, память, вычислительная
    способность (A4). Ищем по трём местам, как советует ТЗ: сам PATH (обычно
    пусто — nvidia-smi туда не добавляют), затем System32 (там он есть у
    большинства установок драйвера) и каталог NVIDIA Corporation\\NVSMI
    (старые/нестандартные установки). Отсутствие утилиты — не ошибка: часть
    систем её не ставит, а видеокарта при этом работает."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        candidates = [
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                         "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
        ]
        exe = next((c for c in candidates if os.path.isfile(c)), None)
    if not exe:
        return None
    try:
        import subprocess
        proc = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, **media._NO_WINDOW)
        return proc.stdout.strip() or proc.stderr.strip() or None
    except Exception as e:
        return f"ошибка запуска: {e}"


def _cuda_diag():
    """Диагностика видеокарты NVIDIA для лога и карточки «почему процессор» (A4).
    Живого NVIDIA-железа у разработчиков нет — CUDA-путь проверят по логу первые
    пользователи (решение владельца), поэтому эта функция обязана рассказать не
    «не сработало», а именно что́ не так: собрана ли CUDA-упаковка сборки вообще
    (DLL грузятся и без видеокарты — так отличаем «сломана упаковка» от «нет
    драйвера»), драйвер старый или его нет, что видит nvidia-smi. На машине без
    видеокарты и не на Windows обязана отработать быстро и без исключений —
    её зовут из hardware_info(), а тот печатается в лог при каждом старте."""
    info = dict(ctranslate2_version=None, cuda_device_count=0, compute_types=[],
                dll_dirs=[], dll_load=None, driver_version=None, driver_msg=None,
                nvidia_smi=None)
    try:
        import ctranslate2
        info["ctranslate2_version"] = ctranslate2.__version__
        n = ctranslate2.get_cuda_device_count()
        info["cuda_device_count"] = n
        types = []
        for i in range(n):
            try:
                types.append(sorted(ctranslate2.get_supported_compute_types("cuda", i)))
            except Exception as e:
                types.append(f"ошибка: {e}")
        info["compute_types"] = types
    except Exception as e:
        info["ctranslate2_version"] = f"ошибка: {e}"

    if sys.platform != "win32":
        return info

    # Каталоги, которые _prepare_cuda_dlls добавляет в поиск DLL, и попытка их
    # реально загрузить: если упаковка сборки не довезла cublas/cudnn (или собран
    # неверный вариант), DLL не загрузятся даже на машине совсем без видеокарты —
    # именно так отличаем «поломана сборка» от «нет NVIDIA».
    _prepare_cuda_dlls()
    roots = [getattr(sys, "_MEIPASS", None), os.path.dirname(os.path.dirname(__file__))]
    try:
        import site
        roots += site.getsitepackages()
    except Exception:
        pass
    dll_dirs = []
    for root in filter(None, roots):
        dll_dirs += glob.glob(os.path.join(root, "nvidia", "*", "bin"))
    info["dll_dirs"] = dll_dirs

    import ctypes
    dll_load = {}
    for name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
        try:
            ctypes.WinDLL(name)
            dll_load[name] = "ok"
        except OSError as e:
            dll_load[name] = str(e)
    info["dll_load"] = dll_load

    # Версия драйвера через nvcuda.dll — она есть и грузится независимо от того,
    # довезли ли мы cublas/cudnn: nvcuda ставит сам драйвер NVIDIA, не мы.
    try:
        nvcuda = ctypes.WinDLL("nvcuda.dll")
        version = ctypes.c_int()
        rc = nvcuda.cuDriverGetVersion(ctypes.byref(version))
        if rc == 0:
            info["driver_version"] = version.value
            if version.value < 12000:
                info["driver_msg"] = ("драйвер старее CUDA 12, нужен 527.41 или новее, "
                                       "видеокарта не используется")
        else:
            info["driver_msg"] = f"cuDriverGetVersion вернул код {rc}"
    except OSError:
        info["driver_msg"] = "драйвера NVIDIA нет"
    except Exception as e:
        info["driver_msg"] = f"ошибка проверки драйвера: {e}"

    info["nvidia_smi"] = _nvidia_smi()

    # Библиотеки для видеокарты едут отдельным дополнением (вместе с ними сборка
    # не влезает в предел размера файла у GitHub). Владелец видеокарты, который
    # его не поставил, иначе просто увидит «процессор» и не поймёт почему —
    # поэтому отличаем «дополнения нет» от «драйвер старый» и от «карты нет».
    if not dll_dirs and info.get("driver_version"):
        info["addon_msg"] = ("У вас есть видеокарта NVIDIA, но дополнение для неё не "
                             "установлено. Скачайте Transkrib-medium-windows-nvidia.zip "
                             "и распакуйте в папку с программой")
    return info


def _has_cuda():
    _prepare_cuda_dlls()
    global _cuda_diag_result
    try:
        import ctranslate2
        ok = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        ok = False
    # Диагностика — один раз за процесс (DLL/nvidia-smi недёшевы, а видеокарта
    # за время работы не появляется и не исчезает): дальше используем кэш, но
    # результат кладём в лог сразу же, чтобы объяснить, почему выбран cpu.
    if _cuda_diag_result is None:
        with _cuda_diag_lock:
            if _cuda_diag_result is None:
                try:
                    _cuda_diag_result = _cuda_diag()
                except Exception as e:
                    _cuda_diag_result = {"error": str(e)}
                log(f"cuda diag: {_cuda_diag_result}")
    return ok


def detect_backend(prefer_gpu=True):
    """'mlx' | 'cuda' | 'cpu'."""
    if prefer_gpu:
        if _has_mlx():
            return "mlx"
        if _has_cuda():
            return "cuda"
    return "cpu"


def backend_label(b):
    return {"mlx": "GPU Apple Silicon (MLX)", "cuda": "GPU NVIDIA (CUDA)",
            "cpu": "Процессор"}.get(b, b)


def backend_phrase(b):
    """Для фраз вида «Распознаю на …»."""
    return {"mlx": "на GPU Apple Silicon (MLX)", "cuda": "на видеокарте NVIDIA (CUDA)",
            "cpu": "на процессоре"}.get(b, b)


def _physical_cpu():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def cpu_workers():
    """Число потоков под расшифровку/диаризацию. Физические ядра, а не
    логические: Hyper-Threading/SMT почти не ускоряет числодробительный CPU-
    инференс, зато psutil.cpu_count() по умолчанию считает и его. На 4+ ядрах
    оставляем одно свободным — иначе ноутбук греется и интерфейс подтормаживает
    во время расшифровки (Б4)."""
    n = _physical_cpu() or os.cpu_count() or 2
    return n - 1 if n >= 4 else n


def hardware_info():
    # cuda=_has_cuda() до cuda_diag=_cuda_diag_result: _has_cuda() и наполняет
    # кэш диагностики (см. её докстринг) — порядок ключей ниже важен, kwargs
    # вычисляются слева направо.
    return dict(platform=sys.platform, machine=platform.machine(), cpu_count=os.cpu_count(),
                physical_cpu=_physical_cpu(), mlx=_has_mlx(), cuda=_has_cuda(),
                cuda_diag=_cuda_diag_result, backend=detect_backend(),
                bundled_model=bundled_model_key())


# ── скачивание моделей ───────────────────────────────────────────────────────
def _repo_for(model_key, backend):
    return MODELS[model_key]["mlx" if backend == "mlx" else "fw"]


_FW_PATTERNS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json",
                "vocabulary.*", "generation_config.json"]


def model_is_cached(model_key, backend):
    if bundled_model_path(model_key, backend):
        return True
    from huggingface_hub import try_to_load_from_cache  # noqa: E402
    repo = _repo_for(model_key, backend)
    probe = "config.json"
    r = try_to_load_from_cache(repo, probe, cache_dir=models_dir())
    return isinstance(r, str) and os.path.exists(r)


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _looks_like_no_internet(e):
    """Похоже ли исключение при скачивании на «нет сети», а не на что-то другое."""
    # huggingface_hub 1.x ходит в сеть через httpx, а не requests: requests у нас
    # вообще не в зависимостях, и на чистой машине его нет (в CI на этом падали тесты).
    try:
        import httpx
        if isinstance(e, httpx.TransportError):   # ConnectError, timeouts и прочая сеть
            return True
    except Exception:
        pass
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
        if isinstance(e, (LocalEntryNotFoundError, OfflineModeIsEnabled)):
            return True
    except Exception:
        pass
    text = str(e)
    return any(needle in text for needle in ("getaddrinfo", "Max retries", "timed out"))


def _download_error(e, model_key, size_gb):
    if _looks_like_no_internet(e):
        return UserError(
            f"Нет доступа к интернету — модель «{model_key}» ({size_gb:.1f} ГБ) скачать не "
            f"удалось. Проверьте подключение или выберите встроенную модель в настройках")
    return RuntimeError(f"не удалось скачать модель {model_key}: {e}")


def ensure_model(model_key, backend, emit, cancel=None):
    """Скачать модель, если её нет. Прогресс — опросом размера папки кэша:
    у huggingface_hub нет нормального колбэка на байты."""
    from huggingface_hub import snapshot_download, HfApi
    bundled = bundled_model_path(model_key, backend)
    if bundled:
        return bundled
    repo = _repo_for(model_key, backend)
    patterns = None if backend == "mlx" else _FW_PATTERNS
    if model_is_cached(model_key, backend):
        return snapshot_download(repo, cache_dir=models_dir(), allow_patterns=patterns,
                                 local_files_only=True)

    size_mb = MODELS[model_key]["size_mb"]
    need = size_mb * 1024 * 1024 * 1.3
    free = shutil.disk_usage(models_dir()).free
    if free < need:
        raise UserError(
            f"На диске нет места для модели «{model_key}»: нужно ~{need / 1e6:.0f} МБ, "
            f"свободно {free / 1e6:.0f} МБ")

    emit(dict(type="stage", stage="download", msg=f"Скачиваю модель «{model_key}» (один раз)"))
    total = None
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        import fnmatch
        total = sum((s.size or 0) for s in info.siblings
                    if patterns is None or any(fnmatch.fnmatch(s.rfilename, p) for p in patterns))
    except Exception:
        total = MODELS[model_key]["size_mb"] * 1024 * 1024

    cache_sub = os.path.join(models_dir(), "models--" + repo.replace("/", "--"))
    result, error = {}, {}

    def worker():
        try:
            result["path"] = snapshot_download(repo, cache_dir=models_dir(), allow_patterns=patterns)
        except Exception as e:
            error["e"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        if cancel and cancel.is_set():
            raise InterruptedError("отменено")
        done = _dir_size(cache_sub) if os.path.isdir(cache_sub) else 0
        emit(dict(type="download", done=done, total=total))
        t.join(0.5)
    if "e" in error:
        raise _download_error(error["e"], model_key, size_mb / 1024)
    emit(dict(type="download", done=total, total=total))
    return result["path"]


# ── загрузка модели (кэш в памяти) ───────────────────────────────────────────
_loaded = {}          # (backend, model_key) -> объект модели или путь (для mlx)
_loaded_lock = threading.Lock()


def load_model(model_key, backend, emit, cancel=None, compute_type=None):
    """compute_type — только для backend="cuda" (ступенчатый откат A4, см.
    transcribe_file): "auto" по умолчанию — ctranslate2 сам выбирает лучший
    поддерживаемый тип и понижает его на старых картах вместо исключения;
    "int8_float32" — второй, более экономный по VRAM шаг отката. Раньше тут
    был жёстко зашит "float16": на GTX 10xx/MX/старых Quadro он не
    поддерживается и WhisperModel бросает исключение, из-за которого мы сразу
    и без объяснений уходили на процессор. При этом medium в float16 занимает
    на видеокарте ≈1.5 ГБ против ≈0.8 ГБ в int8_float16 — а типичная цель,
    вроде GTX 1650 в ноутбуках, часто имеет всего 2–4 ГБ VRAM."""
    ct = compute_type if backend == "cuda" else None
    key = (backend, model_key, ct)
    with _loaded_lock:
        if key in _loaded:
            return _loaded[key]
    path = ensure_model(model_key, backend, emit, cancel)
    path = ascii_safe_path(path)  # кириллица в пути на Windows без UTF-8 — см. ascii_safe_path
    emit(dict(type="stage", stage="load", msg=f"Загружаю модель {model_key} в память"))
    t0 = time.time()
    if backend == "mlx":
        obj = path  # mlx_whisper грузит сам при первом вызове и кэширует внутри
    else:
        from faster_whisper import WhisperModel
        if backend == "cuda":
            obj = WhisperModel(path, device="cuda", compute_type=ct or "auto")
            try:
                actual = obj.model.compute_type
            except Exception:
                actual = "?"
            log(f"cuda compute_type: запрошен {ct or 'auto'}, фактический {actual}")
        else:
            workers = cpu_workers()
            log(f"cpu_threads={workers} (физических {_physical_cpu()})")
            obj = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=workers)
    with _loaded_lock:
        _loaded.clear()   # держим в памяти одну модель: они по 1.5–3 ГБ
        _loaded[key] = obj
    emit(dict(type="stage", stage="load", msg=f"Модель готова за {time.time() - t0:.1f} с"))
    return obj


def unload_models():
    """Выгрузить всё из памяти: faster-whisper (удаляем объект), MLX (его внутренний
    ModelHolder держит модель в классе — иначе 0.5–1.5 ГБ висят до выхода) и кэш Metal."""
    import gc
    with _loaded_lock:
        _loaded.clear()
    try:
        # mlx_whisper.transcribe как атрибут — функция; сам модуль берём из sys.modules
        _t = sys.modules.get("mlx_whisper.transcribe")
        if _t is None:
            raise ImportError
        _t.ModelHolder.model = None
        _t.ModelHolder.model_path = None
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


def models_loaded():
    with _loaded_lock:
        return bool(_loaded)


# ── бережём память ───────────────────────────────────────────────────────────
IDLE_UNLOAD_SEC = 300      # модель без работы дольше этого — выгружаем
LOW_MEM_GB = 1.5           # свободной памяти меньше — выгружаем сразу после задачи
_last_used = 0.0


def memory_available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return None


def maybe_unload(busy=False):
    """Зовётся из фонового таймера окна. Возвращает причину выгрузки или None."""
    if busy or not models_loaded():
        return None
    idle = time.time() - _last_used
    avail = memory_available_gb()
    if idle > IDLE_UNLOAD_SEC:
        unload_models()
        return f"модель выгружена: простой {int(idle // 60)} мин"
    if avail is not None and avail < LOW_MEM_GB:
        unload_models()
        return f"модель выгружена: свободно {avail:.1f} ГБ"
    return None


def _friendly_error(e):
    """Технические исключения, у которых есть понятная человеку причина, — остальное
    вызывающий заворачивает в «не удалось обработать файл (техническая деталь: …)»."""
    text = str(e)
    if isinstance(e, MemoryError) or "Failed to allocate" in text or "out of memory" in text.lower():
        return "Не хватает оперативной памяти. Закройте другие программы или выберите модель поменьше"
    return None


# ── резка на куски по речи (Б2/Б3) ────────────────────────────────────────────
def _pieces(total_sec, regions, target=120.0, max_len=150.0):
    """Границы кусков [a,b), покрывающие всю запись целиком, на промежутках
    между репликами (media.region_gaps по интервалам VAD), а не там, где вообще
    тихо по ffmpeg silencedetect — в тихой музыке пауз в речи может не быть
    совсем. target/max_len разные для двух вызовов: у MLX умолчания дают куски
    ~2 мин, как было раньше (Б2); faster-whisper зовут с 750/900 — куски
    10-15 мин, чтобы не decode'ить и не считать лог-мел на файл целиком (Б3)."""
    gaps = media.region_gaps(regions, min_gap=0.6)
    bounds = [0.0] + media.cut_points(total_sec, gaps, target=target, max_len=max_len) + [total_sec]
    return list(zip(bounds, bounds[1:]))


# ── расшифровка ──────────────────────────────────────────────────────────────
def _transcribe_fw(model, wav, language, backend, emit, cancel, total_sec, regions):
    """Кусками по 10-15 минут (Б3), а не файлом целиком: faster-whisper иначе
    декодирует весь файл в float32 (3 ч ≈ 690 МБ) и считает лог-мел на всю
    запись (ещё ≈550 МБ) — пик под 2 ГБ поверх памяти модели, на ноутбуке с
    8 ГБ почти всё. Заодно отмена и прогресс работают между кусками, а не
    только после того, как whisper досчитает файл целиком. При auto языке
    определяем его на первом куске и дальше передаём явно — иначе whisper
    заново гадает на каждом куске, а конец файла в другом языке путает его."""
    pieces = _pieces(total_sec, regions or [(0.0, total_sec)], target=750.0, max_len=900.0)
    lang = None if language == "auto" else language
    for a, b in pieces:
        if cancel and cancel.is_set():
            raise InterruptedError("отменено")
        chunk = media.wav_slice(wav, a, b)
        segments, info = model.transcribe(
            chunk, language=lang, beam_size=5 if backend == "cuda" else 1,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
        )
        if lang is None:
            lang = info.language
            emit(dict(type="stage", stage="lang", msg=f"Язык: {info.language} ({info.language_probability:.0%})"))
        for s in segments:
            if cancel and cancel.is_set():
                raise InterruptedError("отменено")
            text = s.text.strip()
            if text:
                # Показатели качества сегмента отдают оба движка; фильтр галлюцинаций
                # без них слепнет на самых надёжных признаках (тишина и зацикливание).
                emit(dict(type="segment", start=a + s.start, end=a + s.end, text=text,
                          no_speech_prob=getattr(s, "no_speech_prob", None),
                          avg_logprob=getattr(s, "avg_logprob", None),
                          compression_ratio=getattr(s, "compression_ratio", None)))
        emit(dict(type="progress", done_sec=b, total_sec=total_sec))


def _transcribe_mlx(model_path, wav, language, emit, cancel, total_sec, regions):
    """MLX не стримит сегменты и не умеет vad_filter сам (в отличие от
    faster-whisper) — поэтому границы кусков и участки внутри них берём из
    regions, интервалов речи, которые заранее нашёл Silero VAD
    (media.speech_regions, Б2). Раньше куски по 120-150 с уходили в модель
    целиком, вместе с музыкой и тишиной, и там MLX галлюцинировал чаще, чем
    faster-whisper.

    Звук отдаём массивом, а не путём: по пути mlx_whisper зовёт `ffmpeg` из PATH,
    а у приложения, запущенного из Finder, PATH системный и ffmpeg там нет
    (2026-09-02, «[Errno 2] No such file or directory: 'ffmpeg'» → CPU-фолбэк)."""
    import mlx_whisper
    if regions is not None and not regions:
        # VAD отработал, но речи нет вообще — не гоняем модель по музыке/тишине зря.
        emit(dict(type="stage", stage="vad", note=True, msg="Речи в записи не найдено"))
        emit(dict(type="progress", done_sec=total_sec, total_sec=total_sec))
        return
    if regions is not None:
        pieces = _pieces(total_sec, regions)
        speech = regions
    else:
        # VAD упал (см. лог) — старый запасной путь: резать по паузам ffmpeg,
        # кусок отдаём целиком, как было раньше.
        silences = media.silence_midpoints(wav, cancel=cancel) if total_sec > 150 else []
        bounds = [0.0] + media.cut_points(total_sec, silences) + [total_sec]
        pieces = list(zip(bounds, bounds[1:]))
        speech = [(0.0, total_sec)]

    lang_announced = language != "auto"
    for a, b in pieces:
        if cancel and cancel.is_set():
            raise InterruptedError("отменено")
        overlap = [(max(s, a), min(e, b)) for s, e in speech if e > a and s < b]
        if not overlap:
            emit(dict(type="progress", done_sec=b, total_sec=total_sec))
            continue
        piece_len = b - a
        speech_sec = sum(e - s for s, e in overlap)
        clip = "0"  # тишины в куске < 20% — отдаём целиком, как раньше
        if piece_len > 0 and (1 - speech_sec / piece_len) >= 0.2:
            clip = []
            for s, e in overlap:
                clip += [round(s - a, 3), round(e - a, 3)]
        res = mlx_whisper.transcribe(
            media.wav_slice(wav, a, b), path_or_hf_repo=model_path,
            language=None if language == "auto" else language,
            condition_on_previous_text=False, fp16=True,
            no_speech_threshold=0.6, logprob_threshold=-1.0,
            clip_timestamps=clip,
        )
        try:
            import mlx.core as mx
            mx.clear_cache()   # иначе буферный кэш растёт на каждую новую длину куска
        except Exception:
            pass
        if not lang_announced:
            emit(dict(type="stage", stage="lang", msg=f"Язык: {res.get('language')}"))
            lang_announced = True
        for s in res.get("segments", []):
            text = (s.get("text") or "").strip()
            if text:
                emit(dict(type="segment", start=a + s["start"], end=a + s["end"], text=text,
                          no_speech_prob=s.get("no_speech_prob"),
                          avg_logprob=s.get("avg_logprob"),
                          compression_ratio=s.get("compression_ratio")))
        emit(dict(type="progress", done_sec=b, total_sec=total_sec))


@contextlib.contextmanager
def _prevent_sleep():
    """Не даём компьютеру уснуть на время расшифровки (Б6): часовая запись на
    процессоре легко переживает таймер простоя ноутбука — тогда на macOS
    App Nap придушит процесс и он резко замедлится, а на Windows система может
    уйти в сон и задача зависнет до пробуждения. pyobjc уже в зависимостях
    через cocoa-бэкенд pywebview, поэтому на macOS ничего доставать не нужно.
    Снятие запрета — всегда в finally: даже если расшифровка упала с
    исключением, сон должен вернуться в обычный режим, а не остаться
    запрещённым до конца процесса."""
    mac_activity = None
    is_win = sys.platform == "win32"
    try:
        if sys.platform == "darwin":
            try:
                from Foundation import (NSProcessInfo, NSActivityIdleSystemSleepDisabled,
                                        NSActivityUserInitiated)
                pi = NSProcessInfo.processInfo()
                opts = NSActivityUserInitiated | NSActivityIdleSystemSleepDisabled
                token = pi.beginActivityWithOptions_reason_(opts, "Transkrib: расшифровка")
                mac_activity = (pi, token)
            except Exception as e:
                log(f"не удалось запретить сон (macOS): {e}")
        elif is_win:
            # Из рабочего потока: transcribe_file и так зовётся из фонового Thread
            # (см. app/main.py, Api.start), поэтому вызов уже идёт не из потока окна.
            try:
                import ctypes
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            except Exception as e:
                log(f"не удалось запретить сон (Windows): {e}")
        yield
    finally:
        if mac_activity is not None:
            try:
                pi, token = mac_activity
                pi.endActivity_(token)
            except Exception as e:
                log(f"не удалось снять запрет на сон (macOS): {e}")
        if is_win:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS
            except Exception as e:
                log(f"не удалось снять запрет на сон (Windows): {e}")


def transcribe_file(src, model_key, language, emit, cancel=None, prefer_gpu=True,
                    diarize=False, num_speakers=0):
    """Полный цикл: извлечь звук → модель → сегменты. Всё через emit.
    diarize=True — параллельно считаем говорящих и в конце шлём событие speakers.
    Тело обёрнуто _prevent_sleep() (Б6): на долгой задаче компьютер не должен
    уснуть сам, иначе она зависнет или (на macOS) резко замедлится под App Nap."""
    cancel = cancel or threading.Event()
    t_start = time.time()
    tmpdir = tempfile.mkdtemp(prefix="transkrib_", dir=temp_dir())
    wav = os.path.join(tmpdir, "audio.wav")
    log(f"start {src} model={model_key} lang={language} gpu={prefer_gpu} diarize={diarize}")

    _emit = emit

    # Единственная точка, через которую проходят сегменты обоих движков, — здесь и
    # стоит фильтр галлюцинаций (см. app/cleanup.py): на музыке и тишине Whisper
    # выдаёт повторы и фразы из обучающих субтитров, и это доходило до человека.
    from .cleanup import SegmentFilter
    seg_filter = SegmentFilter(language)

    def emit(e):
        if e["type"] == "segment":
            why = seg_filter.verdict(e)
            if why:
                log(f"фильтр {why}: [{e['start']:.1f}-{e['end']:.1f}] {e['text'][:80]}")
                return
        if e["type"] == "done" and seg_filter.dropped:
            e["dropped"] = seg_filter.dropped
            e["msg"] = f"{e['msg']} (убрано повторов: {seg_filter.dropped})"
        if e["type"] in ("stage", "done", "error", "cancelled", "speakers"):
            log(e.get("msg") or f"{e['type']} count={e.get('count')}")
        _emit(e)
    with _prevent_sleep():
        try:
            # Владельцу видеокарты, не поставившему дополнение, надо сказать об этом
            # словами, а не оставлять его гадать, почему считает процессор.
            hint = (hardware_info().get("cuda_diag") or {}).get("addon_msg")
            if hint and prefer_gpu:
                emit(dict(type="stage", stage="hardware", note=True, msg=hint))
            media.check_readable(src)  # быстрая проверка до старта: файл есть, не пуст, не облачный плейсхолдер
            emit(dict(type="stage", stage="extract", msg="Извлекаю звуковую дорожку"))
            total_sec, audio_tracks = media.extract_wav(
                src, wav, on_progress=lambda p: emit(dict(type="extract", progress=p)), cancel=cancel)
            if audio_tracks > 1:
                emit(dict(type="stage", stage="extract", note=True,
                          msg=f"Дорожек: {audio_tracks}, взята первая"))
            emit(dict(type="stage", stage="extract", msg=f"Звук готов: {_fmt_dur(total_sec)}"))
            if cancel.is_set():
                raise InterruptedError("отменено")

            # Поиск речи (Б2/Б3) — один раз на весь файл, до выбора бэкенда: границы
            # кусков одинаковы что для MLX, что для faster-whisper, и если GPU-путь
            # ниже упадёт и мы откатимся на cpu, пересчитывать VAD незачем.
            # Цена — примерно 1-2% длительности записи, отсюда и своя стадия прогресса.
            emit(dict(type="stage", stage="vad", msg="Ищу речь"))
            try:
                regions = media.speech_regions(
                    wav, total_sec, on_progress=lambda p: emit(dict(type="vad_progress", progress=p)),
                    cancel=cancel)
            except InterruptedError:
                raise
            except Exception as e:
                import traceback
                log(f"VAD FAILED, откат на резку по ffmpeg-паузам:\n{traceback.format_exc()}")
                regions = None
            if cancel.is_set():
                raise InterruptedError("отменено")

            backend = detect_backend(prefer_gpu)
            # Ступенчатый откат для видеокарты (A4): "auto" — ctranslate2 сам выбирает
            # лучший поддерживаемый тип; если исключение (в том числе нехватка VRAM
            # прямо на первом сегменте — оно приходит не при загрузке модели, а из
            # _transcribe_fw) — пробуем более экономный int8_float32, и только если
            # не помогло и это, уходим на процессор. Раньше откат был не ступенчатый:
            # любой сбой на видеокарте сразу вёл на cpu.
            compute_type = "auto" if backend == "cuda" else None
            requested = model_key
            while True:
                model_key = resolve_model(requested, backend)
                try:
                    if requested == "auto":
                        emit(dict(type="stage", stage="model", msg=f"Модель: {model_key} (встроенная)"))
                    model = load_model(model_key, backend, emit, cancel, compute_type=compute_type)
                    emit(dict(type="stage", stage="transcribe", backend=backend,
                              msg=f"Распознаю {backend_phrase(backend)}"))
                    if backend == "mlx":
                        _transcribe_mlx(model, wav, language, emit, cancel, total_sec, regions)
                    else:
                        _transcribe_fw(model, wav, language, backend, emit, cancel, total_sec, regions)
                    break
                except InterruptedError:
                    raise
                except Exception as e:
                    import traceback
                    log(f"{backend}/{compute_type} FAILED on {os.path.basename(src)} model={model_key}:\n"
                        f"{traceback.format_exc()}")
                    if backend == "cuda" and compute_type == "auto":
                        emit(dict(type="stage", stage="fallback", note=True,
                                  msg=f"Видеокарта с автовыбором типа не справилась ({str(e)[:120]}), "
                                      f"пробую int8_float32"))
                        unload_models()
                        compute_type = "int8_float32"
                        continue
                    if backend == "cpu":
                        raise
                    if backend == "cuda":
                        # Текст для карточки без техжаргона (см. правила ТЗ): дословно
                        # то, что попросил владелец — драйвер обновить может каждый.
                        msg = ("Видеокарта NVIDIA не сработала, считаю на процессоре. "
                               f"Обновите драйвер NVIDIA (нужен 527 или новее) "
                               f"(техническая деталь: {str(e)[:160]})")
                    else:
                        msg = f"{backend_label(backend)} не сработал: {str(e)[:160]}. Перехожу на процессор"
                    emit(dict(type="stage", stage="fallback", note=True, msg=msg))
                    unload_models()
                    backend = "cpu"
                    compute_type = None
            if diarize and not cancel.is_set():
                # Стерео-звонки (В6): в записях Zoom, Telegram и телефонных приложений
                # собеседники часто физически разложены по разным каналам стерео — тогда
                # угадывать голоса через sherpa не нужно и вредно (медленнее и ошибается
                # там, где канал уже однозначно говорит, кто есть кто). Пробуем этот
                # способ первым, до sherpa; не подошёл (моно, обычная стереозапись с одним
                # микрофоном) — media.stereo_turns сама вернёт None, идём обычным путём.
                # Если человек прямо указал число голосов и оно не два, канальный
                # способ заведомо не то, что просили: в каналах их всегда двое.
                turns = None
                if num_speakers in (0, 2):
                    # Стадию объявляем заранее: на длинной записи разбор каналов
                    # занимает десятки секунд, и без строки на экране это выглядит
                    # как зависание сразу после «текст готов».
                    emit(dict(type="stage", stage="diarize",
                              msg="Текст готов, смотрю, записаны ли собеседники в разные каналы"))
                    turns = media.stereo_turns(
                        src, total_sec, cancel,
                        on_progress=lambda p: emit(dict(type="diar_progress", progress=p)))
                if turns is not None:
                    n = len({t["speaker"] for t in turns})
                    log(f"стерео-звонок: говорящие разложены по каналам ({n}), sherpa не вызываю")
                    emit(dict(type="stage", stage="diarize", note=True,
                              msg="Похоже, собеседники записаны в разные каналы — разделил по ним, "
                                  "это точнее и быстрее"))
                    emit(dict(type="speakers", turns=turns, count=n))
                else:
                    # После текста, а не параллельно: на 8 ГБ два движка разом выбивали MLX
                    # в CPU-фолбэк (Metal не мог выделить память), а на длинных файлах
                    # диаризация ещё и тяжёлая сама по себе.
                    from . import diarize as _diar
                    avail = memory_available_gb()
                    long_file = total_sec > 3600  # дольше часа — выгружаем всегда, не дожидаясь нехватки памяти
                    if long_file or (avail is not None and avail < 2.5):
                        unload_models()
                        log("unload before diarization: " +
                            ("запись длиннее часа" if long_file else f"{avail:.1f} GB free"))
                    # 0.05× длительности — сложившаяся на практике оценка при шаге окна 0.5 (см. diarize.py)
                    eta = _fmt_dur(total_sec * 0.05)
                    emit(dict(type="stage", stage="diarize", msg=f"Текст готов, определяю говорящих (≈{eta})"))
                    try:
                        turns = _diar.run_in_worker(wav, emit, num_speakers, cancel)
                        n = len({t["speaker"] for t in turns})
                        emit(dict(type="speakers", turns=turns, count=n))
                    except InterruptedError:
                        raise
                    except Exception as e:
                        import traceback
                        log("diarization FAILED:\n" + traceback.format_exc())
                        emit(dict(type="stage", stage="diarize", note=True,
                                  msg=f"Разделить по говорящим не удалось: {str(e)[:120]}"))
            elapsed = time.time() - t_start
            emit(dict(type="done", backend=backend, elapsed=elapsed, total_sec=total_sec,
                      msg=f"Готово за {_fmt_dur(elapsed)} (скорость {total_sec / max(elapsed, 0.01):.1f}x)"))
            global _last_used
            _last_used = time.time()
            avail = memory_available_gb()
            if avail is not None and avail < LOW_MEM_GB:
                unload_models()
                emit(dict(type="stage", stage="unload", msg=f"Памяти мало ({avail:.1f} ГБ свободно), модель выгружена"))
        except InterruptedError:
            emit(dict(type="cancelled", msg="Остановлено"))
        except UserError as e:
            emit(dict(type="error", msg=str(e)))
        except Exception as e:
            import traceback
            log(f"transcribe_file FAILED on {os.path.basename(src)}:\n{traceback.format_exc()}")
            emit(dict(type="error", msg=_friendly_error(e) or f"Не удалось обработать файл (техническая деталь: {e})"))
        finally:
            for f in (wav,):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass


def _fmt_dur(sec):
    sec = int(round(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ── самопроверка (для CI и диагностики) ──────────────────────────────────────
def selftest(model_key=None):
    model_key = model_key or bundled_model_key() or "tiny"
    # Из Finder/проводника PATH системный: прячем внешний ffmpeg, чтобы поймать
    # любую скрытую зависимость от него (mlx_whisper ходил за ним в PATH).
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    os.environ["PATH"] = os.pathsep.join(
        d for d in os.environ.get("PATH", "").split(os.pathsep)
        if d and not os.path.exists(os.path.join(d, exe)))
    print("PATH без ffmpeg:", os.environ["PATH"][:120])
    """Синтетический WAV → полный цикл. Проверяет ffmpeg, модель, железо, поток событий."""
    for stream in (sys.stdout, sys.stderr):  # консоль Windows бывает cp1252 — кириллица её роняет
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("hardware:", hardware_info())
    tmp = tempfile.mkdtemp(prefix="transkrib_selftest_", dir=temp_dir())
    try:
        src = os.path.join(tmp, "tone.wav")
        with wave.open(src, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
            frames = bytearray()
            for i in range(22050 * 3):
                v = int(8000 * math.sin(2 * math.pi * 440 * i / 22050))
                frames += int(v).to_bytes(2, "little", signed=True)
            w.writeframes(bytes(frames))
        events = []

        def emit(e):
            events.append(e)
            if e["type"] in ("stage", "done", "error", "segment", "cancelled"):
                print(e)
        transcribe_file(src, model_key, "ru", emit)
        ok = any(e["type"] == "done" for e in events)
        # Диаризация: если модели в сборке — проверяем, что библиотека и модели грузятся.
        try:
            from . import diarize
            if diarize.available() and diarize.model_paths():
                wav16 = os.path.join(tmp, "tone16.wav")
                media.extract_wav(src, wav16)
                # run_in_worker, а не run(): проверяет, что перезапуск собственного
                # бинарника с флагом --diarize-worker реально работает (A3) — то же,
                # что использует transcribe_file в обычной работе.
                turns = diarize.run_in_worker(wav16, lambda e: None)
                print(f"diarization: ok (через воркер), реплик на синтетическом тоне: {len(turns)}")
            else:
                print("diarization: модели не вшиты, пропускаю")
        except Exception as e:
            print("diarization: FAILED", e)
            ok = False
    finally:
        # Самопроверка не должна оставлять за собой мусор (В3) — иначе после
        # каждого запуска --selftest в CI и на машине разработчика копится
        # ровно то, из-за чего temp и завели: заброшенные transkrib_* папки.
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if len(sys.argv) > 1:
        transcribe_file(sys.argv[1], DEFAULT_MODEL, "ru", lambda e: print(e),
                        diarize="--speakers" in sys.argv)
