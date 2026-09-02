"""Движок расшифровки: выбор железа, скачивание модели, потоковая выдача сегментов.

Три пути, по убыванию предпочтения:
  mlx  — Apple Silicon, GPU через mlx-whisper (перенесено из VK-Vibe unpack/transcribe.py)
  cuda — Windows/Linux с NVIDIA, faster-whisper на CUDA (float16)
  cpu  — везде, faster-whisper int8 (тот же режим, что CPU-фолбэк в VK-Vibe)

Любой сбой GPU-пути → молча переходим на cpu, пользователь видит только строку статуса.
Все события уходят в emit(dict) — окно рисует из них прогресс и живой текст.
"""
import glob
import math
import os
import platform
import sys
import tempfile
import threading
import time
import wave

from . import media

APP_NAME = "Transkrib"


def log(msg):
    """Строка в <data_dir>/transkrib.log — единственное место, где видно, почему
    GPU-путь упал: тост в окне живёт 4 секунды."""
    try:
        with open(os.path.join(data_dir(), "transkrib.log"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + str(msg) + "\n")
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


def _has_cuda():
    _prepare_cuda_dlls()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


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


def hardware_info():
    return dict(platform=sys.platform, machine=platform.machine(), cpu_count=os.cpu_count(),
                mlx=_has_mlx(), cuda=_has_cuda(), backend=detect_backend(),
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
        raise RuntimeError(f"не удалось скачать модель {repo}: {error['e']}")
    emit(dict(type="download", done=total, total=total))
    return result["path"]


# ── загрузка модели (кэш в памяти) ───────────────────────────────────────────
_loaded = {}          # (backend, model_key) -> объект модели или путь (для mlx)
_loaded_lock = threading.Lock()


def load_model(model_key, backend, emit, cancel=None):
    key = (backend, model_key)
    with _loaded_lock:
        if key in _loaded:
            return _loaded[key]
    path = ensure_model(model_key, backend, emit, cancel)
    emit(dict(type="stage", stage="load", msg=f"Загружаю модель {model_key} в память"))
    t0 = time.time()
    if backend == "mlx":
        obj = path  # mlx_whisper грузит сам при первом вызове и кэширует внутри
    else:
        from faster_whisper import WhisperModel
        if backend == "cuda":
            obj = WhisperModel(path, device="cuda", compute_type="float16")
        else:
            obj = WhisperModel(path, device="cpu", compute_type="int8",
                               cpu_threads=max(1, (os.cpu_count() or 4)))
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


# ── расшифровка ──────────────────────────────────────────────────────────────
def _transcribe_fw(model, wav, language, backend, emit, cancel, total_sec):
    segments, info = model.transcribe(
        wav, language=None if language == "auto" else language,
        beam_size=5 if backend == "cuda" else 1,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
    )
    if language == "auto":
        emit(dict(type="stage", stage="lang", msg=f"Язык: {info.language} ({info.language_probability:.0%})"))
    for s in segments:
        if cancel and cancel.is_set():
            raise InterruptedError("отменено")
        text = s.text.strip()
        if text:
            emit(dict(type="segment", start=s.start, end=s.end, text=text))
        emit(dict(type="progress", done_sec=s.end, total_sec=total_sec))


def _wav_slice(path, start, end):
    """Кусок 16 кГц моно WAV → float32 массив, без ffmpeg."""
    import numpy as np
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        w.setpos(int(start * sr))
        n = int((end - start) * sr)
        data = w.readframes(n)
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_mlx(model_path, wav, language, emit, cancel, total_sec):
    """MLX не стримит сегменты — режем файл по паузам на куски ~2 мин и отдаём
    результат по кускам. Швы на паузах, поэтому фраз не теряем.

    Звук отдаём массивом, а не путём: по пути mlx_whisper зовёт `ffmpeg` из PATH,
    а у приложения, запущенного из Finder, PATH системный и ffmpeg там нет
    (2026-09-02, «[Errno 2] No such file or directory: 'ffmpeg'» → CPU-фолбэк)."""
    import mlx_whisper
    silences = media.silence_midpoints(wav) if total_sec > 150 else []
    bounds = [0.0] + media.cut_points(total_sec, silences) + [total_sec]
    tmpdir = tempfile.mkdtemp(prefix="transkrib_")
    try:
        for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
            if cancel and cancel.is_set():
                raise InterruptedError("отменено")
            piece = wav
            res = mlx_whisper.transcribe(
                _wav_slice(wav, a, b), path_or_hf_repo=model_path,
                language=None if language == "auto" else language,
                condition_on_previous_text=False, fp16=True,
            )
            try:
                import mlx.core as mx
                mx.clear_cache()   # иначе буферный кэш растёт на каждую новую длину куска
            except Exception:
                pass
            if i == 0 and language == "auto":
                emit(dict(type="stage", stage="lang", msg=f"Язык: {res.get('language')}"))
            for s in res.get("segments", []):
                text = (s.get("text") or "").strip()
                if text:
                    emit(dict(type="segment", start=a + s["start"], end=a + s["end"], text=text))
            emit(dict(type="progress", done_sec=b, total_sec=total_sec))
    finally:
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def transcribe_file(src, model_key, language, emit, cancel=None, prefer_gpu=True,
                    diarize=False, num_speakers=0):
    """Полный цикл: извлечь звук → модель → сегменты. Всё через emit.
    diarize=True — параллельно считаем говорящих и в конце шлём событие speakers."""
    cancel = cancel or threading.Event()
    t_start = time.time()
    tmpdir = tempfile.mkdtemp(prefix="transkrib_")
    wav = os.path.join(tmpdir, "audio.wav")
    log(f"start {src} model={model_key} lang={language} gpu={prefer_gpu} diarize={diarize}")

    _emit = emit

    def emit(e):
        if e["type"] in ("stage", "done", "error", "cancelled", "speakers"):
            log(e.get("msg") or f"{e['type']} count={e.get('count')}")
        _emit(e)
    try:
        emit(dict(type="stage", stage="extract", msg="Извлекаю звуковую дорожку"))
        total_sec = media.extract_wav(
            src, wav, on_progress=lambda p: emit(dict(type="extract", progress=p)))
        emit(dict(type="stage", stage="extract", msg=f"Звук готов: {_fmt_dur(total_sec)}"))
        if cancel.is_set():
            raise InterruptedError("отменено")

        backend = detect_backend(prefer_gpu)
        requested = model_key
        while True:
            model_key = resolve_model(requested, backend)
            try:
                if requested == "auto":
                    emit(dict(type="stage", stage="model", msg=f"Модель: {model_key} (встроенная)"))
                model = load_model(model_key, backend, emit, cancel)
                emit(dict(type="stage", stage="transcribe", backend=backend,
                          msg=f"Распознаю {backend_phrase(backend)}"))
                if backend == "mlx":
                    _transcribe_mlx(model, wav, language, emit, cancel, total_sec)
                else:
                    _transcribe_fw(model, wav, language, backend, emit, cancel, total_sec)
                break
            except InterruptedError:
                raise
            except Exception as e:
                import traceback
                log(f"{backend} FAILED on {os.path.basename(src)} model={model_key}:\n{traceback.format_exc()}")
                if backend == "cpu":
                    raise
                emit(dict(type="stage", stage="fallback", note=True,
                          msg=f"{backend_label(backend)} не сработал: {str(e)[:160]}. Перехожу на процессор"))
                unload_models()
                backend = "cpu"
        if diarize and not cancel.is_set():
            # После текста, а не параллельно: на 8 ГБ два движка разом выбивали MLX
            # в CPU-фолбэк (Metal не мог выделить память), а на длинных файлах
            # диаризация ещё и тяжёлая сама по себе.
            from . import diarize as _diar
            avail = memory_available_gb()
            if avail is not None and avail < 2.5:
                unload_models()
                log(f"unload before diarization: {avail:.1f} GB free")
            emit(dict(type="stage", stage="diarize", msg="Текст готов, определяю говорящих"))
            try:
                turns = _diar.run(wav, emit, num_speakers, cancel)
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
    except Exception as e:
        emit(dict(type="error", msg=f"Ошибка: {e}"))
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
    tmp = tempfile.mkdtemp(prefix="transkrib_selftest_")
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
            turns = diarize.run(wav16, lambda e: None)
            print(f"diarization: ok, реплик на синтетическом тоне: {len(turns)}")
        else:
            print("diarization: модели не вшиты, пропускаю")
    except Exception as e:
        print("diarization: FAILED", e)
        ok = False
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if len(sys.argv) > 1:
        transcribe_file(sys.argv[1], DEFAULT_MODEL, "ru", lambda e: print(e),
                        diarize="--speakers" in sys.argv)
