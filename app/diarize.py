"""Разделение по говорящим (диаризация) на sherpa-onnx: pyannote-segmentation-3.0
режет речь на реплики, 3D-Speaker CAM++ считает отпечаток голоса, кластеризация
собирает реплики в говорящих. Только onnxruntime, только процессор, без torch.

Проверено 2026-09-02 на записи с четырьмя голосами: все четыре найдены, ~0.12x
от длительности на 4 потоках. Синтетические TTS-голоса не различает — это
свойство моделей, обученных на живой речи.
"""
import json
import os
import subprocess
import sys
import tarfile
import time
import threading
import traceback
import urllib.request
import wave

import numpy as np

from . import engine

SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
SEG_FILE, EMB_FILE = "segmentation.onnx", "embedding.onnx"
# Подобрано 2026-09-02 на двух записях (4 голоса по 57 с; эфир на двоих, 10 мин):
THRESHOLD = 0.65     # 0.6 дробил эфир на 15 «голосов», 0.9 склеивал двоих из четырёх
WINDOW_SHIFT = 0.5   # шаг окна сегментации; 0.1 (дефолт) в 6 раз медленнее без выигрыша
MINOR_SEC, MINOR_FRAC = 3.0, 0.03   # «говорящий» короче этого — обрывок, клеим к соседу
_lock = threading.Lock()


def _dirs():
    return [os.path.join(engine.bundled_root(), "diarization"),
            os.path.join(engine.data_dir(), "diarization")]


def model_paths():
    for d in _dirs():
        seg, emb = os.path.join(d, SEG_FILE), os.path.join(d, EMB_FILE)
        if os.path.isfile(seg) and os.path.isfile(emb):
            return seg, emb
    return None


def available():
    try:
        import sherpa_onnx  # noqa: F401
        return True
    except Exception:
        return False


def _download(url, dst, emit, label):
    tmp = dst + ".part"

    def hook(n, bs, total):
        if total > 0:
            emit(dict(type="download", done=min(n * bs, total), total=total))
    emit(dict(type="stage", stage="download", msg=f"Скачиваю {label} (один раз)"))
    urllib.request.urlretrieve(url, tmp, hook)
    os.replace(tmp, dst)


def fetch_models(target_dir, emit=lambda e: None):
    """Скачать обе модели в target_dir (используется и сборкой, и первым запуском)."""
    os.makedirs(target_dir, exist_ok=True)
    seg, emb = os.path.join(target_dir, SEG_FILE), os.path.join(target_dir, EMB_FILE)
    if not os.path.isfile(seg):
        tarball = os.path.join(target_dir, "seg.tar.bz2")
        _download(SEG_URL, tarball, emit, "модель разделения речи (6 МБ)")
        with tarfile.open(tarball, "r:bz2") as t:
            member = next(m for m in t.getmembers() if m.name.endswith("/model.onnx"))
            member.name = SEG_FILE
            t.extract(member, target_dir)
        os.remove(tarball)
    if not os.path.isfile(emb):
        _download(EMB_URL, emb, emit, "модель отпечатка голоса (27 МБ)")
    return seg, emb


def ensure_models(emit):
    with _lock:
        found = model_paths()
        if found:
            return found
        return fetch_models(_dirs()[1], emit)


def _read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "ожидался 16 кГц моно"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def _diarize_raw(wav_path, emit, num_speakers=0, cancel=None):
    """Сырые реплики sherpa-onnx, без склейки обрывков (см. _finalize) — общий код
    для run() (расчёт в этом же процессе) и worker_main() (расчёт в дочернем
    процессе, см. run_in_worker). ascii_safe_path — здесь, а не в run(), чтобы
    воркер тоже получал безопасные пути к моделям на Windows без UTF-8 (A1)."""
    import sherpa_onnx as so
    seg, emb = ensure_models(emit)
    seg, emb = engine.ascii_safe_path(seg), engine.ascii_safe_path(emb)
    threads = max(1, min(4, (os.cpu_count() or 2) // 2))
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg, window_shift_ratio=WINDOW_SHIFT),
            num_threads=threads),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=threads),
        clustering=so.FastClusteringConfig(num_clusters=num_speakers if num_speakers > 0 else -1,
                                           threshold=THRESHOLD),
        min_duration_on=0.5, min_duration_off=0.8)
    sd = so.OfflineSpeakerDiarization(cfg)
    samples = _read_wav(wav_path)

    t0 = time.time()

    def progress(done, total):
        # Возврат не проверяется сегментацией и кластеризацией (они колбэка не
        # имеют вовсе, см. заголовок файла) — это только для фазы эмбеддингов,
        # надёжной отмены не даёт. Настоящая отмена — run_in_worker, убийством
        # процесса.
        if cancel is not None and cancel.is_set():
            return -1
        if total:
            frac = done / total
            eta = (time.time() - t0) / frac * (1 - frac) if frac > 0.02 else None
            emit(dict(type="diar_progress", progress=frac, eta=eta))
        return 0

    result = sd.process(samples, callback=progress)
    return [dict(start=float(s.start), end=float(s.end), speaker=int(s.speaker))
            for s in result.sort_by_start_time()]


def run(wav_path, emit, num_speakers=0, cancel=None):
    """→ список реплик [{start, end, speaker}], говорящие пронумерованы с 1 в порядке
    появления. Считает в этом же процессе — годится для selftest без реальной отмены.
    Из engine.transcribe_file зовите run_in_worker: колбэк sherpa-onnx не прерывает
    сегментацию и кластеризацию, поэтому надёжная отмена — только убийством процесса."""
    raw = _diarize_raw(wav_path, emit, num_speakers, cancel)
    return _finalize(raw, forced=num_speakers > 0)


# ── воркер: убиваемый процесс вместо колбэка, который не прерывает расчёт ─────
def worker_main(argv):
    """Тело режима --diarize-worker (см. app/main.py::main). argv — всё после
    самого флага: `<wav> <out.json> --speakers N --parent-pid P`.

    Не пишет в transkrib.log (два процесса и ротация лога на Windows дают
    PermissionError) и не импортирует webview/mlx — воркеру они не нужны.
    Кодировка stdout уже переключена на UTF-8 в app.main.main() до диспетчеризации
    режимов — здесь второй раз не делаем."""
    wav_path, out_path = argv[0], argv[1]
    opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
    num_speakers = int(opt("--speakers", "0"))
    parent_pid = int(opt("--parent-pid", "0"))

    def watch_parent():
        # Родитель мог быть убит диспетчером задач или упасть без предупреждения —
        # раз в 2 с проверяем, жив ли он, и выходим сами, чтобы не остаться
        # сиротой, молотящей CPU все сегментацию и кластеризацию.
        import psutil
        while True:
            time.sleep(2.0)
            if not psutil.pid_exists(parent_pid):
                os._exit(1)
    threading.Thread(target=watch_parent, daemon=True).start()

    def emit(e):
        print(json.dumps(e, ensure_ascii=False), flush=True)

    try:
        raw = _diarize_raw(wav_path, emit, num_speakers, cancel=None)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        os.replace(tmp, out_path)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


def run_in_worker(wav_path, emit, num_speakers=0, cancel=None):
    """Как run(), но расчёт идёт в отдельном процессе: колбэк прогресса sherpa-onnx
    вызывается только в фазе эмбеддингов и его возврат не проверяется, а фазы
    сегментации и кластеризации колбэка не имеют вовсе — патч колбэка отмену не
    даёт в принципе. subprocess с перезапуском своего же бинарника, а не
    multiprocessing: очередь multiprocessing тянет за собой SemLock и лишний
    процесс resource_tracker, а proc.kill() убивает мгновенно и надёжно."""
    cancel = cancel or threading.Event()
    ensure_models(emit)  # качаем в родителе — воркер не должен трогать сеть
    out_path = wav_path + ".diarize.json"
    try:
        os.remove(out_path)
    except OSError:
        pass

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--diarize-worker", wav_path, out_path,
               "--speakers", str(int(num_speakers)), "--parent-pid", str(os.getpid())]
        cwd = None
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [sys.executable, "-m", "app.main", "--diarize-worker", wav_path, out_path,
               "--speakers", str(int(num_speakers)), "--parent-pid", str(os.getpid())]
        cwd = repo_root

    from . import media  # тот же флаг прячет консоль ffmpeg — media._NO_WINDOW
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            **media._NO_WINDOW)

    if sys.platform == "win32":
        try:
            import psutil
            psutil.Process(proc.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass  # не критично: приоритет — только чтобы окно приложения не подтормаживало

    stderr_tail = []

    def read_stderr():
        for line in proc.stderr:
            stderr_tail.append(line)
            if len(stderr_tail) > 40:
                stderr_tail.pop(0)
    threading.Thread(target=read_stderr, daemon=True).start()

    def read_stdout():
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                emit(json.loads(line))
            except ValueError:
                pass
    t_out = threading.Thread(target=read_stdout, daemon=True)
    t_out.start()

    try:
        # Ключевой момент: чтение stdout выше блокирующее (readline), а в фазах
        # сегментации и кластеризации воркер молчит десятками секунд — проверить
        # cancel внутри read_stdout негде. Поэтому отдельный цикл-наблюдатель.
        killed = False
        while proc.poll() is None:
            if cancel.wait(0.2):
                killed = True
                try:
                    proc.kill()
                except OSError:
                    pass
                break

        # На Windows временный wav нельзя удалить, пока ОС не закроет хэндлы
        # убитого процесса — transcribe_file чистит его в finally сразу после
        # возврата отсюда, поэтому дожидаемся здесь, а не полагаемся на poll().
        proc.wait(timeout=10)
        t_out.join(timeout=2.0)

        if killed:
            raise InterruptedError("отменено")
        if proc.returncode != 0:
            raise RuntimeError("воркер диаризации упал: " + "".join(stderr_tail[-40:]).strip())

        with open(out_path, encoding="utf-8") as f:
            raw = json.load(f)
        return _finalize(raw, forced=num_speakers > 0)
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def _finalize(raw, forced=False):
    """Обрывки-«говорящие» → ближайший по времени крупный голос; нумерация с 1
    в порядке появления. При явно заданном числе голосов ничего не клеим."""
    if not raw:
        return []
    total = {}
    for t in raw:
        total[t["speaker"]] = total.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    speech = sum(total.values())
    major = {k for k, v in total.items() if v >= max(MINOR_SEC, MINOR_FRAC * speech)}
    if forced or not major:
        major = set(total)
    if len(major) < len(total):
        big = [t for t in raw if t["speaker"] in major]
        for t in raw:
            if t["speaker"] not in major:
                mid = (t["start"] + t["end"]) / 2
                near = min(big, key=lambda b: min(abs(b["start"] - mid), abs(b["end"] - mid)))
                t["speaker"] = near["speaker"]
    order, turns = {}, []
    for t in raw:
        sid = order.setdefault(t["speaker"], len(order) + 1)
        turns.append(dict(start=round(t["start"], 2), end=round(t["end"], 2), speaker=sid))
    return turns


def assign(segments, turns):
    """Каждой фразе — говорящий с наибольшим перекрытием; без перекрытия — ближайший."""
    if not turns:
        return segments
    last = 1
    for seg in segments:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        if best is None:
            mid = (seg["start"] + seg["end"]) / 2
            t = min(turns, key=lambda t: min(abs(t["start"] - mid), abs(t["end"] - mid)))
            best = t["speaker"] if min(abs(t["start"] - mid), abs(t["end"] - mid)) < 3 else last
        seg["speaker"] = best
        last = best
    return segments
