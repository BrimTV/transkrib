"""Разделение по говорящим (диаризация) на sherpa-onnx: pyannote-segmentation-3.0
режет речь на реплики, 3D-Speaker CAM++ считает отпечаток голоса, кластеризация
собирает реплики в говорящих. Только onnxruntime, только процессор, без torch.

Проверено 2026-09-02 на записи с четырьмя голосами: все четыре найдены, ~0.12x
от длительности на 4 потоках. Синтетические TTS-голоса не различает — это
свойство моделей, обученных на живой речи.
"""
import os
import tarfile
import threading
import urllib.request
import wave

import numpy as np

from . import engine

SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
SEG_FILE, EMB_FILE = "segmentation.onnx", "embedding.onnx"
THRESHOLD = 0.6      # порог кластеризации: 0.5–0.7 давали верные 4 голоса, 0.9 склеивал
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


def run(wav_path, emit, num_speakers=0, cancel=None):
    """→ список реплик [{start, end, speaker}], говорящие пронумерованы с 1 в порядке появления."""
    import sherpa_onnx as so
    seg, emb = ensure_models(emit)
    threads = max(1, min(4, (os.cpu_count() or 2) // 2))
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg), num_threads=threads),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=threads),
        clustering=so.FastClusteringConfig(num_clusters=num_speakers if num_speakers > 0 else -1,
                                           threshold=THRESHOLD),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = so.OfflineSpeakerDiarization(cfg)
    samples = _read_wav(wav_path)

    def progress(done, total):
        if cancel is not None and cancel.is_set():
            return -1
        if total:
            emit(dict(type="diar_progress", progress=done / total))
        return 0

    result = sd.process(samples, callback=progress)
    order, turns = {}, []
    for s in result.sort_by_start_time():
        sid = order.setdefault(s.speaker, len(order) + 1)
        turns.append(dict(start=round(float(s.start), 2), end=round(float(s.end), 2), speaker=sid))
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
