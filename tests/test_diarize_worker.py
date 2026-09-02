"""Тесты app/diarize.py::run_in_worker/worker_main (A3) и app/engine.py::ascii_safe_path (A1).

Отмена и «сирота» помечены slow: обе гоняют настоящий дочерний процесс с
реальными моделями диаризации (уже вшиты в bundled_models/diarization для
этого чекаута) — секунды, а не миллисекунды, и это именно то, что нужно
проверить (перезапуск собственного бинарника, а не патч колбэка).
"""
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import psutil
import pytest

from app import diarize, engine

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(diarize.__file__)))
SPEECH_FIXTURE = Path(__file__).parent / "fixtures" / "speech_ru_10s.wav"


def _write_white_noise_wav(path, seconds, sr=16000, seed=0):
    """16 кГц моно белый шум — синтетические голоса TTS диаризация не различает
    (см. app/diarize.py), а тут говорящие и не нужны: только длительность расчёта,
    достаточная, чтобы отмена/убийство застали процесс за работой. Для отмены
    годится: kill() убивает процесс снаружи независимо от того, чем занята
    sherpa-onnx внутри него."""
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(int(seconds * sr)) * 3000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def _write_looped_speech_wav(path, seconds):
    """Реальная речь (см. tests/fixtures/speech_ru_10s.wav), зациклённая до нужной
    длины. Нужна для теста «сироты»: проверено чтением исходников через профилирование
    (сторонний поток ни разу не получает GIL, пока идёт один блокирующий вызов
    sd.process()) — на чистом шуме sherpa-onnx не находит речи и сегментация всё
    равно молотит по всей длине без единого возврата в Python, а сторожевой поток
    воркера физически не может выполниться, пока не начнётся фаза эмбеддингов (там
    есть колбэк — GIL периодически освобождается). С речью и короткой (2 мин, а не
    20) длиной сегментация успевает закончиться за секунды, и сторож успевает
    сработать раньше, чем расчёт — что и проверяет тест."""
    with wave.open(str(SPEECH_FIXTURE), "rb") as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    clip_sec = len(frames) / (sw * nch) / sr
    reps = int(seconds / clip_sec) + 1
    with wave.open(str(path), "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(frames * reps)


def _dead_pid():
    """PID, который заведомо не существует прямо сейчас (проверено через psutil,
    а не жёстко зашитое число — чтобы не зависеть от лимитов конкретной ОС)."""
    candidate = 999_999
    while psutil.pid_exists(candidate):
        candidate += 1
    return candidate


# ── отмена: «Остановить» должно убивать процесс, а не ждать конца расчёта ─────

@pytest.mark.slow
def test_run_in_worker_cancel_is_fast(tmp_path):
    """Патч колбэка sherpa-onnx не помогает (сегментация и кластеризация колбэка
    не имеют вовсе, см. app/diarize.py) — только kill() отдельного процесса.
    18 минут белого шума — расчёт идёт десятки секунд, cancel взводим на 3-й."""
    wav = tmp_path / "noise.wav"
    _write_white_noise_wav(wav, seconds=18 * 60)

    cancel = threading.Event()
    result = {}

    def worker():
        try:
            diarize.run_in_worker(str(wav), lambda e: None, num_speakers=0, cancel=cancel)
        except BaseException as e:
            result["exc"] = e

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(3.0)
    t_cancel = time.monotonic()
    cancel.set()
    t.join(timeout=15)
    elapsed_after_cancel = time.monotonic() - t_cancel

    assert not t.is_alive(), "run_in_worker не вернулся за 15 с после отмены"
    assert isinstance(result.get("exc"), InterruptedError), result.get("exc")
    assert elapsed_after_cancel < 2.0, f"InterruptedError пришёл через {elapsed_after_cancel:.2f} с"

    time.sleep(0.3)  # дать ОС закрыть хэндлы убитого процесса
    children = psutil.Process().children(recursive=True)
    assert not children, f"остались процессы-воркеры: {children}"


# ── сирота: воркер не должен пережить родителя ────────────────────────────────

@pytest.mark.slow
def test_worker_exits_when_parent_is_gone(tmp_path):
    """Родителя с указанным PID никогда не существовало — сторожевой поток
    воркера обязан заметить это за пару секунд и выйти, не дожидаясь конца
    расчёта (иначе воркер остаётся сиротой, молотящей CPU). Реальная речь и
    2 минуты, не 20 (см. _write_looped_speech_wav) — иначе сторож не успевает
    получить GIL до конца сегментации."""
    wav = tmp_path / "speech.wav"
    _write_looped_speech_wav(wav, seconds=120)
    out = tmp_path / "out.json"

    cmd = [sys.executable, "-m", "app.main", "--diarize-worker", str(wav), str(out),
           "--speakers", "0", "--parent-pid", str(_dead_pid())]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    elapsed = time.monotonic() - t0

    assert elapsed <= 3.0, f"воркер вышел только через {elapsed:.2f} с: {proc.stderr[-500:]}"
    assert proc.returncode == 1, proc.stderr[-500:]
    assert not out.exists()


# ── ascii_safe_path (A1): чистая логика, кодовая страница — параметр ─────────

class TestAsciiSafePath:
    def test_non_windows_returns_unchanged(self, monkeypatch):
        monkeypatch.setattr(engine.sys, "platform", "linux")
        p = "/tmp/Папка тест/файл.onnx"
        assert engine.ascii_safe_path(p, acp=1251) == p

    def test_windows_utf8_codepage_returns_unchanged(self, monkeypatch):
        monkeypatch.setattr(engine.sys, "platform", "win32")
        p = r"C:\Users\Иван\model"
        assert engine.ascii_safe_path(p, acp=65001) == p

    def test_windows_ascii_path_returns_unchanged_even_without_utf8(self, monkeypatch):
        monkeypatch.setattr(engine.sys, "platform", "win32")
        p = r"C:\Users\Ivan\model"
        assert engine.ascii_safe_path(p, acp=1251) == p

    def test_windows_uses_short_name_when_it_becomes_ascii(self, monkeypatch):
        monkeypatch.setattr(engine.sys, "platform", "win32")
        monkeypatch.setattr(engine, "_short_path_name", lambda p: r"C:\Users\IVAN~1\model")
        p = r"C:\Users\Иван\model"
        assert engine.ascii_safe_path(p, acp=1251) == r"C:\Users\IVAN~1\model"

    def test_windows_falls_back_to_junction_when_8dot3_disabled(self, monkeypatch):
        """При выключенных коротких именах GetShortPathNameW возвращает тот же
        путь (см. ascii_safe_path) — тогда идём дальше, в junction/копию."""
        monkeypatch.setattr(engine.sys, "platform", "win32")
        p = r"C:\Users\Иван\model"
        monkeypatch.setattr(engine, "_short_path_name", lambda q: q)
        monkeypatch.setattr(engine, "_junction_or_copy",
                            lambda q: r"C:\Users\Public\Transkrib\m-deadbeef")
        assert engine.ascii_safe_path(p, acp=1251) == r"C:\Users\Public\Transkrib\m-deadbeef"
