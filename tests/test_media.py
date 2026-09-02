"""Тесты app/media.py: понятные ошибки вместо сырых (Б5) и отмена ffmpeg (A7)."""
import threading
import time

import psutil
import pytest
import requests

from app import engine, media


# ── check_readable / extract_wav: понятные ошибки вместо сырых (Б5) ──────────

def test_missing_file_raises_user_error(tmp_path):
    missing = tmp_path / "нет-такого.mp4"
    with pytest.raises(engine.UserError, match="Файл не найден"):
        media.check_readable(str(missing))


def test_empty_file_raises_user_error(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(engine.UserError, match="пустой"):
        media.check_readable(str(empty))


def test_extract_wav_missing_file(tmp_path):
    with pytest.raises(engine.UserError, match="Файл не найден"):
        media.extract_wav(str(tmp_path / "нет.mp4"), str(tmp_path / "out.wav"))


def test_extract_wav_empty_file(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(engine.UserError, match="пустой"):
        media.extract_wav(str(empty), str(tmp_path / "out.wav"))


def test_extract_wav_no_audio_track(noaudio_mp4, tmp_path):
    with pytest.raises(engine.UserError, match="нет звуковой дорожки"):
        media.extract_wav(str(noaudio_mp4), str(tmp_path / "out.wav"))


def test_extract_wav_broken_file(broken_mp4, tmp_path):
    with pytest.raises(engine.UserError, match="повреждён или недокачан"):
        media.extract_wav(str(broken_mp4), str(tmp_path / "out.wav"))


# ── -map 0:a:0 и число дорожек (Б5/Б6) ────────────────────────────────────────

def test_extract_wav_reports_track_count(speech_ru_10s, tmp_path):
    dst = tmp_path / "out.wav"
    duration, tracks = media.extract_wav(str(speech_ru_10s), str(dst))
    assert tracks == 1
    assert dst.exists()
    assert duration > 0


def test_extract_wav_multitrack_takes_first(multitrack_mkv, tmp_path):
    """Файл с двумя аудиодорожками (как из OBS) — берём первую (-map 0:a:0),
    но сообщаем количество, чтобы engine мог показать «дорожек: N, взята первая»."""
    dst = tmp_path / "out.wav"
    duration, tracks = media.extract_wav(str(multitrack_mkv), str(dst))
    assert tracks == 2
    assert dst.exists()


# ── скачивание модели без интернета (Б5) ──────────────────────────────────────

def test_ensure_model_no_internet(monkeypatch, tmp_path):
    """Подмена функции скачивания, которая бросает requests.ConnectionError —
    имитация настоящего отсутствия сети."""
    monkeypatch.setattr(engine, "data_dir", lambda: str(tmp_path))

    import huggingface_hub

    class _FakeHfApi:
        def model_info(self, *a, **k):
            raise requests.ConnectionError("no route to host")

    def _fake_download(*a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_download)

    with pytest.raises(engine.UserError, match="Нет доступа к интернету"):
        engine.ensure_model("tiny", "cpu", lambda e: None)


# ── отмена ffmpeg (A7) ────────────────────────────────────────────────────────

def test_extract_wav_cancel_is_fast(long_silence_wav, tmp_path):
    """Кнопка «Остановить» во время извлечения звука из длинного файла должна
    сработать быстро, а не ждать, пока ffmpeg досчитает до конца."""
    cancel = threading.Event()
    dst = tmp_path / "out.wav"
    result = {}

    def worker():
        try:
            media.extract_wav(str(long_silence_wav), str(dst), cancel=cancel)
        except BaseException as e:
            result["exc"] = e

    t = threading.Thread(target=worker)
    t0 = time.monotonic()
    t.start()
    cancel.set()
    t.join(timeout=5)
    elapsed = time.monotonic() - t0

    assert not t.is_alive()
    assert isinstance(result.get("exc"), InterruptedError)
    assert elapsed < 1.0

    time.sleep(0.3)  # дать ОС закрыть хэндлы убитого процесса
    leftover = [p for p in psutil.process_iter(["name"])
                if "ffmpeg" in (p.info.get("name") or "").lower()]
    assert not leftover, f"остались процессы ffmpeg: {leftover}"
