"""Тесты app/media.speech_regions (Б2, Silero VAD) и защита от тихих записей
(Б6, часть про переизвлечение с усилением). Фикстуры — из tests/conftest.py:
speech_silence_speech, silence60 и quiet_wav уже готовы под эти проверки, см.
docs/TZ-release-hardening.md, Б2/Б6.

Сюда же (не заводя отдельный файл — правила задачи ограничивают набор файлов,
которые можно трогать) — тесты engine.cpu_workers (Б4: число потоков под
нагрузку) и engine.cleanup_temp/лог с ротацией (В3: временные папки и лог)."""
import os
import time

import psutil

from app import engine, media


def test_speech_silence_speech_finds_two_regions(speech_silence_speech):
    """Речь 10 с → тишина 40 с → та же речь 10 с: ровно два региона, границы
    в пределах 0.5 с от истинных (речь 0-10 и 50-60 — сорок секунд тишины
    после первых десяти секунд речи)."""
    path = str(speech_silence_speech)
    total = media.wav_duration(path)
    regions = media.speech_regions(path, total)

    assert len(regions) == 2
    (a0, a1), (b0, b1) = regions
    assert abs(a0 - 0.0) <= 0.5
    assert abs(a1 - 10.0) <= 0.5
    assert abs(b0 - 50.0) <= 0.5
    assert abs(b1 - 60.0) <= 0.5


def test_silence60_finds_nothing(silence60):
    """60 с полной тишины — интервалов речи нет вообще."""
    path = str(silence60)
    total = media.wav_duration(path)
    regions = media.speech_regions(path, total)
    assert regions == []


def test_quiet_recording_still_finds_speech(quiet_wav, tmp_path):
    """Тихая запись (-32 дБ): после появления VAD речь на таком уровне рискует
    потеряться в порогах Silero. media.extract_wav переизвлекает её с
    усилением (Б6), и VAD после этого всё равно находит речь."""
    dst = tmp_path / "boosted.wav"
    total, _ = media.extract_wav(str(quiet_wav), str(dst))
    regions = media.speech_regions(str(dst), total)
    assert regions != []


def test_region_gaps_only_keeps_gaps_at_least_min_gap():
    """Середины промежутков между интервалами речи — только там, где пауза не
    короче min_gap; более короткие промежутки не дают точку реза."""
    regions = [(0.0, 10.0), (10.3, 15.0), (20.0, 30.0)]
    gaps = media.region_gaps(regions, min_gap=0.6)
    # 10.0→10.3 короче 0.6 с — не попадает; 15.0→20.0 (5 с) попадает, середина 17.5
    assert gaps == [17.5]


def test_speech_regions_reads_in_blocks(speech_silence_speech):
    """Маленький block всё равно должен находить оба региона — проверяем, что
    резка на блоки по границе не теряет и не дублирует речь на стыке."""
    path = str(speech_silence_speech)
    total = media.wav_duration(path)
    regions = media.speech_regions(path, total, block=15.0)
    assert len(regions) == 2


# ── engine.cpu_workers (Б4) ────────────────────────────────────────────────

def test_cpu_workers_leaves_one_free_on_four_or_more(monkeypatch):
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 8 if not logical else 16)
    assert engine.cpu_workers() == 7


def test_cpu_workers_four_cores(monkeypatch):
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 4 if not logical else 8)
    assert engine.cpu_workers() == 3


def test_cpu_workers_uses_all_below_four(monkeypatch):
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 2 if not logical else 4)
    assert engine.cpu_workers() == 2


def test_cpu_workers_falls_back_to_os_cpu_count(monkeypatch):
    """psutil не смог определить физические ядра (None) — берём os.cpu_count()."""
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: None if not logical else 4)
    monkeypatch.setattr(engine.os, "cpu_count", lambda: 6)
    assert engine.cpu_workers() == 5


# ── engine.cleanup_temp и лог с ротацией (В3) ──────────────────────────────

def test_cleanup_temp_removes_old_keeps_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "data_dir", lambda: str(tmp_path))
    old = os.path.join(engine.temp_dir(), "transkrib_old")
    fresh = os.path.join(engine.temp_dir(), "transkrib_fresh")
    os.makedirs(old)
    os.makedirs(fresh)
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))

    engine.cleanup_temp(max_age_sec=3600)

    assert not os.path.exists(old)
    assert os.path.exists(fresh)


def test_log_rotation_caps_total_size(tmp_path, monkeypatch):
    """3 МБ строк в лог с maxBytes=2 МБ, backupCount=2 → ровно .log и .log.1,
    суммарный размер ограничен (см. docs/TZ-release-hardening.md, В3)."""
    monkeypatch.setattr(engine, "data_dir", lambda: str(tmp_path))
    engine._logger = None
    try:
        line = "x" * 200
        for _ in range(3 * 1024 * 1024 // 200):
            engine.log(line)

        log_path = os.path.join(str(tmp_path), "transkrib.log")
        backup_path = log_path + ".1"
        assert os.path.exists(log_path)
        assert os.path.exists(backup_path)

        total = os.path.getsize(log_path) + os.path.getsize(backup_path)
        assert total < 4.1 * 1024 * 1024
    finally:
        # Не оставляем закэшированный логгер на чужом data_dir() для остальных тестов.
        for h in engine._logger.handlers[:] if engine._logger else []:
            h.close()
        engine._logger = None
