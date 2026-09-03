"""Тесты media.stereo_turns (В6): говорящие по каналам вместо sherpa-onnx
для звонков Zoom/Telegram/телефонии, где собеседники разложены по каналам."""
from app import media


def test_stereo_call_two_speakers_by_channel(stereo_call_wav):
    """Голос слева 0–10 с, справа 12–22 с — ровно два говорящих, реплики на
    своих местах (допуск ±1 с — сглаживание масок режет по границам окон 100 мс)."""
    turns = media.stereo_turns(str(stereo_call_wav), 22.0)
    assert turns is not None
    speakers = sorted({t["speaker"] for t in turns})
    assert speakers == [1, 2]

    left = [t for t in turns if t["speaker"] == 1]
    right = [t for t in turns if t["speaker"] == 2]
    assert left and right

    assert abs(min(t["start"] for t in left) - 0.0) <= 1.0
    assert abs(max(t["end"] for t in left) - 10.0) <= 1.0
    assert abs(min(t["start"] for t in right) - 12.0) <= 1.0
    assert abs(max(t["end"] for t in right) - 22.0) <= 1.0

    # turns отсортированы по началу — так же, как их отдаёт диаризация
    assert turns == sorted(turns, key=lambda t: t["start"])


def test_stereo_same_rejected(stereo_same_wav):
    """Та же речь одинаково в обоих каналах — обычная стереозапись одним
    микрофоном, не звонок: каналы звучат вместе, взаимоисключаемость мала."""
    assert media.stereo_turns(str(stereo_same_wav), 10.0) is None


def test_mono_rejected(speech_ru_10s):
    """Один канал — способ неприменим в принципе, даже пробовать нечего."""
    assert media.stereo_turns(str(speech_ru_10s), 10.0) is None


def test_silence_rejected(silence60):
    """Тишина — фикстура сама моно (см. conftest), но даже будь она стерео,
    оба канала молчат, и активных окон меньше порога в 5%."""
    assert media.stereo_turns(str(silence60), 60.0) is None


def test_длинный_файл_сначала_слушается_кусками(stereo_call_wav, monkeypatch):
    """На длинной записи сначала берётся проба в нескольких местах, и только если
    она похожа на звонок — файл разбирается целиком. Иначе каждая обычная
    стереозапись стоила бы лишнего полного декодирования. Пороги здесь опущены,
    чтобы двухступенчатый путь отработал на короткой фикстуре."""
    monkeypatch.setattr(media, "_STEREO_PROBE_SEC", 4.0)
    monkeypatch.setattr(media, "_STEREO_PROBE_AT", (0.05, 0.55))
    calls = []
    original = media._channel_envelopes

    def spy(src, total_sec, cancel=None, limit_sec=None, seek=None):
        calls.append((limit_sec, None if seek is None else round(seek, 1)))
        return original(src, total_sec, cancel, limit_sec, seek)

    monkeypatch.setattr(media, "_channel_envelopes", spy)
    turns = media.stereo_turns(str(stereo_call_wav), 22.0)

    assert calls == [(4.0, 1.1), (4.0, 12.1), (None, None)], "две пробы, затем разбор целиком"
    assert sorted({t["speaker"] for t in turns}) == [1, 2]
    # Реплика справа начинается на 12-й секунде — дальше первой пробы: значит
    # разбирался весь файл, а решение принято не по одному его началу.
    assert max(t["end"] for t in turns) > 12.0


def test_проба_видит_позднего_собеседника(stereo_call_wav, monkeypatch):
    """Второй голос звучит только с 12-й секунды. Проба, слушающая одно начало,
    решила бы, что это не звонок; разнесённая — видит обоих."""
    monkeypatch.setattr(media, "_STEREO_PROBE_SEC", 4.0)
    monkeypatch.setattr(media, "_STEREO_PROBE_AT", (0.05,))
    assert media.stereo_turns(str(stereo_call_wav), 22.0) is None

    monkeypatch.setattr(media, "_STEREO_PROBE_AT", (0.05, 0.55))
    assert media.stereo_turns(str(stereo_call_wav), 22.0) is not None


def test_обычное_стерео_не_декодируется_целиком(stereo_same_wav, monkeypatch):
    monkeypatch.setattr(media, "_STEREO_PROBE_SEC", 4.0)
    monkeypatch.setattr(media, "_STEREO_PROBE_AT", (0.05, 0.55))
    calls = []
    original = media._channel_envelopes

    def spy(src, total_sec, cancel=None, limit_sec=None, seek=None):
        calls.append(limit_sec)
        return original(src, total_sec, cancel, limit_sec, seek)

    monkeypatch.setattr(media, "_channel_envelopes", spy)
    assert media.stereo_turns(str(stereo_same_wav), 22.0) is None
    assert calls == [4.0, 4.0], "после отказа на пробе полного разбора быть не должно"
