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
