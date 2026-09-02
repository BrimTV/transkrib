"""Тесты app/export.py: таймкоды, стили меток, блоки по паузам/говорящим, форматы."""
import pytest

from app import export


def seg(start, end, text, speaker=None):
    s = dict(start=start, end=end, text=text)
    if speaker is not None:
        s["speaker"] = speaker
    return s


# ── stamp: стили меток ──────────────────────────────────────────────────────

class TestStamp:
    def test_brackets_short_under_minute(self):
        assert export.stamp(5, "brackets_short") == "[0:05]"

    def test_brackets_short_over_minute(self):
        assert export.stamp(65, "brackets_short") == "[1:05]"

    def test_brackets_short_over_hour(self):
        assert export.stamp(3665, "brackets_short") == "[1:01:05]"

    def test_brackets_long_pads_hour(self):
        assert export.stamp(65, "brackets_long") == "[00:01:05]"
        assert export.stamp(7384, "brackets_long") == "[02:03:04]"

    def test_plain_short(self):
        assert export.stamp(65, "plain_short") == "1:05"
        assert export.stamp(7384, "plain_short") == "2:03:04"

    def test_plain_long(self):
        assert export.stamp(65, "plain_long") == "00:01:05"

    def test_timecode_over_an_hour_in_txt(self):
        out = export.to_txt([seg(3665, 3670, "час прошёл")])
        assert out.startswith("[1:01:05] ")


# ── режимы таймкодов ─────────────────────────────────────────────────────────

class TestTsModes:
    def test_none_has_no_timestamp(self):
        out = export.to_txt([seg(0, 1, "привет")], dict(ts_mode="none"))
        assert out == "привет\n"

    def test_paragraph_splits_on_pause(self):
        segs = [seg(0, 1, "a"), seg(5, 6, "b")]  # разрыв 4с > paragraph_gap=2.5
        blocks = export._blocks(segs, export.DEFAULT_OPTS)
        assert [b[0] for b in blocks] == [0, 5]

    def test_paragraph_merges_close_segments(self):
        segs = [seg(0, 1, "a"), seg(1.5, 2, "b")]  # разрыв 0.5с < 2.5
        blocks = export._blocks(segs, export.DEFAULT_OPTS)
        assert len(blocks) == 1
        assert blocks[0][1] == ["a", "b"]

    def test_interval_marks_by_fixed_step(self):
        segs = [seg(0, 5, "a"), seg(10, 15, "b"), seg(65, 70, "c")]
        opts = {**export.DEFAULT_OPTS, "ts_mode": "interval", "ts_interval": 60}
        blocks = export._blocks(segs, opts)
        assert [b[0] for b in blocks] == [0, 65]
        assert blocks[0][1] == ["a", "b"]

    def test_interval_step_has_minimum_of_5(self):
        segs = [seg(0, 1, "a"), seg(1, 2, "b")]
        opts = {**export.DEFAULT_OPTS, "ts_mode": "interval", "ts_interval": 1}
        blocks = export._blocks(segs, opts)
        # min step = 5, оба сегмента внутри первого окна → один блок
        assert len(blocks) == 1

    def test_segment_mode_never_merges(self):
        segs = [seg(0, 1, "a"), seg(1.1, 2, "b")]  # разрыв меньше paragraph_gap
        opts = {**export.DEFAULT_OPTS, "ts_mode": "segment"}
        blocks = export._blocks(segs, opts)
        assert len(blocks) == 2

    def test_segment_mode_uses_single_newline(self):
        out = export.to_txt([seg(0, 1, "a"), seg(1, 2, "b")], dict(ts_mode="segment"))
        assert "\n\n" not in out
        assert out.count("\n") == 2


# ── говорящий начинает новый блок ────────────────────────────────────────────

class TestSpeakerChangeBlocks:
    def test_speaker_change_forces_new_block_even_within_pause_gap(self):
        # разрыв 0.1с << paragraph_gap, но говорящий сменился
        segs = [seg(0, 1, "a", speaker=1), seg(1.1, 2, "b", speaker=2)]
        blocks = export._blocks(segs, export.DEFAULT_OPTS)
        assert len(blocks) == 2
        assert blocks[0][2] == "Говорящий 1"
        assert blocks[1][2] == "Говорящий 2"

    def test_same_speaker_still_merges_on_short_gap(self):
        segs = [seg(0, 1, "a", speaker=1), seg(1.1, 2, "b", speaker=1)]
        blocks = export._blocks(segs, export.DEFAULT_OPTS)
        assert len(blocks) == 1


# ── префикс говорящего в разных форматах ────────────────────────────────────

class TestSpeakerPrefix:
    def test_txt_numeric_speaker_label(self):
        out = export.to_txt([seg(0, 1, "привет", speaker=1)])
        assert "Говорящий 1: привет" in out

    def test_txt_named_speaker_used_as_is(self):
        out = export.to_txt([seg(0, 1, "привет", speaker="Иван")])
        assert "Иван: привет" in out

    def test_md_speaker_bold(self):
        out = export.to_md([seg(0, 1, "привет", speaker="Иван")])
        assert "**Иван:** привет" in out

    def test_srt_speaker_prefix(self):
        out = export.to_srt([seg(0, 1, "привет", speaker="Иван")])
        assert "Иван: привет" in out

    def test_vtt_speaker_prefix(self):
        out = export.to_vtt([seg(0, 1, "привет", speaker="Иван")])
        assert "Иван: привет" in out

    def test_no_speakers_does_not_break_output(self):
        segs = [seg(0, 1, "a"), seg(2, 3, "b")]
        assert export.to_txt(segs)
        assert export.to_md(segs)
        assert export.to_srt(segs)
        assert export.to_vtt(segs)
        # ни в одном формате нет ложного префикса говорящего
        for out in (export.to_txt(segs), export.to_md(segs), export.to_srt(segs), export.to_vtt(segs)):
            assert "Говорящий" not in out


# ── srt / vtt: базовая структура ────────────────────────────────────────────

class TestSrtVtt:
    def test_srt_numbers_and_arrow(self):
        out = export.to_srt([seg(0, 1.5, "текст")])
        assert out.startswith("1\n00:00:00,000 --> 00:00:01,500\nтекст\n")

    def test_vtt_header_and_dot_separator(self):
        out = export.to_vtt([seg(0, 1.5, "текст")])
        assert out.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nтекст\n")


# ── render(): диспетчер форматов ────────────────────────────────────────────

class TestRender:
    @pytest.mark.parametrize("fmt", ["txt", "md", "srt", "vtt"])
    def test_render_dispatches_known_formats(self, fmt):
        assert export.render([seg(0, 1, "a")], fmt)

    def test_render_unknown_format_raises(self):
        with pytest.raises(ValueError):
            export.render([seg(0, 1, "a")], "docx")


# ── граничные случаи ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_segments_list_does_not_crash(self):
        assert export.to_txt([]) == "\n"
        assert export.to_md([]) == "\n"
        assert export.to_srt([]) == ""
        assert export.to_vtt([]) == "WEBVTT\n"

    def test_segment_with_empty_text_is_skipped_in_txt(self):
        segs = [seg(0, 1, ""), seg(1, 2, "текст")]
        out = export.to_txt(segs)
        assert out.count("\n\n") == 0  # пустой сегмент не породил отдельный блок
        assert "текст" in out

    def test_segment_with_empty_text_still_present_in_srt(self):
        # to_srt/to_vtt не фильтруют пустые сегменты (в отличие от _blocks/to_txt)
        segs = [seg(0, 1, ""), seg(1, 2, "текст")]
        out = export.to_srt(segs)
        assert out.count("00:00:0") >= 2

    def test_whitespace_only_text_treated_as_empty(self):
        segs = [seg(0, 1, "   "), seg(1, 2, "текст")]
        out = export.to_txt(segs)
        assert out.strip() == "[0:01] текст"
