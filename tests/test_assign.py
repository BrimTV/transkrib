"""Тесты app/diarize.py::assign и ::_finalize — чистая логика без sherpa-onnx.

Модуль app/diarize.py импортирует sherpa_onnx только внутри функций
available()/run(), поэтому обычный `from app import diarize` моделей не
грузит и sherpa-onnx не требует.
"""
from app import diarize


def turn(start, end, speaker):
    return dict(start=start, end=end, speaker=speaker)


def seg(start, end, text):
    return dict(start=start, end=end, text=text)


# ── assign: привязка фразы к говорящему ──────────────────────────────────────

class TestAssignOverlap:
    def test_picks_speaker_with_largest_overlap(self):
        turns = [turn(0, 5, 1), turn(5, 10, 2)]
        segs = [seg(1, 4, "a"), seg(6, 9, "b")]
        out = diarize.assign(segs, turns)
        assert out[0]["speaker"] == 1
        assert out[1]["speaker"] == 2

    def test_larger_overlap_wins_over_smaller(self):
        # фраза 4..9: перекрытие с t1 (0..5) = 1с, с t2 (5..12) = 4с
        turns = [turn(0, 5, 1), turn(5, 12, 2)]
        segs = [seg(4, 9, "фраза")]
        out = diarize.assign(segs, turns)
        assert out[0]["speaker"] == 2

    def test_tie_overlap_keeps_first_turn(self):
        turns = [turn(0, 10, 1), turn(0, 10, 2)]
        segs = [seg(2, 8, "фраза")]
        out = diarize.assign(segs, turns)
        assert out[0]["speaker"] == 1


class TestAssignNoOverlap:
    def test_no_overlap_but_near_picks_closest_turn(self):
        turns = [turn(0, 5, 1), turn(5, 10, 2)]
        # фраза 10.5..11, середина 10.75, ближайшая граница — конец t2 (10), расстояние 0.75 < 3
        segs = [seg(10.5, 11, "фраза")]
        out = diarize.assign(segs, turns)
        assert out[0]["speaker"] == 2

    def test_no_overlap_far_falls_back_to_previous_speaker(self):
        turns = [turn(0, 5, 1), turn(5, 10, 2)]
        # первая фраза перекрывается с t1 → speaker=1 становится "предыдущим"
        # вторая фраза далеко (100с) от обоих turns → достаётся предыдущему (1)
        segs = [seg(1, 4, "рядом"), seg(100, 101, "далеко")]
        out = diarize.assign(segs, turns)
        assert out[0]["speaker"] == 1
        assert out[1]["speaker"] == 1

    def test_exactly_three_seconds_is_not_near_uses_previous(self):
        turns = [turn(0, 5, 1), turn(5, 10, 2)]
        # середина фразы 13, до конца t2 (10) ровно 3.0 — не "< 3", идёт к предыдущему
        segs = [seg(13, 13, "граница")]
        out = diarize.assign(segs, turns)
        assert out[0]["speaker"] == 1  # предыдущий по умолчанию (last=1), т.к. первый в очереди

    def test_default_previous_speaker_before_any_assignment_is_one(self):
        turns = [turn(20, 25, 1), turn(25, 30, 2)]
        # первая же фраза далеко от обоих turns (>3с) → достаётся дефолтному "предыдущему" = 1
        segs = [seg(0, 1, "далеко от начала")]
        out = diarize.assign(segs, turns)
        assert out[0]["speaker"] == 1


class TestAssignNoTurns:
    def test_no_turns_returns_segments_unchanged(self):
        segs = [seg(0, 1, "x")]
        out = diarize.assign(segs, [])
        assert out == segs
        assert "speaker" not in out[0]


# ── _finalize: склейка обрывков и нумерация ──────────────────────────────────

class TestFinalizeNumbering:
    def test_numbers_speakers_from_one_in_order_of_appearance(self):
        raw = [turn(0, 5, 7), turn(5, 10, 3), turn(10, 15, 7)]
        out = diarize._finalize(raw)
        assert [t["speaker"] for t in out] == [1, 2, 1]

    def test_rounds_start_end_to_two_decimals(self):
        raw = [dict(start=1.23456, end=2.6789, speaker=1)]
        out = diarize._finalize(raw)
        assert out[0]["start"] == 1.23
        assert out[0]["end"] == 2.68

    def test_empty_raw_returns_empty(self):
        assert diarize._finalize([]) == []


class TestFinalizeGluing:
    def test_short_fragment_glued_to_nearest_major_neighbor(self):
        # обрывок 1с (< MINOR_SEC=3.0) между двумя долгими репликами одного спикера
        raw = [turn(0, 10, 1), turn(10, 11, 2), turn(11, 20, 1)]
        out = diarize._finalize(raw)
        assert [t["speaker"] for t in out] == [1, 1, 1]

    def test_forced_does_not_glue_even_short_fragments(self):
        raw = [turn(0, 10, 1), turn(10, 11, 2), turn(11, 20, 1)]
        out = diarize._finalize(raw, forced=True)
        # при forced=True все говорящие считаются «крупными» — обрывок не клеится,
        # нумерация идёт в порядке появления: 1, 2, 1
        assert [t["speaker"] for t in out] == [1, 2, 1]

    def test_long_enough_fragment_not_glued(self):
        # обе реплики длиннее MINOR_SEC=3.0 → обе «крупные», ничего не клеится
        raw = [turn(0, 10, 1), turn(10, 15, 2), turn(15, 20, 1)]
        out = diarize._finalize(raw)
        assert [t["speaker"] for t in out] == [1, 2, 1]

    def test_all_minor_speakers_treated_as_major(self):
        # если "крупных" не нашлось вообще (весь материал короткий) — не клеим никого
        raw = [turn(0, 1, 1), turn(1, 2, 2)]
        out = diarize._finalize(raw)
        assert [t["speaker"] for t in out] == [1, 2]
