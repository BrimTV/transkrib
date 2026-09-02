"""Тесты app/cleanup.py: фильтр галлюцинаций Whisper. Чистая логика на словарях
событий — без моделей и файлов, поэтому прогон занимает миллисекунды (см.
docs/TZ-release-hardening.md, Б1)."""
from app.cleanup import SegmentFilter


def seg(start, end, text, **extra):
    return dict(start=start, end=end, text=text, **extra)


# ── маркеры обучающих субтитров ─────────────────────────────────────────────

class TestMarker:
    def test_single_marker_phrase_dropped(self):
        f = SegmentFilter("ru")
        assert f.verdict(seg(0, 2, "Спасибо за просмотр")) == "marker"
        assert f.dropped == 1

    def test_marker_repeated_is_dropped_every_time(self):
        # «Продолжение следует» — короткая (2 слова) маркерная фраза: правило marker
        # проверяется раньше repeat и срабатывает у любого экземпляра независимо от
        # повтора, поэтому весь хвостовой мусор из пяти одинаковых сегментов уходит
        # целиком, а не «первый показан — остальные обрезаны» (это поведение repeat,
        # см. TestRepeat.test_repeat_drops_after_first_short_marker_free_phrase).
        # Отдельно проверяем, что это не проходит незамеченным: dropped считает все пять.
        f = SegmentFilter("ru")
        verdicts = [f.verdict(seg(i, i + 1, "Продолжение следует")) for i in range(5)]
        assert verdicts == ["marker"] * 5
        assert f.dropped == 5

    def test_long_sentence_with_marker_substring_not_dropped(self):
        # Ограничение в 12 слов защищает живую фразу, где маркерная подстрока —
        # часть настоящей речи, а не хвост из субтитров.
        f = SegmentFilter("ru")
        text = "Спасибо за просмотр, а теперь перейдём к делу и обсудим важные детали проекта"
        assert f.verdict(seg(0, 6, text)) is None

    def test_english_marker(self):
        f = SegmentFilter("en")
        assert f.verdict(seg(0, 2, "Thank you for watching")) == "marker"

    def test_language_scoping_ru_ignores_english_markers(self):
        f = SegmentFilter("ru")
        assert f.verdict(seg(0, 2, "Thank you for watching")) is None

    def test_language_scoping_en_ignores_russian_markers(self):
        f = SegmentFilter("en")
        assert f.verdict(seg(0, 2, "Спасибо за просмотр")) is None

    def test_auto_language_checks_both_lists(self):
        f_auto = SegmentFilter("auto")
        f_none = SegmentFilter(None)
        assert f_auto.verdict(seg(0, 2, "Спасибо за просмотр")) == "marker"
        assert f_auto.verdict(seg(2, 4, "Thank you for watching")) == "marker"
        assert f_none.verdict(seg(0, 2, "Subtitles by someone")) == "marker"


# ── повторы подряд ───────────────────────────────────────────────────────────

class TestRepeat:
    def test_short_reply_twice_kept_in_full(self):
        f = SegmentFilter("ru")
        assert f.verdict(seg(0, 1, "Да.")) is None
        assert f.verdict(seg(1, 2, "Да.")) is None
        assert f.dropped == 0

    def test_third_copy_of_short_text_dropped(self):
        f = SegmentFilter("ru")
        assert f.verdict(seg(0, 1, "Да.")) is None
        assert f.verdict(seg(1, 2, "Да.")) is None
        assert f.verdict(seg(2, 3, "Да.")) == "repeat"

    def test_repeat_drops_after_first_short_marker_free_phrase(self):
        # Фраза без маркеров, ≤3 слов: первая копия показана, вторая тоже (короткая
        # фраза не режется на 2-й копии), третья и далее — repeat.
        f = SegmentFilter("ru")
        text = "Хорошо, понял"
        r = [f.verdict(seg(i, i + 1, text)) for i in range(4)]
        assert r == [None, None, "repeat", "repeat"]

    def test_second_copy_of_long_phrase_dropped(self):
        # Фраза длиннее трёх слов: уже вторая копия отбрасывается.
        f = SegmentFilter("ru")
        text = "Я не знаю что сказать"
        assert f.verdict(seg(0, 2, text)) is None
        assert f.verdict(seg(2, 4, text)) == "repeat"

    def test_natural_repetition_inside_one_utterance_not_cut(self):
        # «Да, да, конечно» — законный повтор внутри живой русской речи, не должен
        # резаться: это один сегмент, а не последовательность одинаковых сегментов.
        f = SegmentFilter("ru")
        assert f.verdict(seg(0, 1, "Да, да, конечно")) is None

    def test_different_texts_reset_repeat_counter(self):
        f = SegmentFilter("ru")
        assert f.verdict(seg(0, 1, "Один")) is None
        assert f.verdict(seg(1, 2, "Два")) is None
        assert f.verdict(seg(2, 3, "Один")) is None  # снова первая копия, счётчик сброшен


# ── сигналы качества модели: no_speech_prob / avg_logprob / compression_ratio ──

class TestModelSignals:
    def test_low_no_speech_prob_passes_through(self):
        f = SegmentFilter("ru")
        text = "Сегодня мы обсудим результаты квартального отчёта и планы на следующий месяц"
        e = seg(0, 6, text, no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=1.4)
        assert f.verdict(e) is None

    def test_no_speech_and_low_logprob_dropped(self):
        f = SegmentFilter("ru")
        e = seg(0, 3, "какой-то шум", no_speech_prob=0.7, avg_logprob=-1.5)
        assert f.verdict(e) == "no_speech"

    def test_no_speech_prob_alone_not_enough(self):
        # Правило требует оба условия сразу — только высокий no_speech_prob без
        # низкого avg_logprob не отбрасывает.
        f = SegmentFilter("ru")
        e = seg(0, 3, "нормальный текст без второй метрики", no_speech_prob=0.9)
        assert f.verdict(e) is None

    def test_high_compression_ratio_dropped(self):
        f = SegmentFilter("ru")
        e = seg(0, 2, "какой-то повторяющийся текст", compression_ratio=3.1)
        assert f.verdict(e) == "compression"

    def test_missing_probability_keys_do_not_crash(self):
        f = SegmentFilter("ru")
        e = seg(0, 3, "обычный текст без вероятностей вообще")
        assert f.verdict(e) is None


# ── низкое разнообразие слов внутри одного сегмента (loop) ─────────────────────

class TestLoop:
    def test_repeated_marker_phrase_inside_one_segment_caught_by_loop(self):
        # Живой случай из ТЗ: тон 440 Гц → «Редактор субтитров А.Синецкая» двадцать
        # раз одним сегментом. Суммарно больше 12 слов, поэтому правило marker не
        # применяется (защита длинных фраз), но ловит loop по низкому разнообразию.
        f = SegmentFilter("ru")
        text = " ".join(["Редактор субтитров А.Синецкая."] * 20)
        assert f.verdict(seg(0, 20, text)) == "loop"

    def test_legit_repetition_with_enough_unique_words_not_cut(self):
        text = "да да я понимаю повторяю да да это так"
        f = SegmentFilter("ru")
        assert f.verdict(seg(0, 5, text)) is None


# ── неправдоподобная плотность слов в секунду (density) ─────────────────────

class TestDensity:
    def test_too_many_words_per_second_dropped(self):
        f = SegmentFilter("ru")
        text = "раз два три четыре пять шесть семь восемь девять десять"
        assert f.verdict(seg(0, 1.3, text)) == "density"

    def test_short_duration_not_penalized(self):
        # Правило density применяется только при длительности больше секунды.
        f = SegmentFilter("ru")
        text = "раз два три четыре пять шесть семь восемь девять десять"
        assert f.verdict(seg(0, 0.5, text)) is None


# ── dropped и reset ───────────────────────────────────────────────────────────

class TestState:
    def test_dropped_counts_only_dropped_segments(self):
        f = SegmentFilter("ru")
        f.verdict(seg(0, 1, "Обычная фраза один"))
        f.verdict(seg(1, 2, "Спасибо за просмотр"))
        f.verdict(seg(2, 3, "Другая обычная фраза"))
        assert f.dropped == 1

    def test_reset_clears_dropped_and_repeat_state(self):
        f = SegmentFilter("ru")
        f.verdict(seg(0, 1, "Спасибо за просмотр"))
        f.verdict(seg(1, 2, "Да."))
        f.verdict(seg(2, 3, "Да."))
        assert f.dropped == 1
        f.reset()
        assert f.dropped == 0
        # После сброса счётчик повторов тоже начинается заново: та же фраза
        # снова считается первой копией и не режется.
        assert f.verdict(seg(0, 1, "Да.")) is None
        assert f.verdict(seg(1, 2, "Да.")) is None
