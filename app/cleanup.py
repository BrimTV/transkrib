"""Фильтр галлюцинаций Whisper: чистая логика, без импорта моделей/сети/файловой
системы — тесты должны идти мгновенно (см. docs/TZ-release-hardening.md, Б1).

Оба движка (faster-whisper и mlx_whisper) отдают на музыке, джинглах и тишине один
из двух видов мусора: либо повтор одной и той же фразы много раз подряд, либо
фразу-маркер из обучающих субтитров («Субтитры сделал DimaTorzok», «Продолжение
следует», «Спасибо за просмотр» и т.п.). SegmentFilter.verdict() решает по каждому
событию segment, показывать его пользователю или нет, используя либо служебные
метрики модели (no_speech_prob, avg_logprob, compression_ratio — если движок их
передал), либо форму самого текста (маркер, повтор, низкое разнообразие слов,
неправдоподобная плотность речи).

Подключение к движку (обёртка emit() в engine.transcribe_file) — забота другого
агента; этот модуль сам по себе ничего не решает о потоке событий.
"""
import re

# Фразы-маркеры из обучающих субтитров: если сегмент короткий (см. MARKER_MAX_WORDS)
# и содержит одну из них после нормализации — почти наверняка не то, что действительно
# сказано в записи, а строка, которую Whisper выучил из субтитров на YouTube.
MARKERS_RU = (
    "субтитры сделал",
    "субтитры создал",
    "субтитры подготовил",
    "редактор субтитров",
    "корректор",
    "продолжение следует",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "ставьте лайк",
    "dimatorzok",
)
MARKERS_EN = (
    "thank you for watching",
    "thanks for watching",
    "subtitles by",
    "subs by",
    "please subscribe",
    "like and subscribe",
    "amara org",
    "www",
    "copyright",
)

# Пороги — подобраны по описанному в ТЗ живому случаю (тон 440 Гц → «Редактор
# субтитров А.Синецкая» двадцать раз одним сегментом) и по обычным сигналам качества
# Whisper; не претендуют на идеальность, но не должны резать обычную речь с легитимными
# повторами («Да, да, конечно»).
NO_SPEECH_PROB_MIN = 0.6
AVG_LOGPROB_MAX = -1.0
COMPRESSION_RATIO_MAX = 2.4
MARKER_MAX_WORDS = 12
REPEAT_SHORT_WORDS_MAX = 3  # до стольки слов включительно короткая фраза (типа «Да, да») не режется на 2-й копии
LOOP_MIN_WORDS = 8
LOOP_MAX_UNIQUE_RATIO = 0.35
DENSITY_WORDS_PER_SEC_MAX = 7
DENSITY_MIN_DURATION = 1.0

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")


def normalize(text):
    """Нижний регистр, ё→е, без пунктуации, схлопнутые пробелы — общий вид текста
    и для сравнения на повтор, и для поиска маркеров."""
    t = (text or "").lower().replace("ё", "е")
    t = _PUNCT_RE.sub(" ", t)
    return _SPACES_RE.sub(" ", t).strip()


class SegmentFilter:
    """Стейтфул-фильтр на один файл: помнит предыдущий текст (для счётчика повторов)
    и число отброшенных сегментов. language — код языка расшифровки ('ru', 'en', ...),
    либо 'auto'/None, когда язык ещё не определён или определяется автоматически —
    в этом случае проверяются оба списка маркеров, чтобы не пропустить мусор."""

    def __init__(self, language=None):
        self.language = language
        self.reset()

    def reset(self):
        """Между файлами: сбросить счётчик повторов и статистику отброшенного."""
        self._prev_norm = None
        self._repeat_count = 0
        self.dropped = 0

    def _markers(self):
        lang = (self.language or "auto").lower()
        if lang.startswith("ru"):
            return MARKERS_RU
        if lang.startswith("en"):
            return MARKERS_EN
        return MARKERS_RU + MARKERS_EN

    def verdict(self, event):
        """Причина отброса события segment (str) или None, если сегмент показываем.
        Правила проверяются по порядку, первое сработавшее и определяет причину —
        но счётчик повторов всё равно обновляется по факту текста, а не только когда
        до него доходит очередь, иначе следующий вызов не будет знать, что текст
        уже повторяется."""
        text = event.get("text") or ""
        norm = normalize(text)
        words = norm.split()
        word_count = len(words)
        start = event.get("start") or 0
        end = event.get("end") or 0
        duration = float(end) - float(start)

        reason = None

        no_speech_prob = event.get("no_speech_prob")
        avg_logprob = event.get("avg_logprob")
        if (
            no_speech_prob is not None
            and avg_logprob is not None
            and no_speech_prob > NO_SPEECH_PROB_MIN
            and avg_logprob < AVG_LOGPROB_MAX
        ):
            reason = "no_speech"

        if reason is None:
            compression_ratio = event.get("compression_ratio")
            if compression_ratio is not None and compression_ratio > COMPRESSION_RATIO_MAX:
                reason = "compression"

        if reason is None and word_count <= MARKER_MAX_WORDS:
            markers = self._markers()
            if any(m in norm for m in markers):
                reason = "marker"

        # Счётчик повторов подряд: считаем, сколько раз подряд пришёл тот же
        # нормализованный текст. Первый экземпляр никогда не отзывается этим
        # правилом — он уже показан пользователю и может быть настоящей речью.
        is_repeat = self._prev_norm is not None and norm == self._prev_norm
        self._repeat_count = self._repeat_count + 1 if is_repeat else 1

        if reason is None and is_repeat:
            if self._repeat_count >= 3 or (
                self._repeat_count == 2 and word_count > REPEAT_SHORT_WORDS_MAX
            ):
                reason = "repeat"

        if reason is None and word_count >= LOOP_MIN_WORDS:
            unique_ratio = len(set(words)) / word_count
            if unique_ratio < LOOP_MAX_UNIQUE_RATIO:
                reason = "loop"

        if reason is None and duration > DENSITY_MIN_DURATION:
            if word_count / duration > DENSITY_WORDS_PER_SEC_MAX:
                reason = "density"

        self._prev_norm = norm
        if reason is not None:
            self.dropped += 1
        return reason
