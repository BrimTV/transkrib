"""Общие фикстуры pytest: синтетические аудио- и видеофайлы для тестов.

Все фикстуры, кроме одной, генерируются на лету через ffmpeg из imageio_ffmpeg
(тот же бинарник, что использует само приложение — см. app/media.ffmpeg_exe) в
tmp_path_factory: бинарники в репозиторий не кладём, генерация всего набора
занимает секунды, а не десятки секунд. Исключение — tests/fixtures/speech_ru_10s.wav:
живая TTS-речь на русском (сгенерирована один раз через `say -v Milena` + ffmpeg),
нужна тестам, которые не должны зависеть от наличия macOS в момент прогона.

Диаризация синтетические голоса TTS не различает (см. app/diarize.py) — эта
фикстура годится только для тестов текста (export, VAD, фильтр галлюцинаций),
не для тестов разделения по говорящим.

Все фикстуры scope="session" — генерируются один раз на весь прогон.
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _ffmpeg(*args):
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="session")
def speech_ru_10s():
    """Живая (TTS) русская речь, ~10 с, 16 кГц моно — единственный бинарник в репозитории."""
    return FIXTURES_DIR / "speech_ru_10s.wav"


@pytest.fixture(scope="session")
def silence60(tmp_path_factory):
    """60 с полной тишины, 16 кГц моно."""
    path = tmp_path_factory.mktemp("fixtures") / "silence60.wav"
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "60", str(path))
    return path


@pytest.fixture(scope="session")
def long_silence_wav(tmp_path_factory):
    """30 минут тишины, исходник для теста отмены ffmpeg (A7): длинный файл нужен,
    чтобы извлечение звука ещё шло, когда тест взводит cancel."""
    path = tmp_path_factory.mktemp("fixtures") / "long_silence.wav"
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1800", str(path))
    return path


@pytest.fixture(scope="session")
def music60(tmp_path_factory):
    """60 с синтетической «музыки» (тон с тремоло) — не речь, повод для фильтра галлюцинаций."""
    path = tmp_path_factory.mktemp("fixtures") / "music60.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=60", "-af", "tremolo=f=2",
            "-ar", "16000", "-ac", "1", str(path))
    return path


@pytest.fixture(scope="session")
def speech_silence_speech(tmp_path_factory, speech_ru_10s):
    """Речь 10 с → тишина 40 с → та же речь 10 с, 16 кГц моно."""
    path = tmp_path_factory.mktemp("fixtures") / "speech_silence_speech.wav"
    _ffmpeg("-i", str(speech_ru_10s), "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-filter_complex",
            "[1:a]atrim=0:40[sil];[0:a][sil][0:a]concat=n=3:v=0:a=1[out]",
            "-map", "[out]", str(path))
    return path


@pytest.fixture(scope="session")
def noaudio_mp4(tmp_path_factory):
    """Видео без звуковой дорожки вообще."""
    path = tmp_path_factory.mktemp("fixtures") / "noaudio.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=d=3:s=160x120", "-an", str(path))
    return path


@pytest.fixture(scope="session")
def broken_mp4(tmp_path_factory):
    """Первые 10 КБ настоящего mp4 — контейнер обрублен, moov atom (в хвосте) потерян."""
    tmp_dir = tmp_path_factory.mktemp("fixtures")
    full = tmp_dir / "full.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=d=2:s=160x120", "-f", "lavfi", "-i", "sine=f=440:d=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(full))
    path = tmp_dir / "broken.mp4"
    with open(full, "rb") as f_in, open(path, "wb") as f_out:
        f_out.write(f_in.read(10240))
    return path


@pytest.fixture(scope="session")
def quiet_wav(tmp_path_factory, speech_ru_10s):
    """Та же речь, усиление −32 дБ."""
    path = tmp_path_factory.mktemp("fixtures") / "quiet.wav"
    _ffmpeg("-i", str(speech_ru_10s), "-af", "volume=-32dB", str(path))
    return path


@pytest.fixture(scope="session")
def phone8k_wav(tmp_path_factory, speech_ru_10s):
    """Та же речь на телефонной частоте дискретизации 8 кГц."""
    path = tmp_path_factory.mktemp("fixtures") / "phone8k.wav"
    _ffmpeg("-i", str(speech_ru_10s), "-ar", "8000", str(path))
    return path


@pytest.fixture(scope="session")
def stereo_call_wav(tmp_path_factory, speech_ru_10s):
    """Голос слева 0–10 с, голос справа 12–22 с — имитация звонка с раздельными каналами."""
    path = tmp_path_factory.mktemp("fixtures") / "stereo_call.wav"
    _ffmpeg("-i", str(speech_ru_10s), "-i", str(speech_ru_10s), "-filter_complex",
            "[0:a]apad=whole_dur=22[left];[1:a]adelay=12000,apad=whole_dur=22[right];"
            "[left][right]amerge=inputs=2[out]",
            "-map", "[out]", "-ac", "2", "-ar", "16000", str(path))
    return path


@pytest.fixture(scope="session")
def stereo_same_wav(tmp_path_factory, speech_ru_10s):
    """Та же речь одинаково в обоих каналах — обычная стереозапись одним микрофоном,
    не звонок с раздельными каналами. На ней media.stereo_turns обязан отказаться
    (взаимоисключаемость каналов около нуля — они звучат одновременно)."""
    path = tmp_path_factory.mktemp("fixtures") / "stereo_same.wav"
    _ffmpeg("-i", str(speech_ru_10s), "-af", "pan=stereo|c0=c0|c1=c0", str(path))
    return path


@pytest.fixture(scope="session")
def stereo_echo_wav(tmp_path_factory, speech_ru_10s):
    """Слева речь 0–10 с, справа её же тихая копия 11–21 с: так выглядит эхо
    комнаты или спикерфона, попавшее ровно в паузу. Каналы звучат строго по
    очереди — по одной этой примете эхо неотличимо от второго собеседника, и
    отличает его только громкость (В6)."""
    path = tmp_path_factory.mktemp("fixtures") / "stereo_echo.wav"
    _ffmpeg("-i", str(speech_ru_10s), "-i", str(speech_ru_10s), "-filter_complex",
            "[0:a]apad=whole_dur=22[left];"
            "[1:a]adelay=11000,volume=-18dB,apad=whole_dur=22[right];"
            "[left][right]amerge=inputs=2[out]",
            "-map", "[out]", "-ac", "2", "-ar", "16000", str(path))
    return path


@pytest.fixture(scope="session")
def multitrack_mkv(tmp_path_factory):
    """Видео с двумя независимыми аудиодорожками (как в записях OBS)."""
    path = tmp_path_factory.mktemp("fixtures") / "multitrack.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=d=5:s=160x120",
            "-f", "lavfi", "-i", "sine=f=440:d=5",
            "-f", "lavfi", "-i", "sine=f=880:d=5",
            "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path))
    return path
