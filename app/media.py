"""Всё, что делает ffmpeg: извлечение звука, конвертация, поиск пауз.

ffmpeg берём из imageio-ffmpeg — pip-пакет с готовым статическим бинарником под
Windows x64 и macOS arm64, поэтому ничего отдельно качать и класть в PATH не надо.
В PyInstaller-сборке бинарник лежит внутри пакета, get_ffmpeg_exe() его находит.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave

from .engine import UserError

# Консольное окно ffmpeg не должно всплывать поверх приложения на Windows.
_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform == "win32" else {}

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac", ".wma", ".aiff", ".aif", ".amr"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg", ".ts", ".3gp"}

# Форматы для вкладки «Конвертер»: расширение → аргументы кодека.
CONVERT_PRESETS = {
    "mp3":  ["-vn", "-c:a", "libmp3lame", "-q:a", "2"],
    "wav":  ["-vn", "-c:a", "pcm_s16le"],
    "m4a":  ["-vn", "-c:a", "aac", "-b:a", "192k"],
    "flac": ["-vn", "-c:a", "flac"],
    "ogg":  ["-vn", "-c:a", "libvorbis", "-q:a", "6"],
    "mp4":  ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"],
    "mkv":  ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "160k"],
    "webm": ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-c:a", "libopus", "-b:a", "128k"],
    "gif":  ["-vf", "fps=12,scale=480:-1:flags=lanczos", "-an"],
}

# Подстрока в stderr ffmpeg → понятный текст для человека. Нет совпадения — отдаём
# хвост stderr как раньше (его дальше в engine.transcribe_file завернут в
# «техническая деталь»). Порядок важен: проверяем по очереди, первое совпадение побеждает.
FFMPEG_HINTS = [
    # На ffmpeg 7.1 (наш бинарник из imageio-ffmpeg) при отсутствии звука в файле
    # -map 0:a:0 не резолвится вообще, и подстрока в stderr — «Stream map ''»,
    # а не «Stream map '0:a:0'»: проверено на noaudio.mp4 (см. tests/test_media.py).
    ("matches no streams", "В этом видео нет звуковой дорожки"),
    ("Invalid data found", "Не удалось прочитать файл как аудио или видео: он повреждён или недокачан"),
    ("moov atom not found", "Не удалось прочитать файл как аудио или видео: он повреждён или недокачан"),
    ("Permission denied", "Нет доступа к файлу (права или файл открыт другой программой)"),
    ("No space left", "На диске нет места для временного файла"),
]


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def is_media(path):
    return os.path.splitext(path)[1].lower() in AUDIO_EXT | VIDEO_EXT


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXT


def check_readable(path, duration_sec=None, tmp_dir=None):
    """Проверить файл перед обработкой и поднять понятную UserError, если что-то не так.
    Без duration_sec проверяются только сам файл и облачные плейсхолдеры (для быстрой
    проверки «до старта задачи», когда длительность ещё не известна); с duration_sec —
    ещё и место на диске под временный WAV (нужно вызывать уже после probe_info)."""
    if not os.path.exists(path):
        raise UserError(f"Файл не найден: {os.path.basename(path)}. Возможно, он перемещён или переименован")
    size = os.path.getsize(path)
    if size == 0:
        raise UserError("Файл пустой (0 байт)")
    if sys.platform == "win32":
        OFFLINE, RECALL_ON_DATA_ACCESS = 0x1000, 0x400000
        attrs = getattr(os.stat(path), "st_file_attributes", 0)
        if attrs & (OFFLINE | RECALL_ON_DATA_ACCESS):
            raise UserError("Файл хранится только в облаке OneDrive. Откройте папку, "
                             "выберите «Всегда сохранять на этом устройстве» и повторите")
    elif sys.platform == "darwin":
        d, name = os.path.split(path)
        if os.path.exists(os.path.join(d, f".{name}.icloud")):
            raise UserError("Файл ещё не скачан из iCloud")
    if duration_sec:
        need = duration_sec * 32000 * 1.2
        free = shutil.disk_usage(tmp_dir or tempfile.gettempdir()).free
        if free < need:
            raise UserError(
                f"На диске нет места для временного файла: нужно ~{need / 1e6:.0f} МБ, "
                f"свободно {free / 1e6:.0f} МБ")


def _watch_cancel(proc, cancel):
    """Фоновый поток-наблюдатель: при взведённом cancel убивает процесс.
    Нужен отдельно от основного цикла разбора stderr — чтение оттуда блокирующее,
    и если ffmpeg надолго замолкает (например, между муксом дорожек), проверить
    флаг в этом же потоке негде."""
    killed = threading.Event()
    if cancel is None:
        return None, killed

    def _watch():
        while proc.poll() is None:
            if cancel.wait(0.2):
                killed.set()
                try:
                    proc.kill()
                except OSError:
                    pass
                return
    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return t, killed


def _run(args, timeout=None, on_progress=None, total_sec=None, cancel=None):
    """Запустить ffmpeg, по желанию отдавая прогресс (0..1) из его stderr и позволяя
    отменить через cancel (threading.Event)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-y"] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", **_NO_WINDOW)
    watcher, killed = _watch_cancel(proc, cancel)
    tail = []
    for line in proc.stderr:
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        if on_progress and total_sec:
            m = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", line)
            if m:
                cur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                on_progress(min(cur / total_sec, 1.0))
    proc.wait(timeout=timeout)
    if watcher:
        watcher.join(timeout=1.0)
    if killed.is_set():
        raise InterruptedError("отменено")
    if proc.returncode != 0:
        text = "".join(tail)
        for needle, human in FFMPEG_HINTS:
            if needle in text:
                raise UserError(human)
        raise RuntimeError("ffmpeg: " + "".join(tail[-8:]).strip())


def probe_info(path):
    """Длительность в секундах, число аудиодорожек и частота дискретизации
    первой дорожки через ffmpeg -i (ffprobe в imageio-ffmpeg нет).
    sample_rate — только для диагностики (Б6): 8 кГц телефония ничего не
    меняет в поведении, но полезна в логе при жалобах на качество."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", **_NO_WINDOW)
    err = proc.stderr or ""
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    audio_tracks = len(re.findall(r"Stream #\d+:\d+(?:\(\w+\))?:\s*Audio:", err))
    sr_m = re.search(r"Audio:[^\n]*?(\d+)\s*Hz", err)
    sample_rate = int(sr_m.group(1)) if sr_m else None
    return dict(duration=duration, audio_tracks=audio_tracks, sample_rate=sample_rate)


def probe_duration(path):
    """Длительность в секундах. Обёртка над probe_info для мест, которым не нужно
    число дорожек (convert, cut_points)."""
    return probe_info(path)["duration"]


def _log(msg):
    """engine.py импортирует media снизу своего файла (см. app/engine.py) — на
    верхнем уровне этого модуля engine.log ещё не существует, поэтому берём
    его лениво, в момент вызова, а не через обычный import наверху файла."""
    try:
        from . import engine
        engine.log(msg)
    except Exception:
        pass


def _measure_volume(path):
    """mean_volume/max_volume через ffmpeg -af volumedetect: только декодирование
    без перекодирования, поэтому быстро даже на длинной записи (Б6)."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path, "-af", "volumedetect",
                          "-f", "null", "-"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **_NO_WINDOW)
    err = proc.stderr or ""
    mean_m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", err)
    max_m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", err)
    mean = float(mean_m.group(1)) if mean_m else None
    peak = float(max_m.group(1)) if max_m else None
    return mean, peak


def extract_wav(src, dst, on_progress=None, cancel=None):
    """Любой медиафайл → 16 кГц моно WAV: именно это ест whisper.
    -map 0:a:0 — берём только первую аудиодорожку: без этого в файлах с несколькими
    дорожками (например, из OBS) ffmpeg может сам выбрать не ту, а с явной картой
    отсутствие звука даёт стабильную строку в stderr (см. FFMPEG_HINTS).

    Тихие записи (Б6): после VAD (Silero) тихая речь рискует потеряться в
    порогах. Меряем средний уровень и, если он ниже −30 дБ, переизвлекаем со
    статическим усилением (не компрессором — компрессор поднял бы и шум в
    паузах вместе с речью)."""
    info = probe_info(src)
    check_readable(src, duration_sec=info["duration"], tmp_dir=os.path.dirname(dst) or None)
    args = ["-i", src, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst]
    _run(args, on_progress=on_progress, total_sec=info["duration"], cancel=cancel)
    # cancel — при взведённом флаге _run уже бросил InterruptedError выше, сюда не дойдём
    mean, peak = _measure_volume(dst)
    if mean is not None and mean < -30:
        gain = min(-20 - mean, -1 - peak) if peak is not None else -20 - mean
        _log(f"тихая запись: mean_volume={mean:.1f}dB — переизвлекаю с усилением {gain:.1f}dB")
        _run(["-i", src, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
              "-af", f"volume={gain:.2f}dB", "-c:a", "pcm_s16le", dst],
             total_sec=info["duration"], cancel=cancel)
    if on_progress:
        on_progress(1.0)  # длительность контейнера бывает больше звука — добиваем до 100%
    _log(f"звук: {info['sample_rate']} Гц, дорожек {info['audio_tracks']}"
        + (" (телефония)" if (info["sample_rate"] or 0) and info["sample_rate"] <= 8000 else ""))
    return wav_duration(dst), info["audio_tracks"]


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def convert(src, dst, fmt, on_progress=None, cancel=None):
    """Конвертация по пресету. dst уже должен иметь нужное расширение."""
    if fmt not in CONVERT_PRESETS:
        raise ValueError(f"неизвестный формат: {fmt}")
    total = probe_duration(src)
    _run(["-i", src] + CONVERT_PRESETS[fmt] + [dst], on_progress=on_progress, total_sec=total, cancel=cancel)


def extract_audio(src, dst, fmt="mp3", on_progress=None, cancel=None):
    """Вытащить звуковую дорожку из видео. fmt: mp3 / wav / m4a / flac / ogg."""
    convert(src, dst, fmt, on_progress, cancel)


def silence_midpoints(path, noise_db="-30dB", min_sec="0.6", cancel=None):
    """Середины пауз, по возрастанию. Пусто при сбое запуска — вызывающий режет по
    таймеру. Перенесено из VK-Vibe unpack/transcribe.py."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostats", "-i", path, "-af",
           f"silencedetect=noise={noise_db}:d={min_sec}", "-f", "null", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", **_NO_WINDOW)
    except Exception:
        return []
    watcher, killed = _watch_cancel(proc, cancel)
    try:
        _, err = proc.communicate(timeout=600)
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass
        err = ""
    if watcher:
        watcher.join(timeout=1.0)
    if killed.is_set():
        raise InterruptedError("отменено")
    log = err or ""
    starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", log)]
    return sorted((a + b) / 2 for a, b in zip(starts, ends))


def cut_points(dur, silences, target=120.0, max_len=150.0):
    """Границы кусков: к каждой отметке target ищем ближайшую паузу, не выходя
    за max_len. Нет паузы — жёсткий рез. Куски короче, чем в VK-Vibe (там 240/300),
    потому что тут важнее живой прогресс на экране, чем минимум швов."""
    pts, cur = [], 0.0
    while dur - cur > max_len:
        lo, hi = cur + target * 0.5, cur + max_len
        cand = [s for s in silences if lo <= s <= hi]
        nxt = min(cand, key=lambda s: abs(s - (cur + target))) if cand else cur + max_len
        pts.append(nxt)
        cur = nxt
    return pts


def slice_wav(src, dst, start, end):
    """Вырезать [start, end) из WAV без перекодирования потерь."""
    _run(["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src, "-ac", "1", "-ar", "16000",
          "-c:a", "pcm_s16le", dst])


def wav_slice(path, start, end):
    """Кусок WAV → float32 массив в диапазоне [-1, 1], без ffmpeg (не тратим
    процесс на маленький кусок). Перенесено из engine._wav_slice (Б2): нужно
    и здесь для чтения VAD блоками, и в engine — для нарезки кусков перед
    моделью (Б2/Б3)."""
    import numpy as np
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        w.setpos(int(start * sr))
        n = int((end - start) * sr)
        data = w.readframes(n)
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def speech_regions(wav, total_sec, on_progress=None, cancel=None, block=600.0):
    """Интервалы речи в WAV через Silero VAD (faster_whisper.vad) — модель
    уже попадает в сборку через collect_all("faster_whisper") (см. Б2).

    Читаем блоками по `block` секунд (по умолчанию 10 минут), а не всю запись
    разом: на трёхчасовой записи держать весь звук во float32 в памяти сразу —
    лишние ≈650 МБ, которых на ноутбуке с 8 ГБ жалко. Регионы на границе блока
    не теряются: спад речи короче, чем блок, а на стыке двух блоков между ними
    склеиваем интервалы с промежутком меньше секунды (см. ниже).

    Возвращает список (start, end) в секундах записи, по возрастанию."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    with wave.open(wav, "rb") as w:
        sr = w.getframerate()
    opts = VadOptions(threshold=0.5, min_speech_duration_ms=250,
                      min_silence_duration_ms=500, speech_pad_ms=400)
    regions = []
    pos = 0.0
    while pos < total_sec:
        if cancel and cancel.is_set():
            raise InterruptedError("отменено")
        end = min(pos + block, total_sec)
        chunk = wav_slice(wav, pos, end)
        for r in get_speech_timestamps(chunk, opts, sampling_rate=sr):
            regions.append((pos + r["start"] / sr, pos + r["end"] / sr))
        if on_progress:
            on_progress(min(end / total_sec, 1.0))
        pos = end
    merged = []
    for a, b in regions:
        if merged and a - merged[-1][1] < 1.0:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def region_gaps(regions, min_gap=0.6):
    """Середины промежутков между соседними интервалами речи длиннее min_gap —
    подставляются в cut_points вместо ffmpeg-паузы: режем по настоящим
    промежуткам между репликами (VAD), а не там, где ffmpeg silencedetect
    считает тихо (там могла быть тихая музыка без пауз в самой речи)."""
    gaps = []
    for (_, end), (start, _) in zip(regions, regions[1:]):
        if start - end >= min_gap:
            gaps.append((end + start) / 2)
    return gaps
