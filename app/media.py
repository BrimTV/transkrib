"""Всё, что делает ffmpeg: извлечение звука, конвертация, поиск пауз.

ffmpeg берём из imageio-ffmpeg — pip-пакет с готовым статическим бинарником под
Windows x64 и macOS arm64, поэтому ничего отдельно качать и класть в PATH не надо.
В PyInstaller-сборке бинарник лежит внутри пакета, get_ffmpeg_exe() его находит.
"""
import os
import re
import subprocess
import sys
import wave

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


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def is_media(path):
    return os.path.splitext(path)[1].lower() in AUDIO_EXT | VIDEO_EXT


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXT


def _run(args, timeout=None, on_progress=None, total_sec=None):
    """Запустить ffmpeg, по желанию отдавая прогресс (0..1) из его stderr."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-y"] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", **_NO_WINDOW)
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
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg: " + "".join(tail[-8:]).strip())


def probe_duration(path):
    """Длительность в секундах через ffmpeg -i (ffprobe в imageio-ffmpeg нет)."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", **_NO_WINDOW)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def extract_wav(src, dst, on_progress=None):
    """Любой медиафайл → 16 кГц моно WAV: именно это ест whisper."""
    total = probe_duration(src)
    _run(["-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst],
         on_progress=on_progress, total_sec=total)
    return wav_duration(dst)


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def convert(src, dst, fmt, on_progress=None):
    """Конвертация по пресету. dst уже должен иметь нужное расширение."""
    if fmt not in CONVERT_PRESETS:
        raise ValueError(f"неизвестный формат: {fmt}")
    total = probe_duration(src)
    _run(["-i", src] + CONVERT_PRESETS[fmt] + [dst], on_progress=on_progress, total_sec=total)


def extract_audio(src, dst, fmt="mp3", on_progress=None):
    """Вытащить звуковую дорожку из видео. fmt: mp3 / wav / m4a / flac / ogg."""
    convert(src, dst, fmt, on_progress)


def silence_midpoints(path, noise_db="-30dB", min_sec="0.6"):
    """Середины пауз, по возрастанию. Пусто при сбое — вызывающий режет по таймеру.
    Перенесено из VK-Vibe unpack/transcribe.py."""
    try:
        proc = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-nostats", "-i", path, "-af",
             f"silencedetect=noise={noise_db}:d={min_sec}", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600, **_NO_WINDOW)
    except Exception:
        return []
    log = proc.stderr or ""
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
