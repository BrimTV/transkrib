"""Сборка результата в txt / srt / vtt / md из списка сегментов {start, end, text}."""


def _ts(sec, sep=","):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _short(sec):
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def to_txt(segments, paragraph_gap=2.5):
    """Сплошной текст. Новый абзац — когда пауза между сегментами больше paragraph_gap."""
    out, cur, prev_end = [], [], None
    for s in segments:
        if prev_end is not None and s["start"] - prev_end > paragraph_gap and cur:
            out.append(" ".join(cur)); cur = []
        cur.append(s["text"].strip())
        prev_end = s["end"]
    if cur:
        out.append(" ".join(cur))
    return "\n\n".join(out) + "\n"


def to_srt(segments):
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{_ts(s['start'])} --> {_ts(s['end'])}", s["text"].strip(), ""]
    return "\n".join(lines)


def to_vtt(segments):
    lines = ["WEBVTT", ""]
    for s in segments:
        lines += [f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}", s["text"].strip(), ""]
    return "\n".join(lines)


def to_md(segments, title=None):
    lines = [f"# {title}", ""] if title else []
    for s in segments:
        lines.append(f"**[{_short(s['start'])}]** {s['text'].strip()}  ")
    return "\n".join(lines) + "\n"


FORMATS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "md": to_md}


def render(segments, fmt, title=None):
    if fmt == "md":
        return to_md(segments, title)
    return FORMATS[fmt](segments)
