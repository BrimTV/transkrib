"""Сборка результата в txt / srt / vtt / md из списка сегментов {start, end, text}.

Таймкоды в txt/md настраиваются (opts):
  ts_mode:     none | segment | paragraph | interval
  ts_interval: секунд между метками в режиме interval
  ts_style:    brackets_short [0:00] | brackets_long [00:00:00] | plain_short 0:00 | plain_long 00:00:00
"""

DEFAULT_OPTS = dict(ts_mode="paragraph", ts_interval=60, ts_style="brackets_short", paragraph_gap=2.5)


def _ts(sec, sep=","):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def stamp(sec, style="brackets_short"):
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if style.endswith("_long"):
        t = f"{h:02d}:{m:02d}:{s:02d}"
    else:
        t = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    return f"[{t}]" if style.startswith("brackets") else t


def _speaker(s):
    v = s.get("speaker")
    if v in (None, ""):
        return None
    return v if isinstance(v, str) else f"Говорящий {v}"


def _blocks(segments, opts):
    """Разбить сегменты на блоки [(start, [texts], speaker)] по выбранному режиму.
    Смена говорящего всегда начинает новый блок."""
    mode = opts.get("ts_mode", "paragraph")
    gap = float(opts.get("paragraph_gap", 2.5))
    step = max(5, int(opts.get("ts_interval", 60)))
    blocks, cur, prev_end, next_mark, prev_spk = [], None, None, 0, None
    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        spk = _speaker(s)
        new = cur is None or spk != prev_spk
        if mode == "segment":
            new = True
        elif mode == "interval":
            if s["start"] >= next_mark:
                new = True
                next_mark = (int(s["start"]) // step + 1) * step
        else:  # paragraph / none — по паузам между репликами
            if prev_end is not None and s["start"] - prev_end > gap:
                new = True
        if new:
            cur = [s["start"], [], spk]
            blocks.append(cur)
        cur[1].append(text)
        prev_end, prev_spk = s["end"], spk
    return blocks


def to_txt(segments, opts=None):
    o = {**DEFAULT_OPTS, **(opts or {})}
    out = []
    for start, texts, spk in _blocks(segments, o):
        line = " ".join(texts)
        if spk:
            line = f"{spk}: {line}"
        if o["ts_mode"] != "none":
            line = f"{stamp(start, o['ts_style'])} {line}"
        out.append(line)
    sep = "\n" if o["ts_mode"] == "segment" else "\n\n"
    return sep.join(out) + "\n"


def to_md(segments, title=None, opts=None):
    o = {**DEFAULT_OPTS, **(opts or {})}
    lines = [f"# {title}", ""] if title else []
    for start, texts, spk in _blocks(segments, o):
        line = " ".join(texts)
        if spk:
            line = f"**{spk}:** {line}"
        if o["ts_mode"] != "none":
            line = f"**{stamp(start, o['ts_style'])}** {line}"
        lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _cue_text(s):
    spk = _speaker(s)
    return f"{spk}: {s['text'].strip()}" if spk else s["text"].strip()


def to_srt(segments):
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{_ts(s['start'])} --> {_ts(s['end'])}", _cue_text(s), ""]
    return "\n".join(lines)


def to_vtt(segments):
    lines = ["WEBVTT", ""]
    for s in segments:
        lines += [f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}", _cue_text(s), ""]
    return "\n".join(lines)


def render(segments, fmt, title=None, opts=None):
    if fmt == "txt":
        return to_txt(segments, opts)
    if fmt == "md":
        return to_md(segments, title, opts)
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "vtt":
        return to_vtt(segments)
    raise ValueError(fmt)
