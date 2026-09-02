"""Окно приложения: pywebview + мост Python↔JS. Никаких портов и серверов —
JS зовёт методы Api напрямую, Python толкает события через evaluate_js."""
import json
import os
import subprocess
import sys
import threading
import time
import traceback

from . import __version__, engine, export, media


def _ui_path():
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "app", "ui", "index.html")


def _settings_path():
    return os.path.join(engine.data_dir(), "settings.json")


DEFAULT_SETTINGS = dict(model=engine.DEFAULT_MODEL, language="ru", prefer_gpu=True,
                        autosave=True, autosave_formats=["txt", "srt"],
                        ts_mode="paragraph", ts_interval=60, ts_style="brackets_short",
                        diarize=False, num_speakers=0)


def load_settings():
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(s):
    with open(_settings_path(), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


class Api:
    def __init__(self):
        self.window = None
        self._cancel = None
        self._busy = False
        self._last_push = {}

    # ── события в окно ───────────────────────────────────────────────────
    def _emit(self, event, job_id=None):
        # download/extract/progress летят часто — не чаще 10 раз в секунду на тип
        t = event.get("type")
        if t in ("download", "extract", "progress", "diar_progress"):
            now = time.time()
            if now - self._last_push.get(t, 0) < 0.1:
                return
            self._last_push[t] = now
        if job_id is not None:
            event["job"] = job_id
        payload = json.dumps(event, ensure_ascii=False)
        try:
            self.window.evaluate_js(f"window.onEngineEvent({payload})")
        except Exception:
            pass

    # ── справочная информация ────────────────────────────────────────────
    def info(self):
        hw = engine.hardware_info()
        models = [dict(key="auto", label=engine.auto_label(), size_mb=0, cached=True,
                       resolved=engine.resolve_model("auto", hw["backend"]))]
        for key, m in engine.MODELS.items():
            models.append(dict(key=key, label=m["label"], size_mb=m["size_mb"], resolved=key,
                               cached=engine.model_is_cached(key, hw["backend"])))
        from . import diarize
        return dict(version=__version__, hardware=hw, backend_label=engine.backend_label(hw["backend"]),
                    diarize_available=diarize.available(), diarize_models=bool(diarize.model_paths()),
                    models=models, languages=engine.LANGUAGES, settings=load_settings(),
                    models_dir=engine.models_dir(), convert_formats=list(media.CONVERT_PRESETS))

    def set_settings(self, s):
        cur = load_settings()
        cur.update(s or {})
        save_settings(cur)
        return cur

    # ── диалоги ──────────────────────────────────────────────────────────
    def pick_files(self):
        import webview
        exts = sorted(e.lstrip(".") for e in media.AUDIO_EXT | media.VIDEO_EXT)
        types = ("Аудио и видео (" + ";".join("*." + e for e in exts) + ")", "Все файлы (*.*)")
        res = self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True, file_types=types)
        return [p for p in (res or []) if os.path.isfile(p)]

    def open_log(self):
        self.open_folder(engine.data_dir())

    def open_folder(self, path):
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if sys.platform == "win32":
            os.startfile(folder)  # noqa
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    # ── расшифровка ──────────────────────────────────────────────────────
    def start(self, job_id, path, model, language, prefer_gpu, diarize=False, num_speakers=0):
        if self._busy:
            return dict(ok=False, error="уже идёт задача")
        if not os.path.isfile(path):
            return dict(ok=False, error="файл не найден")
        self._busy = True
        self._cancel = threading.Event()

        def run():
            try:
                engine.transcribe_file(path, model, language,
                                       lambda e: self._emit(e, job_id), self._cancel, prefer_gpu,
                                       diarize=bool(diarize), num_speakers=int(num_speakers or 0))
            except Exception as e:
                self._emit(dict(type="error", msg=f"{e}\n{traceback.format_exc()[-400:]}"), job_id)
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()
        return dict(ok=True)

    def cancel(self):
        if self._cancel:
            self._cancel.set()
        return True

    @staticmethod
    def _opts():
        s = load_settings()
        return dict(ts_mode=s.get("ts_mode"), ts_interval=s.get("ts_interval"), ts_style=s.get("ts_style"))

    def render(self, segments, fmt, title=None):
        return export.render(segments, fmt, title, self._opts())

    def autosave(self, path, segments, formats):
        """Положить результат рядом с исходником. Возвращает список записанных файлов."""
        base = os.path.splitext(path)[0]
        written = []
        for fmt in formats:
            out = f"{base}.{fmt}"
            with open(out, "w", encoding="utf-8") as f:
                f.write(export.render(segments, fmt, os.path.basename(path), self._opts()))
            written.append(out)
        return written

    def save_as(self, segments, fmt, suggested):
        import webview
        res = self.window.create_file_dialog(webview.SAVE_DIALOG, save_filename=f"{suggested}.{fmt}")
        if not res:
            return None
        out = res[0] if isinstance(res, (list, tuple)) else res
        with open(out, "w", encoding="utf-8") as f:
            f.write(export.render(segments, fmt, suggested, self._opts()))
        return out

    def memory(self):
        return dict(available_gb=engine.memory_available_gb(), loaded=engine.models_loaded())

    # ── фоновый уход за памятью ──────────────────────────────────────────
    def start_housekeeping(self):
        def loop():
            while True:
                time.sleep(30)
                try:
                    reason = engine.maybe_unload(self._busy)
                    if reason:
                        self._emit(dict(type="info", msg=reason.capitalize()))
                except Exception:
                    pass
        threading.Thread(target=loop, daemon=True).start()

    # ── конвертер ────────────────────────────────────────────────────────
    def convert(self, job_id, path, fmt):
        if not os.path.isfile(path):
            return dict(ok=False, error="файл не найден")
        base, ext = os.path.splitext(path)
        out = f"{base}.{fmt}"
        if ext.lower() == f".{fmt}":
            out = f"{base}_converted.{fmt}"
        n = 1
        while os.path.exists(out):
            out = f"{base}_{n}.{fmt}"; n += 1

        def run():
            try:
                self._emit(dict(type="convert_progress", progress=0.0), job_id)
                media.convert(path, out, fmt,
                              on_progress=lambda p: self._emit(dict(type="convert_progress", progress=p), job_id))
                self._emit(dict(type="convert_done", out=out), job_id)
            except Exception as e:
                self._emit(dict(type="convert_error", msg=str(e)), job_id)

        threading.Thread(target=run, daemon=True).start()
        return dict(ok=True, out=out)


def main():
    if "--selftest" in sys.argv:
        # Для CI: без окна, результат ещё и в файл — у windowed-сборки нет stdout.
        log_path = os.path.join(os.getcwd(), "selftest.log")
        with open(log_path, "w", encoding="utf-8") as log:
            class Tee:
                def write(self, s):
                    log.write(s); log.flush()
                    try:
                        sys.__stdout__.write(s); sys.__stdout__.flush()
                    except Exception:
                        pass
                def flush(self):
                    pass
            sys.stdout = sys.stderr = Tee()
            sys.exit(engine.selftest())

    if "--cli" in sys.argv:
        # Transkrib --cli файл [--speakers] [--model small] [--lang ru] — без окна, события в stdout
        args = sys.argv[sys.argv.index("--cli") + 1:]
        path = next((a for a in args if not a.startswith("--")), None)
        if not path:
            sys.exit("укажите файл")
        opt = lambda k, d: args[args.index(k) + 1] if k in args else d  # noqa: E731
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        segs = []

        def show(e):
            if e["type"] == "segment":
                segs.append(e); print(f"[{e['start']:7.1f}] {e['text']}", flush=True)
            elif e["type"] == "speakers":
                from . import diarize
                diarize.assign(segs, e["turns"]); print(f"говорящих: {e['count']}", flush=True)
            elif e["type"] in ("stage", "done", "error", "cancelled"):
                print("::", e["msg"], flush=True)
        engine.transcribe_file(path, opt("--model", "auto"), opt("--lang", "ru"), show,
                               diarize="--speakers" in args)
        print(export.render(segs, "txt", opts=dict(ts_mode="paragraph")))
        sys.exit(0)

    import webview
    api = Api()
    window = webview.create_window(f"Transkrib {__version__}", _ui_path(), js_api=api,
                                   width=1100, height=740, min_size=(820, 560),
                                   background_color="#111418")
    api.window = window
    api.start_housekeeping()
    webview.start(private_mode=False, debug="--debug" in sys.argv)


if __name__ == "__main__":
    main()
