"""Скачать модель в bundled_models/ для вшивания в сборку.

Использование: python build/fetch_model.py small [lite]
Кладёт faster-whisper-версию в bundled_models/<key>/fw, на macOS arm64 ещё и
mlx-версию в bundled_models/<key>/mlx. Пишет bundled_models/variant.json с ключом
модели — движок читает его как «встроенная модель по умолчанию» — и блоком build
(версия, короткий хеш коммита, дата, вариант поставки lite/medium) для бейджа в
шапке UI и диагностики по логу пользователя.
"""
import datetime
import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import __version__, engine  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

key = sys.argv[1] if len(sys.argv) > 1 else "small"
if key not in engine.MODELS:
    sys.exit(f"неизвестная модель {key}; есть: {list(engine.MODELS)}")
# Вариант поставки (lite/medium) — то, что видит пользователь в имени архива.
# Отдельно от key, потому что у medium он случайно совпадает с ключом модели
# ("medium"), а у lite — нет (key="small").
variant = sys.argv[2] if len(sys.argv) > 2 else key

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bundled_models")
# На Apple Silicon вшиваем только MLX-версию: она работает на любом таком маке,
# а CPU-фолбэк (faster-whisper) в редком случае докачает свою копию сам.
# Иначе пакет вдвое тяжелее (две копии одной модели).
if sys.platform == "darwin" and platform.machine() == "arm64":
    targets = [("mlx", engine.MODELS[key]["mlx"], None)]
else:
    targets = [("fw", engine.MODELS[key]["fw"], engine._FW_PATTERNS)]

for kind, repo, patterns in targets:
    dst = os.path.join(root, key, kind)
    print(f"[fetch_model] {repo} -> {dst}", flush=True)
    snapshot_download(repo, local_dir=dst, allow_patterns=patterns)
    # служебный кэш hf внутри local_dir в сборке не нужен
    import shutil
    shutil.rmtree(os.path.join(dst, ".cache"), ignore_errors=True)
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dst) for f in fs)
    print(f"[fetch_model] {kind}: {size / 1048576:.0f} МБ", flush=True)

from app import diarize  # noqa: E402
diarize.fetch_models(os.path.join(root, "diarization"), print)
print("[fetch_model] diarization: ok")

def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=os.path.dirname(root), capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


build_info = {
    "version": __version__,
    "commit": _git_commit(),
    "date": datetime.date.today().isoformat(),
    "variant": variant,
}
with open(os.path.join(root, "variant.json"), "w", encoding="utf-8") as f:
    json.dump({"model": key, "build": build_info}, f, ensure_ascii=False)
print("[fetch_model] variant.json:", key, build_info)
