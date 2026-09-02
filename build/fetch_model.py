"""Скачать модель в bundled_models/ для вшивания в сборку.

Использование: python build/fetch_model.py small
Кладёт faster-whisper-версию в bundled_models/<key>/fw, на macOS arm64 ещё и
mlx-версию в bundled_models/<key>/mlx. Пишет bundled_models/variant.json с ключом
модели — движок читает его как «встроенная модель по умолчанию».
"""
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import engine  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

key = sys.argv[1] if len(sys.argv) > 1 else "small"
if key not in engine.MODELS:
    sys.exit(f"неизвестная модель {key}; есть: {list(engine.MODELS)}")

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

with open(os.path.join(root, "variant.json"), "w", encoding="utf-8") as f:
    json.dump({"model": key}, f)
print("[fetch_model] variant.json:", key)
