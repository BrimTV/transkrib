# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: одна папка (onedir) с exe/приложением внутри.
Запуск: pyinstaller build/transkrib.spec  (из корня репозитория)."""
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

datas = [(os.path.join(ROOT, "app", "ui"), os.path.join("app", "ui"))]
# Встроенная модель: build/fetch_model.py <key> кладёт её в bundled_models/
BUNDLED = os.path.join(ROOT, "bundled_models")
if os.path.isdir(BUNDLED):
    datas.append((BUNDLED, "bundled_models"))
    print(f"[spec] встраиваю модели из {BUNDLED}")
else:
    print("[spec] bundled_models/ нет — сборка без встроенной модели (модель скачается при первом запуске)")
binaries, hiddenimports = [], []

# Нативные зависимости целиком: библиотеки, данные, скрытые импорты.
for pkg in ["ctranslate2", "faster_whisper", "av", "imageio_ffmpeg", "tokenizers",
            "huggingface_hub", "onnxruntime", "webview"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:
        print(f"[spec] пропускаю {pkg}: {e}")

if IS_MAC:
    for pkg in ["mlx", "mlx_whisper"]:
        try:
            d, b, h = collect_all(pkg)
            datas += d; binaries += b; hiddenimports += h
        except Exception as e:
            print(f"[spec] пропускаю {pkg}: {e}")

if IS_WIN:
    # DLL cublas/cudnn: кладём как данные, чтобы сохранить структуру nvidia/<lib>/bin,
    # которую engine._prepare_cuda_dlls добавляет в поиск DLL.
    for pkg in ["nvidia.cublas", "nvidia.cudnn"]:
        try:
            datas += collect_data_files(pkg, include_py_files=False, includes=["**/*.dll"])
        except Exception as e:
            print(f"[spec] пропускаю {pkg}: {e}")
    hiddenimports += ["clr_loader", "pythonnet", "webview.platforms.winforms", "webview.platforms.edgechromium"]
if IS_MAC:
    hiddenimports += ["webview.platforms.cocoa", "objc", "Foundation", "AppKit", "WebKit"]

a = Analysis(
    [os.path.join(ROOT, "run.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "torchaudio", "tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide6", "gi"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Transkrib",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Transkrib")

if IS_MAC:
    app = BUNDLE(
        coll,
        name="Transkrib.app",
        icon=None,
        bundle_identifier="ru.transkrib.app",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
