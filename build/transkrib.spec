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
            "huggingface_hub", "onnxruntime", "webview", "sherpa_onnx", "psutil"]:
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
    # setuptools/pkg_resources исключены намеренно: ни одна наша зависимость их в
    # рантайме не импортирует, а рантайм-хук PyInstaller pyi_rth_pkgres падает на
    # setuptools >= 81 («module 'pkg_resources' has no attribute 'NullProvider'») —
    # на этом 02.09.2026 разом легли все четыре сборки в CI, когда раннер подтянул 84.
    excludes=["torch", "torchaudio", "tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide6", "gi",
              "setuptools", "pkg_resources"],
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
# Второй exe с тем же кодом, но с консолью: пользователи и CI получают вывод
# --cli/--check-env без Start-Process+файла. Общий _internal, +1-2 МБ к сборке.
exe_console = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Transkrib-console",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    icon=None,
)
# exe (windowed) — последним: BUNDLE ниже наследует console-флаг от последнего
# EXE в COLLECT, а windowed-режим нужен для normal .app-поведения (без него
# BUNDLE поставит LSBackgroundOnly=True и окно не будет показываться в Dock).
coll = COLLECT(exe_console, exe, a.binaries, a.datas, strip=False, upx=False, name="Transkrib")

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
