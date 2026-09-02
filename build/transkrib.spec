# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: одна папка (onedir) с exe/приложением внутри.
Запуск: pyinstaller build/transkrib.spec  (из корня репозитория)."""
import os
import re
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

sys.path.insert(0, ROOT)
from app import __version__ as APP_VERSION  # noqa: E402 — версия в одном месте: app/__init__.py

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

ICON_ICO = os.path.join(ROOT, "build", "icon.ico")
ICON_ICNS = os.path.join(ROOT, "build", "icon.icns")
MANIFEST = os.path.join(ROOT, "build", "transkrib.manifest")


def _version_tuple(v):
    """"0.1.0" / "0.1.0+a1b2c3d" / "1.2.3" -> (1,2,3,0). Ресурс версии Windows
    принимает только 4 целых числа — хеш коммита и всё нечисловое отбрасываем."""
    base = v.split("+", 1)[0]
    nums = []
    for part in base.split(".")[:4]:
        m = re.match(r"\d+", part)
        nums.append(int(m.group()) if m else 0)
    nums += [0] * (4 - len(nums))
    return tuple(nums)


def _version_resource(internal_name, original_filename):
    # Exe без сведений о версии выглядит подозрительно и для людей, и для
    # антивирусов. StringTable "040904B0" = английский (US), кодировка Unicode —
    # стандартная комбинация для VarStruct('Translation', [0x409, 1200]) ниже.
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
    )
    vt = _version_tuple(APP_VERSION)
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=vt, prodvers=vt, OS=0x40004, fileType=0x1, subtype=0x0),
        kids=[
            StringFileInfo([
                StringTable("040904B0", [
                    StringStruct("CompanyName", "Transkrib"),
                    StringStruct("FileDescription", "Transkrib — расшифровка аудио и видео"),
                    StringStruct("FileVersion", APP_VERSION),
                    StringStruct("InternalName", internal_name),
                    StringStruct("LegalCopyright", "Transkrib"),
                    StringStruct("OriginalFilename", original_filename),
                    StringStruct("ProductName", "Transkrib"),
                    StringStruct("ProductVersion", APP_VERSION),
                ]),
            ]),
            VarFileInfo([VarStruct("Translation", [0x409, 1200])]),
        ],
    )


exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Transkrib",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=ICON_ICO if IS_WIN else None,
    manifest=MANIFEST if IS_WIN else None,
    version=_version_resource("Transkrib", "Transkrib.exe") if IS_WIN else None,
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
    icon=ICON_ICO if IS_WIN else None,
    manifest=MANIFEST if IS_WIN else None,
    version=_version_resource("Transkrib-console", "Transkrib-console.exe") if IS_WIN else None,
)
# exe (windowed) — последним: BUNDLE ниже наследует console-флаг от последнего
# EXE в COLLECT, а windowed-режим нужен для normal .app-поведения (без него
# BUNDLE поставит LSBackgroundOnly=True и окно не будет показываться в Dock).
coll = COLLECT(exe_console, exe, a.binaries, a.datas, strip=False, upx=False, name="Transkrib")

if IS_MAC:
    app = BUNDLE(
        coll,
        name="Transkrib.app",
        icon=ICON_ICNS,
        bundle_identifier="ru.transkrib.app",
        info_plist={
            "CFBundleShortVersionString": APP_VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
