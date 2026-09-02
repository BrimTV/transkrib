# Transkrib

Локальная расшифровка аудио и видео. Без облака, без ключей, без нейросетевых API:
закинул файл, смотришь прогресс, текст появляется по мере распознавания.

- **Вход:** mp4, mkv, mov, avi, webm, mp3, wav, m4a, ogg, flac и другие.
- **Выход:** txt, srt, vtt, md. По умолчанию txt и srt сохраняются рядом с файлом.
- **Конвертер:** любой формат в mp3 / wav / m4a / flac / ogg / mp4 / mkv / webm / gif,
  в том числе вытащить звук из видео.
- **Железо:** Windows с NVIDIA считает на видеокарте (CUDA), Mac на Apple Silicon на GPU (MLX).
  Без видеокарты работает на процессоре. Переключение автоматическое, с запасным путём на CPU.
- **Две сборки, модель уже внутри, ничего не качается:**
  - `lite` — Whisper small, только процессор. Работает на любом ноутбуке, архив самый маленький.
  - `medium` — Whisper medium, точнее и в 2–3 раза медленнее. На Windows внутри ещё CUDA-библиотеки,
    так что с NVIDIA считает на видеокарте.
  В настройках можно догрузить large-v3-turbo (1.6 ГБ), это имеет смысл только с видеокартой.

Основано на логике расшифровки из VK-Vibe (`unpack/transcribe.py`): mlx-whisper на маке,
faster-whisper как CPU-путь, резка длинных записей по паузам через ffmpeg.

## Установка из сборки

Готовые архивы собирает GitHub Actions (вкладка Actions → последняя сборка → Artifacts,
или Releases для тегов `v*`).

**Windows.** Распаковать `Transkrib-lite-windows-x64.zip` (или `medium`), запустить `Transkrib\Transkrib.exe`.
Нужен Windows 10/11 с WebView2 (есть в системе по умолчанию). CUDA ставить не надо,
библиотеки внутри. При первом запуске SmartScreen может спросить: «Подробнее → Выполнить в любом случае».

**macOS (Apple Silicon).** Распаковать `Transkrib-lite-macos-arm64.zip` (или `medium`), перетащить `Transkrib.app`
в Программы. Приложение не подписано сертификатом разработчика, поэтому при первом запуске
правой кнопкой → Открыть. Если система пишет «повреждено», снять карантин:

```bash
xattr -cr /Applications/Transkrib.app
```

## Запуск из исходников

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt     # на Windows: requirements-win.txt
python run.py
```

Проверка движка без окна (скачает tiny-модель, прогонит синтетический звук):

```bash
python -m app.engine --selftest
```

## Сборка

```bash
python build/fetch_model.py small        # или medium: модель ляжет в bundled_models/
pyinstaller build/transkrib.spec --noconfirm --clean
```

Без `bundled_models/` сборка тоже соберётся, но модель скачается при первом запуске.

Результат в `dist/Transkrib` (Windows) или `dist/Transkrib.app` (macOS).

## Где что лежит

- Модели: `%LOCALAPPDATA%\Transkrib\models` на Windows,
  `~/Library/Application Support/Transkrib/models` на macOS.
- Настройки: там же, `settings.json`.

## Устройство

```
app/engine.py   выбор железа (mlx / cuda / cpu), скачивание модели, поток сегментов
app/media.py    ffmpeg: извлечение звука, конвертация, поиск пауз для резки
app/export.py   txt / srt / vtt / md
app/main.py     окно pywebview и мост Python ↔ JS
app/ui/         интерфейс (один html-файл)
build/          spec для PyInstaller и fetch_model.py для вшивания модели
bundled_models/ встроенная модель (создаётся fetch_model.py, в git не попадает)
```
