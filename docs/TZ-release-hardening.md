# Transkrib — ТЗ на доводку перед раздачей людям (Windows + macOS)

## 0. Кому и зачем этот документ

Документ самодостаточный: исполнитель (человек или другая модель) не видел переписки и
должен собрать всё по нему. Он описывает, что уже есть, что сломано, что доделать, и как
каждый пункт проверить. Порядок разделов: контекст → текущее устройство → правила работы →
задачи с критериями приёмки → сквозная проверка → сдача.

## 1. Контекст

**Продукт.** Transkrib — настольное приложение «закинул файл и ждёшь»: перетащил аудио или
видео, смотришь прогресс, текст появляется по мере распознавания, рядом с файлом ложатся
txt и srt. Плюс конвертер форматов на ffmpeg и разделение по говорящим. Всё локально, без
облака и ключей. Целевые пользователи — обычные люди на Windows (основная аудитория) и
macOS, часто со слабыми ноутбуками и 8 ГБ памяти.

**Решения владельца, менять нельзя:**
- Две сборки со встроенной моделью: `lite` (Whisper small, только процессор) и `medium`
  (Whisper medium; на Windows внутри CUDA-библиотеки для NVIDIA). Whisper large по умолчанию
  не предлагать: «перебор для людей и их компов». large-v3-turbo остаётся только
  необязательной догрузкой через настройки.
- После распаковки ничего не должно качаться: модель и модели говорящих внутри пакета.
- Windows и macOS обе нужны. Сборки делает GitHub Actions, у владельца Mac (Apple Silicon,
  8 ГБ), живого Windows нет ни у кого из разработчиков: всё Windows-специфичное проверяется
  только в CI на runner без GPU.
- Сертификат подписи кода не покупаем (решение владельца): SmartScreen и антивирусы
  закрываем инструкцией, не подписью. Тестера с NVIDIA нет: CUDA-путь проверят первые
  пользователи, поэтому диагностика и мягкий откат на процессор (A4) обязательны.

**Что уже случилось и чему научило (не повторять):**
- MLX-путь тихо падал в процессор из-за `ffmpeg`, которого нет в PATH у приложения,
  запущенного из Finder. Починено: MLX получает звук массивом. Урок: любую проверку
  сборки гонять с пустым PATH (`env -i PATH=/usr/bin:/bin`), selftest уже прячет ffmpeg.
- Параллельная диаризация на 8 ГБ выбивала MLX по памяти. Починено: диаризация после
  текста, при нехватке памяти Whisper выгружается. Урок: на целевых машинах два тяжёлых
  движка одновременно не запускать.
- Диаризация со стандартным шагом окна была в 6 раз медленнее нужного и дробила эфир на
  двоих на 15 «голосов». Подобрано: шаг окна 0.5, порог 0.65, обрывки короче 3 с клеятся
  к соседу. Урок: параметры проверять на двух записях: 4 голоса (`0-four-speakers-zh.wav`
  из релизов sherpa-onnx) и реальный эфир на двоих.
- Синтетические голоса из TTS (`say`) диаризация не различает. Для тестов говорящих
  нужна живая речь.
- Сообщение о переходе на процессор жило 4 секунды в тосте, и его никто не видел. Теперь
  предупреждение остаётся в карточке файла и пишется в лог с traceback. Урок: любое
  «тихое» ухудшение режима — в карточку и в лог.

## 2. Текущее устройство (состояние на момент написания)

Репозиторий: `github.com/BrimTV/transkrib` (приватный), ветка `main`. Python 3.11.

```
run.py                  точка входа (multiprocessing.freeze_support + app.main.main)
app/main.py             окно pywebview, класс Api — мост Python↔JS, настройки, --cli, --selftest
app/engine.py           выбор железа, модели, потоковая расшифровка, лог, память
app/diarize.py          разделение по говорящим (sherpa-onnx), привязка фраз к говорящим
app/media.py            ffmpeg: извлечение звука, конвертация, поиск пауз, резка
app/export.py           txt / srt / vtt / md, режимы таймкодов, говорящие
app/ui/index.html       весь интерфейс одним файлом (CSS+JS inline), русский
build/transkrib.spec    PyInstaller onedir, console=False, collect_all по нативным пакетам
build/fetch_model.py    кладёт модель и модели говорящих в bundled_models/ перед сборкой
.github/workflows/build.yml   матрица variant(lite, medium) × os(windows-latest, macos-14)
requirements*.txt       общие / -mac (mlx-whisper) / -win (nvidia-cublas-cu12, nvidia-cudnn-cu12)
README.md               установка, запуск из исходников, сборка
```

**Поток данных.** `Api.start` (app/main.py) → поток → `engine.transcribe_file(src, model_key,
language, emit, cancel, prefer_gpu, diarize, num_speakers)`:
1. `media.extract_wav` → 16 кГц моно WAV во временной папке `transkrib_*`.
2. `detect_backend`: `mlx` (macOS arm64 + mlx_whisper) → `cuda` (ctranslate2 видит GPU) → `cpu`.
3. `resolve_model('auto')` → встроенная модель из `bundled_models/variant.json`.
4. `load_model` → `ensure_model` (встроенная → кэш HF в `models_dir()` → скачивание с
   опросом размера папки для прогресса).
5. `_transcribe_fw` (faster-whisper, VAD включён, сегменты стримятся генератором) или
   `_transcribe_mlx` (файл режется по паузам `media.silence_midpoints` + `media.cut_points`
   на куски 120–150 с, каждый кусок отдаётся `mlx_whisper.transcribe` массивом float32;
   VAD нет).
6. Любое исключение GPU-пути → `log()` с traceback → событие `stage` с `note=True` →
   `unload_models()` → повтор на `cpu`.
7. Если `diarize`: при свободной памяти < 2.5 ГБ `unload_models()`, затем `diarize.run(wav,
   emit, num_speakers, cancel)` → событие `speakers` с `turns`.
8. `done` с временем и скоростью. В `finally` удаляется временный wav.

**События** (`emit(dict)`), все идут в JS через `window.evaluate_js("window.onEngineEvent(...)")`,
частые типы (`download`, `extract`, `progress`, `diar_progress`) прореживаются до 10/с:
`stage{stage,msg,note?}`, `download{done,total}`, `extract{progress}`, `progress{done_sec,total_sec}`,
`segment{start,end,text}`, `diar_progress{progress,eta}`, `speakers{turns,count}`,
`done{backend,elapsed,total_sec,msg}`, `cancelled`, `error{msg}`, `info{msg}`,
`convert_progress/convert_done/convert_error`.

**Api (JS→Python):** `info()`, `set_settings()`, `pick_files()`, `open_folder()`, `open_log()`,
`start()`, `cancel()`, `render()`, `autosave()`, `save_as()`, `memory()`, `convert()`.

**Данные пользователя** (`engine.data_dir()`): Windows `%LOCALAPPDATA%\Transkrib`, macOS
`~/Library/Application Support/Transkrib`. Внутри `settings.json`, `models/` (кэш HF),
`diarization/` (если модели говорящих докачивались), `transkrib.log`.

**Режимы запуска бинарника:** без аргументов — окно; `--selftest` — без окна, синтетический
тон через весь цикл + диаризация, пишет `selftest.log` в cwd, выход 0/1 (это и есть
проверка сборки в CI); `--cli <файл> [--speakers] [--model X] [--lang ru]` — расшифровка
в stdout для диагностики.

**Память.** `Api.start_housekeeping` раз в 30 с зовёт `engine.maybe_unload`: модель без работы
5 минут или свободной памяти < 1.5 ГБ → выгрузка (faster-whisper объект, MLX `ModelHolder`
через `sys.modules["mlx_whisper.transcribe"]`, `mx.clear_cache()`).

**Проверенные факты:** локальная сборка на Mac работает с MLX; на CI-маке MLX определяется,
но падает в CPU-фолбэк (виртуалка без Metal) — это норма; CUDA-путь на Windows никогда не
запускался на реальной видеокарте; кириллица в путях на macOS работает (ctranslate2, PyAV,
sherpa-onnx проверены в папке «Папка тест»); на Windows не проверялась.

**Факты из чтения исходников библиотек (pywebview 6.2.1, PyInstaller 6.22.2, ctranslate2
4.8.2, sherpa-onnx 1.13.7), на которые опираются задачи:**
- **Перетаскивание файлов сейчас не работает ни на одной ОС.** `pywebviewFullPath`
  подставляется только в Python-словарь события (`webview/util.py:299`) и только если
  зарегистрирован Python-обработчик `element.events.drop`; в JS-объект `File` он не
  попадает никогда. Наш JS в `index.html` (`f.pywebviewFullPath`) всегда получает пустой
  список. Работает только кнопка выбора файла.
- Колбэк прогресса sherpa-onnx вызывается только в фазе эмбеддингов, возврат не
  проверяется; сегментация и кластеризация колбэка не имеют. Отмена возможна только
  убийством отдельного процесса.
- `multiprocessing.Queue` создаёт `SemLock`, из-за которого стартует `resource_tracker`
  (тот самый лишний процесс, замеченный на macOS). Для воркера — `subprocess` с
  перезапуском собственного exe, не `multiprocessing`.
- ctranslate2 `compute_type="float16"` на карте без поддержки бросает исключение; `"auto"`
  выбирает лучший поддерживаемый тип с даунгрейдом.
- Дефолтный манифест PyInstaller уже содержит `longPathAware=true`; свой `manifest=` в
  `EXE()` заменяет его целиком, поэтому свой делать копией дефолтного плюс `activeCodePage`.
- В windowed-сборке (`console=False`) `sys.stdout` и `sys.stderr` равны `None`:
  прогресс-бар tqdm от huggingface_hub при скачивании модели падает с
  `AttributeError: 'NoneType' object has no attribute 'write'`.
- Проводник при распаковке zip ставит на файлы поток `Zone.Identifier` (Mark-of-the-Web);
  .NET отказывается грузить такие сборки (`Python.Runtime.dll`, `Microsoft.Web.WebView2.*`)
  с ошибкой 0x80131515 — вероятно, самая частая причина «окно не открывается».
- `media._run` не знает про `cancel`: «Остановить» во время извлечения звука из
  двухчасового видео не работает.

## 3. Правила работы для исполнителя

1. Не менять архитектуру (pywebview + один html, PyInstaller onedir, две сборки). Задачи
   ниже — доводка, не переписывание.
2. Каждая задача — отдельный коммит с понятным русским сообщением; пуш в `main` запускает
   CI, зелёный CI обязателен перед следующей задачей. Заголовок коммита ≤ 72 символов.
3. Любая проверка сборки — с пустым PATH: macOS `env -i HOME=$HOME PATH=/usr/bin:/bin
   ./dist/Transkrib.app/Contents/MacOS/Transkrib --selftest`; на Windows в CI PATH уже
   без ffmpeg (selftest сам его прячет).
4. Всё, что может тихо ухудшить режим (переход на CPU, отключение говорящих, автосохранение
   в другое место), обязано попасть и в карточку файла (событие `stage` с `note=True`),
   и в `engine.log()`.
5. Тексты для пользователя — по-русски, без жаргона: не «CUDA init failed», а «Видеокарта
   не сработала, считаю на процессоре». Техническая деталь — в скобках после, коротко.
6. Комментарии в коде — по-русски, объясняют «почему», а не «что» (см. существующие).
7. Ничего не скачивать при обычном запуске, если модель встроена. Скачивание только по
   явному выбору пользователя в настройках.
8. Не добавлять torch, не добавлять onefile, не включать UPX (ложные срабатывания антивирусов).
9. Тесты: минимальный pytest в `tests/` для чистой логики (export, фильтры, привязка
   говорящих, именование файлов), selftest для сквозного прохода в сборке, CI для платформ.

## 4. Задачи

Формат каждой: **Проблема → Что сделать → Где → Приёмка**. Приоритет: П0 — блокирует
раздачу, П1 — первый релиз, П2 — можно после.

### Блок А. Надёжность на Windows и общие П0

#### A0. Перетаскивание файлов (П0, обе ОС)
**Проблема.** Главный сценарий «закинул файл» не работает: см. факты выше.
**Что сделать.** В `main()` после `create_window`: `window.events.loaded += api._wire_dnd`.
`Api._wire_dnd`: для `("#dropTr", "transcribe")` и `("#dropCv", "convert")` —
`self.window.dom.get_element(sel).events.drop += functools.partial(self._on_drop, kind)`.
`Api._on_drop(kind, e)`: `paths = [f["pywebviewFullPath"] for f in
e.get("dataTransfer", {}).get("files", []) if f.get("pywebviewFullPath")]` →
`self._emit(dict(type="dropped", target=kind, paths=paths))`. В JS: `case 'dropped'` →
`addToQueue` / `addToConvert`; JS-слушатели `dragover/drop` оставить только для
`preventDefault` и подсветки `.over`. На Windows нужен WebView2 ≥ 1.0.1774 (Evergreen
обновляется сам). Известное ограничение: сопоставление по имени файла — два одноимённых
файла из разных папок за один раз дадут один путь.
**Приёмка.** Режим `--smoke` (см. Г2): после `loaded` диспатчить синтетический
`new DragEvent('drop', {dataTransfer: new DataTransfer(), bubbles: true})` на `#dropTr` и
проверять, что Python-обработчик вызван (пустой список путей). Настоящий drop из
Проводника/Finder — в ручной чек-лист Г4, на Mac владельца обязательно.

#### A1. Кириллица и пробелы в путях (П0)
**Проблема.** ctranslate2 и sherpa-onnx открывают файлы через `std::ifstream` по узкой
строке; на Windows это ANSI-кодовая страница, и путь `C:\Users\Иван\Downloads\Transkrib\...`
(куда пользователь распаковал `bundled_models`) или `%LOCALAPPDATA%` с кириллическим
именем не откроется. У большинства русских пользователей папка именно такая.
**Что сделать.** Два слоя.
1. Манифест exe с `activeCodePage=UTF-8` (Windows 10 1903+, на старых молча игнорируется):
   файл `build/transkrib.manifest` — копия `_DEFAULT_MANIFEST_XML` из
   `PyInstaller/utils/win32/winmanifest.py` (сохранить `longPathAware`, `supportedOS`) плюс
   в `<windowsSettings>`: `<activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/
   WindowsSettings">UTF-8</activeCodePage>`. В spec: `EXE(..., manifest=os.path.join(
   SPECPATH, "transkrib.manifest") if IS_WIN else None)`. Чинит все C++-библиотеки разом,
   включая воркер диаризации (тот же exe). В лог при старте писать `ACP=<GetACP()>`;
   65001 — маркер, что манифест применился.
2. Страховка для старых Windows: `engine.ascii_safe_path(p)` — если `win32`, путь не ASCII
   и `ctypes.windll.kernel32.GetACP() != 65001`: короткое имя через `GetShortPathNameW`
   (проверить результат `.isascii()`: при выключенных 8.3 вернётся тот же путь) → иначе
   junction `_winapi.CreateJunction(dir, r"C:\Users\Public\Transkrib\m-<sha1(path)[:8]>")`
   (stdlib, без прав администратора, ноль байт, между томами можно) → иначе копия.
   Применять к каталогу модели в `load_model` и к `seg`/`emb` в `diarize.run`. WAV
   отдавать faster-whisper массивом (`_wav_slice` уже есть), а не путём.
**Приёмка (CI, Windows, оба варианта).** Шаг после упаковки:
```
$base = "D:\a\Тест кириллицы ~ путь"; New-Item -ItemType Directory "$base\Лок","$base\Врем"
Expand-Archive Transkrib-*.zip $base
$env:LOCALAPPDATA="$base\Лок"; $env:TEMP=$env:TMP="$base\Врем"
Start-Process "$base\Transkrib\Transkrib.exe" --selftest -Wait -WorkingDirectory $base
```
Ожидание: `SELFTEST OK`, в логе `ACP=65001`, `transkrib.log` появился в `$base\Лок\Transkrib`.
На runner ACP=1252, поэтому без манифеста шаг гарантированно падает на загрузке модели —
честное воспроизведение бага. Плюс assert через `winmanifest.read_manifest_from_executable`,
что манифест содержит `activeCodePage`; юнит-тест `ascii_safe_path` на кириллической
папке с принудительным «ACP≠65001» через параметр.

#### A2. Проверка окружения Windows до создания окна (П0)
**Проблема.** Без WebView2 pywebview откатывается на MSHTML (Internet Explorer) и окно
пустое; без .NET Framework 4.7.2 pythonnet не стартует; без объяснений.
**Что сделать.** Модуль `app/winprep.py` (только Windows, импортируется в `main()` до
`import webview`, только в GUI-режиме):
- `webview2_version()`: `winreg`, значение `pv` в
  `HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}`,
  том же ключе без `WOW6432Node` и в `HKCU`; установлен, если `pv` непустой и не `0.0.0.0`
  (те же ключи проверяет pywebview в `winforms._is_chromium`).
- `dotnet_release()`: `HKLM\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full\Release` ≥ 461808.
- `require_ui_runtime()`: при отсутствии — `ctypes.windll.user32.MessageBoxW(0, текст,
  "Transkrib", MB_ICONERROR | MB_YESNO | MB_SETFOREGROUND)`; текст: «Для работы нужен
  компонент Microsoft WebView2. Открыть страницу загрузки?»; `IDYES` →
  `os.startfile("https://go.microsoft.com/fwlink/p/?LinkId=2124703")`; `sys.exit(2)`.
- `webview.start(gui="edgechromium", ...)` явно: любой рассинхрон даст исключение, а не
  тихий откат; исключение ловим и показываем тот же MessageBox.
**Приёмка.** Режим `Transkrib.exe --check-env` (Г2) пишет версию WebView2, .NET Release,
ACP, длину `sys._MEIPASS`, наличие ключевых DLL и ffmpeg; на runner exit 0. Отрицательная
ветка: env `TRANSKRIB_FAKE_NO_WEBVIEW2=1` → `webview2_version()` возвращает None, в
неинтерактивном режиме без MessageBox exit 3 — assert в CI.

#### A3. Отмена диаризации: отдельный процесс (П0)
**Проблема.** «Остановить» не действует на диаризацию (проверено: отмена через 3 с,
досчитала до конца). Патч колбэка не поможет (см. факты).
**Что сделать.** Воркер через `subprocess` с перезапуском собственного бинарника, не
`multiprocessing`:
- В `main()` ветка `--diarize-worker <wav> <out.json> --speakers N --parent-pid P` до
  `import webview` (по образцу `--selftest`/`--cli`).
- `diarize.worker_main(argv)`: `sys.stdout.reconfigure(encoding="utf-8")`; прогресс —
  JSON-строки в stdout с `flush=True`; сторожевой daemon-поток раз в 2 с
  `psutil.pid_exists(parent_pid)`, иначе `os._exit(1)`; `turns = run(wav, emit, n)`
  (сырые реплики, без `_finalize`); запись `out.json.tmp` + `os.replace`; исключение →
  traceback в stderr, exit 1. Воркер не импортирует webview/mlx и не пишет в
  `transkrib.log` (два процесса и ротация на Windows = `PermissionError`).
- `diarize.run_in_worker(wav_path, emit, num_speakers, cancel)` в родителе (вызывать из
  `transcribe_file` вместо `run`): `ensure_models(emit)` в родителе; команда frozen →
  `[sys.executable, "--diarize-worker", ...]`, из исходников → `[sys.executable, "-m",
  "app.main", "--diarize-worker", ...]` с `cwd` = корень репо; `stderr=PIPE` читается
  отдельным потоком в хвост 40 строк; `creationflags=_NO_WINDOW` из `media`; понизить
  приоритет `psutil.Process(pid).nice(BELOW_NORMAL_PRIORITY_CLASS)` на Windows, чтобы окно
  не подвисало. Поток-наблюдатель: `while proc.poll() is None: if cancel.wait(0.2):
  proc.kill(); break` (readline по stdout блокируется, в фазах сегментации воркер молчит).
  После цикла обязательно `proc.wait(timeout=10)` — иначе на Windows `audio.wav` нельзя
  удалить, пока ОС не закрыла хэндлы убитого процесса. `cancel` → `InterruptedError`;
  `rc != 0` → `RuntimeError(хвост stderr)`; иначе `json.load` → `_finalize` в родителе.
- Путь к wav, не массив: файл уже лежит в tmpdir, воркер читает своим `_read_wav`.
**Приёмка.** Все selftest (исходники и exe, обе ОС) переключить на `run_in_worker` — это
проверяет перезапуск frozen-бинарника с флагом. В selftest тест отмены: 20 минут белого
шума numpy в WAV, `run_in_worker` с `cancel`, взвести через 3 с; assert `InterruptedError`
< 2 с после взвода, `psutil.Process().children()` пуст, wav удаляется без `PermissionError`.
Тест сироты: воркер вручную с несуществующим `--parent-pid` → выход ≤ 3 с.

#### A4. NVIDIA-путь: типы вычислений и диагностика (П0)
**Что сделать.**
- `WhisperModel(path, device="cuda", compute_type="auto")`: ct2 выберет int8_float16 на
  современных картах, на старых уронит до float32/int8_float32 без исключения. Аргументы
  против фиксированного float16: исключение на GTX 10xx/MX/старых Quadro; VRAM medium
  float16 ≈ 1.5 ГБ против int8_float16 ≈ 0.8 ГБ, а карты с 2–4 ГБ (GTX 1650, ноутбуки) —
  типичная цель. После загрузки писать в лог фактический `model.model.compute_type`.
  Цепочка в `transcribe_file`: `cuda/auto` → при исключении (в том числе `CUDA out of
  memory` на первом сегменте) `cuda/int8_float32` → `cpu`. Сейчас цикл идёт сразу в cpu.
- `engine._cuda_diag()` один раз из `_has_cuda`, в лог и в `hardware_info` (тултип бейджа
  «почему процессор»): `ctranslate2.__version__`, `get_cuda_device_count()`,
  `get_supported_compute_types("cuda", i)`; каталоги из `_prepare_cuda_dlls` и
  `ctypes.WinDLL` для `cublas64_12.dll`, `cublasLt64_12.dll`, `cudnn64_9.dll` с текстом
  `WinError` (грузятся и без GPU — проверяет упаковку); `ctypes.WinDLL("nvcuda.dll").
  cuDriverGetVersion` → `< 12000` → «драйвер старее CUDA 12 (нужен 527.41 или новее),
  видеокарта не используется», нет nvcuda.dll → «драйвера NVIDIA нет»; `nvidia-smi
  --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader`
  (PATH, затем `%SystemRoot%\System32`, `%ProgramFiles%\NVIDIA Corporation\NVSMI`;
  `timeout=5`, без окна консоли; отсутствие — не ошибка); время загрузки модели и первого
  сегмента.
- Текст в карточке при провале: «Видеокарта NVIDIA не сработала, считаю на процессоре.
  Обновите драйвер NVIDIA (нужен 527 или новее)».
**Приёмка.** CI Windows medium: assert наличия `_internal\nvidia\cublas\bin\cublas64_12.dll`,
`cublasLt64_12.dll`, `_internal\nvidia\cudnn\bin\cudnn64_9.dll`, `cudnn_ops64_9.dll`,
`cudnn_cnn64_9.dll`; selftest: `_cuda_diag()` не бросает, DLL грузятся через ctypes,
`get_cuda_device_count()==0`. Живой GPU-путь проверят первые пользователи по логу
(см. решение владельца), поэтому диагностика выше обязательна.

#### A5. Mark-of-the-Web и самолечение при старте (П0)
**Проблема.** См. факты: после распаковки Проводником у всех файлов есть поток
`Zone.Identifier`, .NET не грузит `Python.Runtime.dll` и WebView2-сборки → окно не
открывается.
**Что сделать.** В `run.py` до импорта `app`: если существует
`_internal\webview\lib\Python.Runtime.dll:Zone.Identifier` (`os.path.exists(path +
":Zone.Identifier")`) — пройти по `_internal` и `os.remove(f + ":Zone.Identifier")` для
`.dll/.pyd/.exe`, ошибки игнорировать, результат в лог. Один раз, секунды. В ЧИТАЙ.txt —
«правой кнопкой по zip → Свойства → Разблокировать» как дополнительная мера.
**Приёмка (CI, Windows).** После `Expand-Archive` навесить на все DLL
`Set-Content -Path $f -Stream Zone.Identifier -Value "[ZoneTransfer]`r`nZoneId=3"`,
запустить `--smoke`, assert OK и что потоки исчезли. Это же проверяет гипотезу про
0x80131515 на реальном .NET.

#### A6. Инструкция в архиве, упаковка, версия, иконка (П0)
**Что сделать.**
- `build/README-windows.txt` (UTF-8 с BOM, CRLF через `.gitattributes`), в workflow
  `Copy-Item build\README-windows.txt dist\ЧИТАЙ.txt` и `7z a -tzip -mx=3 -mcu=on
  Transkrib-<variant>.zip .\dist\Transkrib .\dist\ЧИТАЙ.txt` (`-mcu=on` — имена в UTF-8,
  иначе на нерусской Windows кракозябры). Имя архива короткое: `Transkrib-medium.zip`,
  `Transkrib-lite.zip` (длина пути после распаковки). Разделы файла: что это и вариант
  сборки; требования (Windows 10 1903+ / 11, 64-бит; 4 ГБ ОЗУ для lite, 6–8 для medium;
  2–3 ГБ диска; WebView2); установка — распаковать целиком («Извлечь всё»), запускать
  `Transkrib\Transkrib.exe`, из окна архива нельзя (будет «Failed to load Python DLL»),
  папку держать недалеко от корня диска и не в OneDrive; «Разблокировать» в свойствах
  zip; SmartScreen («Подробнее → Выполнить в любом случае», программа не подписана);
  антивирус (возможное ложное срабатывание на PyInstaller, при карантине добавить папку в
  исключения); NVIDIA (драйвер ≥ 527, иначе процессор, бейдж в шапке показывает режим);
  первый запуск 10–20 с; где лог (`%LOCALAPPDATA%\Transkrib\transkrib.log`, кнопка
  «Открыть лог» в настройках) и куда присылать; где результаты.
- Версия в одном месте `app/__init__.py`; CI при теге `vX.Y.Z` передаёт
  `TRANSKRIB_VERSION`, иначе `0.1.0+<sha7>`; `fetch_model.py` пишет в `variant.json`
  поле `build: {version, commit, date, variant}`; spec генерирует Windows-ресурс версии
  (`EXE(version=VSVersionInfo(...))` прямо в spec) и `icon=` (exe без иконки и версии
  выглядит для людей и антивирусов как малварь; иконка — простой svg→ico/icns в
  `build/icon.*`). В шапке UI бейдж «Transkrib 0.2.0 · medium · 2026-09-02», клик →
  `webbrowser.open` на Releases.
- Второй консольный exe `Transkrib-console.exe` (`console=True`) в том же `COLLECT` —
  общий `_internal`, +1–2 МБ: пользователи смогут прислать вывод `--cli`/`--check-env`, CI
  избавится от `Start-Process`+`selftest.log`.
- `--selftest` пишет `selftest.log` в `data_dir()`, а не в cwd (папка exe бывает только
  для чтения).
**Приёмка.** В артефакте `ЧИТАЙ.txt` в корне и `Transkrib\Transkrib.exe`;
`(Get-Item Transkrib.exe).VersionInfo.ProductVersion` равен ожидаемой; selftest печатает
`build`; README ссылается на ЧИТАЙ.txt.

#### A7. Мелкие, но обязательные (П0/П1)
- **stdout/stderr = None в windowed-сборке.** В `run.py` до импортов:
  `HF_HUB_DISABLE_PROGRESS_BARS=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, `DO_NOT_TRACK=1`; в frozen-GUI-режиме подменить
  `sys.stdout/sys.stderr` на поток в `logging`. Проверка CI: `Transkrib.exe --cli tone.wav
  --model tiny` (windowed, с сетью) — ловит падение tqdm при скачивании. (П0)
- **Отмена ffmpeg.** Прокинуть `cancel` в `media._run` / `extract_wav` /
  `silence_midpoints`: поток-наблюдатель + `proc.kill()`. CI: отмена сразу после старта
  извлечения 30-минутного файла → `InterruptedError` < 1 с, процесса ffmpeg нет. (П0)
- **Двойной запуск.** Лок `data_dir()/instance.lock` (Windows `msvcrt.locking`, POSIX
  `fcntl.flock`), только GUI-режим; при конфликте на Windows `FindWindowW(None,
  "Transkrib <версия>")` → `ShowWindow(SW_RESTORE)` + `SetForegroundWindow`, exit 0.
  CI: `--smoke --hold 30`, второй `--smoke` → exit 5 за < 5 с. (П1)
- **VC++ runtime.** Assert в CI, что `msvcp140.dll`, `vcruntime140.dll`,
  `vcruntime140_1.dll` лежат в `_internal` (PyInstaller берёт их из System32 runner;
  иначе на чистой Windows «VCRUNTIME140_1.dll не найден»). (П0)
- **Профиль WebView2.** `private_mode=False` создаёт профиль в `%APPDATA%\pywebview`
  (roaming, десятки МБ). Интерфейс не использует localStorage → `storage_path=data_dir()/
  webview` или вернуть `private_mode=True`. (П1)
- **`os.startfile`** бросает `OSError` при сломанной ассоциации — обернуть, возвращать
  `dict(ok, error)`; для файла `explorer /select,"path"`, URL через `webbrowser`. (П1)
- **`pick_files`.** В описании типов нельзя `/ - , .` (иначе `ValueError` и клик молча
  ничего не делает); winforms при пустом `directory` читает `HOMEPATH` — передавать
  `directory=` явно (последняя папка из настроек). Selftest: `webview.util.parse_file_type`
  на реальном кортеже. (П1)
- **Длинные пути.** `longPathAware` есть, но работает только при
  `LongPathsEnabled=1` в реестре, у большинства 0. При старте `len(sys._MEIPASS) > 170` →
  сообщение «перенесите папку ближе к корню диска». CI: `LongPathsEnabled=0` + распаковка
  в путь ~200 символов + selftest. (П1)
- **`sys.excepthook` и `threading.excepthook`** → в лог: сейчас исключения фоновых
  потоков в windowed-сборке исчезают. (П0)
- **Минимальная ОС** в ЧИТАЙ.txt: Windows 10 1903+ x64; 7/8.1 не поддерживаются, сказать явно.

### Блок Б. Качество расшифровки и нагрузка

Общее наблюдение, на котором строится блок: оба бэкенда сходятся в одной точке — обёртке
`emit()` внутри `engine.transcribe_file` (сейчас строки ~421–426). Всё, что должно
работать одинаково для faster-whisper и MLX (фильтры, подсчёты), ставится туда, а не в
`_transcribe_fw` / `_transcribe_mlx`.

#### Б1. Фильтр галлюцинаций Whisper (П0)
**Проблема.** На музыке, джинглах и тишине Whisper выдаёт повтор одной фразы много раз
подряд или фразы-маркеры («Субтитры сделал DimaTorzok», «Продолжение следует», «Спасибо за
просмотр», «Thank you for watching», «Редактор субтитров А.Синецкая»). Сейчас всё это
доходит до пользователя. Оба движка уже отдают сигналы качества на сегмент
(`no_speech_prob`, `avg_logprob`, `compression_ratio`: у faster-whisper поля `Segment`, у
mlx_whisper те же ключи в `res["segments"][i]`), но мы их выбрасываем.
**Что сделать.**
1. В `_transcribe_fw` и `_transcribe_mlx` добавлять в событие `segment` поля
   `no_speech_prob`, `avg_logprob`, `compression_ratio` (UI и экспорт лишние поля игнорируют).
2. Новый модуль `app/cleanup.py` с чистым классом `SegmentFilter(language)` и методом
   `verdict(event) -> str | None` (причина отброса или None). Правила по порядку, первое
   сработавшее отбрасывает:
   - `no_speech_prob > 0.6 and avg_logprob < -1.0` → `no_speech`;
   - `compression_ratio > 2.4` → `compression`;
   - сегмент ≤ 12 слов, содержащий фразу-маркер после нормализации → `marker`.
     Нормализация: нижний регистр, ё→е, без пунктуации, схлопнуть пробелы. Маркеры ru:
     «субтитры сделал», «субтитры создал», «субтитры подготовил», «редактор субтитров»,
     «корректор», «продолжение следует», «спасибо за просмотр», «подписывайтесь на канал»,
     «ставьте лайк», «dimatorzok»; en: «thank you for watching», «thanks for watching»,
     «subtitles by», «subs by», «please subscribe», «like and subscribe», «amara org»,
     «www», «copyright». При языке `auto` — оба списка;
   - нормализованный текст равен предыдущему: счётчик повторов; отбрасывать третью копию
     любого текста и уже вторую копию фразы от 4 слов → `repeat` (первый экземпляр не
     отзывать: он уже показан и может быть настоящей речью);
   - ≥ 8 слов и доля уникальных < 0.35 → `loop`;
   - плотность > 7 слов/с при длительности > 1 с → `density`.
3. Подключение в обёртке `emit()`: для `segment` спросить `verdict`; при отбросе —
   `log(f"drop {why} [{start:.1f}-{end:.1f}] {text[:80]}")` и не передавать дальше.
   Счётчик отброшенных в `done` (`dropped=N`, в `msg` хвост «(убрано N повторов)», если N>0).
4. `hallucination_silence_threshold` faster-whisper не включать: требует word_timestamps
   и +10–20% времени на CPU.
**Приёмка.** `tests/test_cleanup.py` без моделей: «Продолжение следует» ×5 → остаётся 1;
«Да. Да.» ×2 короткие → остаются оба; одиночное «спасибо за просмотр» → drop; длинный
нормальный сегмент с `no_speech_prob=0.1` → pass; `compression_ratio=3.1` → drop.
Selftest: на синтетическом тоне 440 Гц после фильтра должно быть 0 сегментов (сейчас tiny
выдумывает «Редактор субтитров…» и это проходит) — добавить ассерт.

#### Б2. VAD и резка для MLX-пути (П1)
**Проблема.** MLX получает куски по 120–150 с целиком, включая музыку и тишину, и там
галлюцинирует чаще, чем faster-whisper с VAD.
**Что сделать.** Переиспользовать Silero VAD из faster-whisper: `faster_whisper.vad.
get_speech_timestamps(audio_np, VadOptions(...), sampling_rate=16000)`; модель
`assets/silero_vad_v6.onnx` уже попадает в сборку через `collect_all("faster_whisper")`.
1. `media.speech_regions(wav, total_sec, on_progress, cancel, block=600)`: читать wav
   блоками по 10 минут (перенести `engine._wav_slice` в `media`), для каждого блока
   `get_speech_timestamps` с `VadOptions(threshold=0.5, min_speech_duration_ms=250,
   min_silence_duration_ms=500, speech_pad_ms=400)`, регионы сдвигать на начало блока,
   в конце склеить регионы с промежутком < 1 с. Стадия в UI «Ищу речь» с прогрессом
   (цена ≈ 1–2% длительности).
2. В `_transcribe_mlx`: если регионов нет → `stage` с `note=True` «Речи в записи не
   найдено», прогресс 100%, выход. Границы кусков брать из промежутков между регионами
   длиной ≥ 0.6 с через существующий `media.cut_points` (ffmpeg `silence_midpoints`
   оставить как фолбэк, если VAD упал). Для куска `[a,b)`: пересечь с регионами; пусто →
   пропустить кусок без вызова модели (прогресс продвинуть); если тишины внутри куска <
   20% — отдать кусок целиком; иначе передать `clip_timestamps=[s-a, e-a, ...]` по
   регионам. `clip_timestamps` у `mlx_whisper.transcribe` считает сегменты от начала
   массива, поэтому пересчёт таймкодов не нужен — прибавляем `a`, как сейчас. Явно
   передавать `no_speech_threshold=0.6, logprob_threshold=-1.0`.
3. Убрать из `_transcribe_mlx` мёртвые `tmpdir` и `piece` (остались после перехода на массивы).
**Влияние на таймкоды.** Начало первого сегмента в клипе сдвинется на `speech_pad_ms`
(0.4 с) раньше — в пределах точности самого Whisper.
**Приёмка.** `tests/test_media.py`: `speech_silence_speech.wav` (речь 10 с, тишина 40 с,
речь 10 с) → ровно 2 региона с границами ±0.5 с; `silence60.wav` → пусто. Selftest:
`faster_whisper.vad.get_vad_model()` грузится (ловит отсутствие onnx в сборке). Ручная
на Mac владельца: подкаст с музыкальной заставкой 30 с — первый сегмент после заставки,
«Субтитры сделал…» нет.

#### Б3. Резка на куски для faster-whisper и память на длинных файлах (П1)
**Проблема.** `model.transcribe(path)` декодирует весь файл в float32 (3 ч ≈ 690 МБ),
считает лог-мел на весь файл (≈ 550 МБ), копирует при сборке чанков — пик около 2 ГБ плюс
модель 1.5 ГБ. На Windows-ноутбуке с 8 ГБ это на грани.
**Что сделать.** Общая функция `_pieces(wav, total_sec, regions)` → список `[a,b)` по
10–15 минут на паузах (те же регионы VAD, что в Б2). `_transcribe_fw` зовёт
`model.transcribe(np_array, vad_filter=True, language=lang, ...)` на каждый кусок,
прибавляя `a` к таймкодам; при `language='auto'` язык определяется на первом куске и
дальше передаётся явно. Бонус: отмена и прогресс между кусками, единая логика с MLX.
Диаризация на файлах длиннее часа: всегда выгружать Whisper перед ней (не только при
< 2.5 ГБ) и показывать оценку «примерно N мин» до старта (0.05× длительности при шаге окна 0.5).
**Приёмка.** `tests/test_engine_slow.py` (`@pytest.mark.slow`, tiny): сегменты есть в
речи и нет в тишине на `speech_silence_speech.wav`; таймкоды второго фрагмента ≈ 50 с.
Замер `psutil` peak RSS на 2-часовом файле (склеить эфир владельца дважды) на CPU-пути
< 2.5 ГБ с моделью small.

#### Б4. Нагрузка на процессор (П1)
**Что сделать.** `engine.cpu_workers()`: `n = psutil.cpu_count(logical=False) or
os.cpu_count() or 2`; вернуть `n-1` при `n ≥ 4`, иначе `n`. Использовать в `load_model`
(`cpu_threads`) и в `diarize.run` (`threads = min(4, cpu_workers())`; выше 4 сегментация
sherpa не масштабируется, а параллельного Whisper больше нет). `hardware_info()`
дополнить `physical_cpu`. Пункт настроек «Не занимать все ядра» (по умолчанию включён).
**Приёмка.** pytest с `monkeypatch` `psutil.cpu_count`: 8→7, 4→3, 2→2, None→os.cpu_count.
Лог при загрузке модели: `cpu_threads=N (физических M)`.

#### Б5. Понятные ошибки (П0)
**Что сделать.** `class UserError(Exception)` в `engine.py`: текст сразу для пользователя;
в `transcribe_file` → `msg = str(e)` для UserError, иначе «Не удалось обработать файл
(техническая деталь: …)», traceback в лог. Общая `media.check_readable(path)` для
`extract_wav` и `Api.start`:
- нет файла → «Файл не найден: имя. Возможно, он перемещён или переименован»;
- размер 0 → «Файл пустой (0 байт)»;
- OneDrive/iCloud-плейсхолдер: Windows `os.stat(p).st_file_attributes & (0x1000 | 0x400000)`
  (OFFLINE, RECALL_ON_DATA_ACCESS) → «Файл хранится только в облаке OneDrive. Откройте
  папку, выберите «Всегда сохранять на этом устройстве» и повторите»; macOS — рядом лежит
  `.имя.icloud` → «Файл ещё не скачан из iCloud»;
- место: `shutil.disk_usage(tmp).free < длительность × 32000 × 1.2` → «На диске нет
  места для временного файла: нужно ~N МБ, свободно M МБ».
В `extract_wav` явный `-map 0:a:0` (нужен и для нескольких дорожек). Таблица
`FFMPEG_HINTS` подстрока→текст в `media._run`: `Stream map '0:a:0' matches no streams` →
«В этом видео нет звуковой дорожки»; `Invalid data found` / `moov atom not found` → «Не
удалось прочитать файл как аудио или видео: он повреждён или недокачан»; `Permission
denied` → «Нет доступа к файлу (права или файл открыт другой программой)»; `No space left`
→ текст про место; иначе прежний хвост stderr. В `ensure_model` и `diarize._download`:
перед скачиванием проверять место (`size_mb × 1.3`); `ConnectionError`,
`LocalEntryNotFoundError`, тексты `getaddrinfo` / `Max retries` / `timed out` → «Нет
доступа к интернету — модель «medium» (1.5 ГБ) скачать не удалось. Проверьте подключение
или выберите встроенную модель в настройках». `MemoryError` и mlx «Failed to allocate» /
«out of memory» → «Не хватает оперативной памяти. Закройте другие программы или выберите
модель поменьше».
**Приёмка.** `tests/test_media.py`: `noaudio.mp4` (`ffmpeg -f lavfi -i testsrc=d=3 -an`),
`broken.mp4` (первые 10 КБ настоящего mp4), пустой файл, несуществующий путь →
`pytest.raises(UserError, match=...)`. Нет интернета — `monkeypatch snapshot_download`,
бросающий `requests.ConnectionError`.

#### Б6. Тихие записи, несколько дорожек, сон компьютера (П1)
- **Тихие записи.** После добавления VAD Silero будет пропускать речь на тихом звуке.
  `ffmpeg -af volumedetect -f null -` (только декод) → `mean_volume`; если < −30 дБ,
  переизвлечь с `-af volume=<gain>dB`, `gain = min(-20 - mean, -1 - max)` (статическое
  усиление, шум в паузах не поднимать компрессором). Проверка: `quiet.wav` (−32 дБ) даёт
  те же сегменты, что исходник.
- **Несколько аудиодорожек** (OBS, скринкасты): `-map 0:a:0` + в лог и карточку
  «дорожек: N, взята первая». Выбор дорожки в UI — П2.
- **Сон ноутбука во время долгой задачи** (на Mac ещё и App Nap): на время задачи macOS
  `NSProcessInfo.processInfo.beginActivityWithOptions_reason_` (pyobjc уже в зависимостях
  через webview.cocoa) с `ActivityUserInitiated | IdleSystemSleepDisabled`; Windows
  `ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` из
  рабочего потока, сброс в `finally`.
- **8 кГц телефония**: ничего не делать, только логировать исходную частоту из
  `media.probe_info()` для диагностики жалоб.

### Блок В. Поведение приложения

#### В1. Автосохранение без перезаписи и с фолбэком (П0)
**Проблема.** `Api.autosave` пишет `<имя>.txt`/`.srt` рядом с файлом и молча перезаписывает
существующие (реальный случай: рядом с `video.mp4` уже лежат скачанные `video.srt`); на
папке только для чтения или сетевом диске падает.
**Что сделать.**
- Реестр наших файлов `data_dir()/autosave_index.json`: `{abs_path: {size, mtime}}`,
  заполняется при каждой нашей записи. Перезаписывать файл можно, только если он в
  реестре и его `size/mtime` совпадают (создали мы и после нас не правили). Иначе имя
  `<имя>.transkrib.<fmt>`, при коллизии `<имя>.transkrib-2.<fmt>` и т.д. Плееры
  подхватывают `video.transkrib.srt` так же, как `video.srt`.
- Не проверять `os.access` (врёт на сетевых дисках и OneDrive): пробовать писать, ловить
  `PermissionError` / `OSError` с `errno in (EACCES, EROFS, EPERM, ENOENT)` → каталог
  `~/Documents/Transkrib/` (Windows `%USERPROFILE%\Documents\Transkrib`), если и он не
  пишется → `data_dir()/output`. `ENOSPC` — не фолбэк, а ошибка «Нет места на диске».
- Запись атомарная: `out + ".tmp"` → `os.replace`.
- Возврат `dict(written=[...], fallback=None | {"dir", "reason"})`. В JS: тост «Папка с
  файлом недоступна для записи — сохранил в Документы/Transkrib», `j.outputs = written`,
  кнопка «Открыть папку» открывает папку первого сохранённого файла.
**Приёмка.** `tests/test_autosave.py` с реальным `Api` (`window=None`): чужой `a.txt` →
результат в `a.transkrib.txt`, чужой не изменён (хеш); повторный autosave → перезаписан
`a.transkrib.txt`, суффикс не растёт; «наш» файл, изменённый пользователем → следующий
раз `a.transkrib-2.txt`; `monkeypatch open` с `PermissionError` → `fallback` заполнен,
файлы в фолбэк-каталоге; `ENOSPC` → ошибка про место. Ручная: `chmod 555` на папку.

#### В2. Ошибки моста, гонка занятости, зависшие карточки (П0)
**Проблема.** Исключение в `Api.start`/`convert`/`autosave` уходит в JS как reject, его
никто не ловит, карточка остаётся в «Запуск…». Гонка: `done` уходит из `transcribe_file`
раньше, чем `finally` в `Api.start.run` сбросит `_busy`; при выключенном автосохранении JS
успевает вызвать `start` следующего файла и получить «уже идёт задача». Исключение внутри
`window.onEngineEvent` глотается в `Api._emit` (`except Exception: pass`), очередь замирает.
**Что сделать.**
- JS: единая `async function api(name, ...args)` с try/catch вокруг всех
  `pywebview.api.*` (pywebview отдаёт `Error` с `message`); при ошибке тост и проброс.
  `pump()`: при исключении или `!r.ok` → карточка в `error`, `running=false`,
  `renderQueue()`, `pump()` дальше. `window.onEngineEvent` в try/catch. Сторожевой
  таймер: карточка `running` с `msg==='Запуск…'` без событий > 120 с → error «Движок не
  ответил». `init()` при ошибке — в шапке «Не удалось запуститься, см. transkrib.log» и
  кнопка «Открыть лог».
- Python: `Api.start` целиком в `try/except → dict(ok=False, error=...)`, `_busy=True`
  только после успешного `Thread.start()`, хранить `self._thread`; если `_busy` и поток
  жив — `join(3.0)` прежде чем отвечать «занято».
**Приёмка.** Ручной хак в `--debug`: `pywebview.api.start = () => { throw new Error('boom') }`
→ карточка красная, следующий файл стартует. Три файла с выключенным автосохранением
подряд — ни одного «уже идёт задача».

#### В3. Временные файлы, уборка, лог (П1)
- Временные wav в `data_dir()/tmp` (`tempfile.mkdtemp(prefix="transkrib_", dir=...)` во
  всех местах: `transcribe_file`, selftest), а не в системном temp: на Windows он бывает на
  маленьком диске.
- `engine.cleanup_temp(max_age_sec=3600)` при старте в фоновом потоке: удалять
  `transkrib_*` старше часа в `data_dir()/tmp` и в `tempfile.gettempdir()` (старые
  версии). Порог в час защищает wav второго запущенного экземпляра.
- Лог через `logging` + `RotatingFileHandler(maxBytes=2 МБ, backupCount=2, encoding=utf-8)`;
  `engine.log()` остаётся обёрткой. При старте строка-разделитель: версия, платформа,
  `hardware_info()` — иначе по логу от пользователя не понять, какая сборка.
**Приёмка.** pytest: `transkrib_x` с mtime −2 ч удалён, свежий `transkrib_y` нет;
3 МБ строк в лог → есть `.log` и `.log.1`, суммарно < 4.1 МБ.

#### В4. Очередь (П1)
- Повторное добавление файла со статусом `done/error` не создаёт новую карточку, а
  сбрасывает существующую (`queued`, пустые segments/turns/names/notes). На карточках
  `done`/`error` кнопки «Повторить» и «×», у `running` только «Остановить».
- После «Остановить» остальные `queued` не стартуют сами: помечать «Ожидает» и общая
  кнопка «Продолжить очередь».
- Когда очередь опустела — тост «Готово N из M, ошибок K».
- В карточке писать, какой моделью и бэкендом сделано (`j.model`, `j.backend`).
**Приёмка.** Три файла, второй `broken.mp4`: первый done, второй error с понятным
текстом, третий done, итоговый тост; «Повторить» на втором — карточка одна; «Остановить»
на первом из трёх — остальные ждут.

#### В5. Версия и обновления (П2)
В настройках версия (`app.__version__`) и ссылка «Проверить обновления» на Releases
(через `webbrowser`). Версия в одном месте: `build/transkrib.spec` читает её из пакета.

#### В6. Стерео-звонки: говорящие по каналам (П2)
Записи Zoom и телефонных приложений часто держат собеседников в разных каналах.
`media.probe_info()` → если 2 канала: извлечь `-ac 2`, RMS-огибающие L/R по 100 мс, маски
активности (> −40 дБFS), взаимоисключаемость `1 − |L∧R| / |L∨R|`; если > 0.7 и в обоих
каналах речь ≥ 5% — строить `turns` из масок (сглаживание 0.5 с) и не звать sherpa.
Проверка: `stereo_call.wav` (голос слева 0–10 с, справа 12–22 с) → 2 говорящих без sherpa.

### Блок Г. Проверки

#### Г1. Фикстуры и юнит-тесты (pytest)
`tests/conftest.py` генерирует фикстуры в `tmp_path_factory` через
`imageio_ffmpeg.get_ffmpeg_exe()` (тот же бинарник, что в приложении); бинарники в
репозиторий не класть, кроме одного `tests/fixtures/speech_ru_10s.wav` (16 кГц моно,
~300 КБ, живая или TTS-речь для тестов текста, не говорящих):
- `silence60.wav`: `-f lavfi -i anullsrc=r=16000:cl=mono -t 60`;
- `music60.wav`: `-f lavfi -i "sine=f=440:d=60" -af tremolo=f=2`;
- `speech_silence_speech.wav`: concat речь + тишина 40 с + речь;
- `noaudio.mp4`: `-f lavfi -i testsrc=d=3:s=160x120 -an`; `broken.mp4`: первые 10 КБ;
- `quiet.wav`: `-af volume=-32dB`; `phone8k.wav`: `-ar 8000`;
- `stereo_call.wav`: два голоса в разных каналах со сдвигом 12 с;
- `multitrack.mkv`: видео + две аудиодорожки.
Файлы тестов: `test_cleanup.py` (Б1), `test_media.py` (Б5, Б2, Б6), `test_autosave.py`
(В1), `test_cpu.py` (Б4), `test_housekeeping.py` (В3), `test_export.py`, `test_assign.py`
(`diarize.assign`, `_finalize`), `test_paths.py` (A1, на Windows-runner), `test_engine_slow.py`
(`@pytest.mark.slow`, tiny). В CI `pytest -q -m "not slow"` до сборки на обеих платформах;
slow — по расписанию раз в сутки или вручную.

#### Г2. Режимы бинарника для проверок
Диспетчер в `main()` до `import webview`:
- `--selftest` (расширить): тон 440 Гц → после фильтра 0 сегментов; диаризация через
  воркер; отмена диаризации через 3 с → `InterruptedError` < 2 с, детей нет; отмена ffmpeg;
  `faster_whisper.vad.get_vad_model()` грузится; `cleanup_temp()` и ротация не падают;
  `_cuda_diag()` не бросает и DLL грузятся; `parse_file_type` на кортеже `pick_files`;
  атрибут OFFLINE (`attrib +O`) даёт понятный текст; в лог: версия/build, ACP, бэкенд,
  `cpu_workers`, `physical_cpu`, пути моделей и их `ascii_safe_path`. `selftest.log` в
  `data_dir()`.
- `--check-env`: факты окружения (WebView2, .NET, ACP, длина `_MEIPASS`, DLL, ffmpeg),
  коды выхода 0 ок / 3 нет WebView2 / 4 длинный путь.
- `--smoke [--hold N]`: создать окно, на `loaded` вызвать `pywebview.api.info()`,
  синтетический drop на `#dropTr`, записать `READY`/`OK` в `smoke.log`, `window.destroy()`;
  сторожевой таймер 90 с → exit 2. GUI на GitHub-runner Windows и macOS работает
  (pywebview сам так тестируется).
- `--diarize-worker` (A3), `--cli` (есть).

#### Г3. CI (build.yml)
Порядок: install → `pytest -m "not slow"` → fetch model → selftest (source) → build →
selftest (built, через `Transkrib-console.exe`) → package с `ЧИТАЙ.txt` → проверки
артефакта → upload. Проверки артефакта на Windows: (1) selftest в кириллической папке с
кириллическими `LOCALAPPDATA`/`TEMP` (A1); (2) Zone.Identifier на все DLL + `--smoke`
(A5); (3) `LongPathsEnabled=0` + путь ~200 символов + selftest; (4) двойной запуск
(A7); (5) selftest под «глухим» прокси `HTTPS_PROXY=http://127.0.0.1:9` — ни один вызов
не ушёл в сеть; (6) assert-ы: манифест содержит `activeCodePage`, ресурс версии, есть
`cublas64_12.dll`/`cudnn64_9.dll` (medium), `msvcp140.dll`/`vcruntime140_1.dll`,
`ЧИТАЙ.txt` в корне zip; (7) `Transkrib.exe --cli tone.wav --model tiny` с сетью
(tqdm при stdout=None). На macOS: (5), `--smoke`, selftest с воркером, `env -i
PATH=/usr/bin:/bin`. Порог времени: ≤ 30 минут на самый тяжёлый вариант.

#### Г4. Ручная приёмка на живых машинах (перед раздачей)
Чек-лист, прогнать на Mac владельца и на любом доступном Windows без NVIDIA (виртуалка
Parallels/UTM с Windows 11 подойдёт для всего, кроме CUDA):
1. Распаковать zip в `C:\Users\<кириллица>\Downloads`, запустить, окно открылось.
2. Перетащить mp4 в окно из Проводника/Finder — файл принят (после A0; до неё не работает
   нигде). Если на конкретной машине не сработало — отметить как известную проблему с
   версией WebView2 из `--check-env`, кнопка выбора остаётся.
3. Полуторачасовой mp3 с говорящими: текст появляется в течение первой минуты,
   «Остановить» в любой фазе останавливает за ≤ 3 с, «Повторить» работает.
4. Рядом с файлом лежит чужой `имя.txt` — он не тронут, результат в `имя.transkrib.txt`.
5. Подкаст с музыкальной заставкой — в тексте нет «Субтитры сделал…» и повторов.
6. Лог содержит версию, бэкенд, потоки; закрыть приложение во время работы, открыть
   снова — мусора нет, окно живое.
7. Выключить Wi-Fi, выбрать turbo в настройках — понятное сообщение, не traceback.

## 5. Порядок выполнения

1. Г1 каркас тестов и фикстур; Г2 диспетчер режимов (`--check-env`, `--smoke`, перенос
   `selftest.log` в `data_dir()`), консольный exe (A6) — без них остальное не проверить.
2. A0 перетаскивание → В2 мост и гонка → A3 воркер диаризации → A7 отмена ffmpeg,
   stdout=None, excepthook, VC++ assert.
3. A1 кириллица (манифест + `ascii_safe_path` + CI-шаг) → A5 Mark-of-the-Web → A2
   WebView2 → A4 CUDA auto и диагностика → A6 ЧИТАЙ.txt, версия, иконка, упаковка.
4. Б5 тексты ошибок и `-map 0:a:0` → В1 автосохранение → Б1 фильтр → Б4 потоки → В3
   temp и лог → остальное из A7.
5. Б2 VAD на MLX → Б3 резка fw и память → Б6 тихие, дорожки, сон → В4 очередь.
6. В5, В6 — по остатку.
7. Г4 — ручная приёмка, отчёт владельцу.

Каждый пункт — коммит и зелёный CI. Оценка: п.1–2 один день, п.3 один день, п.4 один
день, п.5 один день, приёмка полдня.

## 6. Сдача

- Тег `v0.2.0`, четыре архива в Releases, `ЧИТАЙ.txt` внутри, README обновлён.
- Отчёт: таблица задач со статусом, ссылки на коммиты, результаты Г4 по машинам, список
  известных ограничений для пользователей (перебивающие друг друга голоса, музыка,
  очень тихие записи, NVIDIA без свежего драйвера).
