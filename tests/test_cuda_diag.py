"""Тесты app/engine.py::_cuda_diag (A4) и ступенчатого отката видеокарты в
transcribe_file: cuda/auto → cuda/int8_float32 → cpu.

Живой NVIDIA-видеокарты у разработчиков нет — CUDA-путь на настоящем железе
проверят по логу первые пользователи (решение владельца, см.
docs/TZ-release-hardening.md, A4). Здесь можно и нужно проверить только то,
что не требует GPU: диагностика не падает и быстро отрабатывает на машине без
видеокарты, а цепочка отката переключает backend/compute_type в правильном
порядке при подставных исключениях.
"""
import time

from app import engine


# ── _cuda_diag(): без видеокарты, без исключений, быстро ───────────────────

def test_cuda_diag_no_gpu_returns_dict_with_zero_devices():
    info = engine._cuda_diag()
    assert isinstance(info, dict)
    assert info["cuda_device_count"] == 0
    assert info["compute_types"] == []


def test_cuda_diag_is_fast_without_gpu():
    # На машине без NVIDIA не должно быть ожидания nvidia-smi/DLL с таймаутом —
    # тут либо нечего проверять (не-Windows), либо всё падает мгновенно на
    # отсутствии driver/DLL.
    t0 = time.monotonic()
    engine._cuda_diag()
    assert time.monotonic() - t0 < 1.0


def test_cuda_diag_never_raises_even_if_ctranslate2_broken(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "ctranslate2":
            raise RuntimeError("подставная поломка ctranslate2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    info = engine._cuda_diag()  # не должно бросать
    assert "ошибка" in str(info["ctranslate2_version"])


# ── ступенчатый откат: cuda/auto → cuda/int8_float32 → cpu ─────────────────
# Настоящей видеокарты нет: подделываем detect_backend/load_model/_transcribe_fw,
# чтобы «cuda» шаги отрабатывали с исключением, а «cpu» — успешно, и смотрим,
# в каком порядке и с каким compute_type transcribe_file зовёт load_model.

class TestComputeTypeFallback:
    def _patch_common(self, monkeypatch, calls):
        monkeypatch.setattr(engine, "detect_backend", lambda prefer_gpu=True: "cuda")
        monkeypatch.setattr(engine, "resolve_model", lambda requested, backend: "tiny")
        monkeypatch.setattr(engine, "unload_models", lambda: None)
        monkeypatch.setattr(engine, "memory_available_gb", lambda: None)
        monkeypatch.setattr(engine.media, "check_readable", lambda src: None)
        monkeypatch.setattr(
            engine.media, "extract_wav",
            lambda src, wav, on_progress=None, cancel=None: (1.0, 1))
        monkeypatch.setattr(
            engine.media, "speech_regions",
            lambda wav, total_sec, on_progress=None, cancel=None: [(0.0, total_sec)])

        def fake_load_model(model_key, backend, emit, cancel=None, compute_type=None):
            calls.append(("load", backend, compute_type))
            return object()

        monkeypatch.setattr(engine, "load_model", fake_load_model)

    def test_auto_fails_then_int8_float32_fails_then_falls_back_to_cpu(self, monkeypatch):
        calls = []
        self._patch_common(monkeypatch, calls)

        def fake_fw(model, wav, language, backend, emit, cancel, total_sec, regions):
            calls.append(("transcribe", backend))
            if backend == "cuda":
                raise RuntimeError("CUDA out of memory на первом сегменте")

        monkeypatch.setattr(engine, "_transcribe_fw", fake_fw)

        events = []
        engine.transcribe_file("fake.wav", "tiny", "ru", events.append)

        load_calls = [c for c in calls if c[0] == "load"]
        assert load_calls == [
            ("load", "cuda", "auto"),
            ("load", "cuda", "int8_float32"),
            ("load", "cpu", None),
        ]
        done = [e for e in events if e["type"] == "done"]
        assert done and done[0]["backend"] == "cpu"

        # Текст для карточки — по формулировке ТЗ, а не голый traceback.
        fallback_msgs = [e["msg"] for e in events if e.get("stage") == "fallback"]
        assert any("Видеокарта NVIDIA не сработала, считаю на процессоре" in m
                   for m in fallback_msgs)
        assert any("Обновите драйвер NVIDIA" in m for m in fallback_msgs)

    def test_auto_succeeds_no_fallback_needed(self, monkeypatch):
        calls = []
        self._patch_common(monkeypatch, calls)
        monkeypatch.setattr(
            engine, "_transcribe_fw",
            lambda *a, **k: calls.append(("transcribe", "cuda")))

        events = []
        engine.transcribe_file("fake.wav", "tiny", "ru", events.append)

        assert [c for c in calls if c[0] == "load"] == [("load", "cuda", "auto")]
        done = [e for e in events if e["type"] == "done"]
        assert done and done[0]["backend"] == "cuda"
        assert not [e for e in events if e.get("stage") == "fallback"]

    def test_int8_float32_succeeds_after_auto_fails(self, monkeypatch):
        calls = []
        self._patch_common(monkeypatch, calls)

        def fake_fw(model, wav, language, backend, emit, cancel, total_sec, regions):
            calls.append(("transcribe", backend))
            # Первый вызов (auto) валится, второй (int8_float32) — успех.
            if len([c for c in calls if c[0] == "transcribe"]) == 1:
                raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(engine, "_transcribe_fw", fake_fw)

        events = []
        engine.transcribe_file("fake.wav", "tiny", "ru", events.append)

        load_calls = [c for c in calls if c[0] == "load"]
        assert load_calls == [("load", "cuda", "auto"), ("load", "cuda", "int8_float32")]
        done = [e for e in events if e["type"] == "done"]
        assert done and done[0]["backend"] == "cuda"
