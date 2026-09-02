"""Тесты Api.autosave (В1): не затирать чужие файлы, реестр «наших» файлов,
фолбэк на другой каталог при недоступности исходной папки, понятная ошибка
про нехватку места. Настоящий объект Api, окно не создаём (window не нужен —
autosave его не трогает)."""
import errno
import hashlib
import os

import pytest

from app import engine, main


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Изолируем данные приложения (реестр, settings.json) и «Документы» во
    временную папку — реальный домашний каталог пользователя не трогаем."""
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    docs_dir = tmp_path / "Documents" / "Transkrib"
    monkeypatch.setattr(engine, "data_dir", lambda: str(data_dir))
    monkeypatch.setattr(main, "_documents_dir", lambda: str(docs_dir))
    a = main.Api()
    a.window = None
    return a


def _segments():
    return [dict(start=0.0, end=1.0, text="Привет")]


# ── чужой файл не трогаем, наш пишем под другим именем ──────────────────────

def test_foreign_file_untouched_result_gets_transkrib_suffix(api, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "video.mp4"
    video.write_bytes(b"fake video")
    foreign = src_dir / "video.txt"
    foreign.write_text("субтитры скачаны из интернета, не трогать", encoding="utf-8")
    before = _hash(foreign)

    res = api.autosave(str(video), _segments(), ["txt"])

    assert foreign.exists()
    assert _hash(foreign) == before, "чужой файл не должен измениться"
    assert res["fallback"] is None
    assert res["written"] == [str(src_dir / "video.transkrib.txt")]
    assert (src_dir / "video.transkrib.txt").exists()


def test_repeat_autosave_overwrites_own_file_suffix_does_not_grow(api, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "video.mp4"
    video.write_bytes(b"fake video")
    (src_dir / "video.txt").write_text("чужое", encoding="utf-8")

    res1 = api.autosave(str(video), _segments(), ["txt"])
    out = src_dir / "video.transkrib.txt"
    assert res1["written"] == [str(out)]
    mtime1 = out.stat().st_mtime

    res2 = api.autosave(str(video), _segments(), ["txt"])
    assert res2["written"] == [str(out)], "суффикс не должен расти при повторном сохранении"
    # файл действительно перезаписан (реестр признал его «нашим»)
    assert out.exists()
    assert out.stat().st_mtime >= mtime1


def test_own_file_edited_by_user_gets_new_suffix(api, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "video.mp4"
    video.write_bytes(b"fake video")
    (src_dir / "video.txt").write_text("чужое", encoding="utf-8")

    res1 = api.autosave(str(video), _segments(), ["txt"])
    out = src_dir / "video.transkrib.txt"
    assert res1["written"] == [str(out)]

    # человек открыл наш файл и что-то в нём поправил — размер/mtime меняются
    out.write_text("человек дописал сюда что-то своё", encoding="utf-8")

    res2 = api.autosave(str(video), _segments(), ["txt"])
    out2 = src_dir / "video.transkrib-2.txt"
    assert res2["written"] == [str(out2)]
    assert out2.exists()
    assert out.read_text(encoding="utf-8") == "человек дописал сюда что-то своё"


# ── фолбэк при недоступности каталога ────────────────────────────────────────

def test_permission_error_falls_back_to_documents(api, tmp_path, monkeypatch):
    src_dir = tmp_path / "readonly_src"
    src_dir.mkdir()
    video = src_dir / "video.mp4"
    video.write_bytes(b"fake video")

    real_open = open

    def fake_open(path, *args, **kwargs):
        if str(path).startswith(str(src_dir)) and "w" in (args[0] if args else kwargs.get("mode", "")):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    res = api.autosave(str(video), _segments(), ["txt"])

    assert res["fallback"] is not None
    fb_dir = res["fallback"]["dir"]
    assert fb_dir == str(main._documents_dir())
    assert res["written"] == [os.path.join(fb_dir, "video.txt")]
    assert os.path.exists(res["written"][0])


def test_permission_error_on_documents_too_falls_back_to_data_dir(api, tmp_path, monkeypatch):
    src_dir = tmp_path / "readonly_src"
    src_dir.mkdir()
    video = src_dir / "video.mp4"
    video.write_bytes(b"fake video")
    docs_dir = tmp_path / "Documents" / "Transkrib"

    real_open = open

    def fake_open(path, *args, **kwargs):
        p = str(path)
        mode = args[0] if args else kwargs.get("mode", "")
        if "w" in mode and (p.startswith(str(src_dir)) or p.startswith(str(docs_dir))):
            raise PermissionError(errno.EACCES, "Permission denied", p)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    res = api.autosave(str(video), _segments(), ["txt"])

    output_dir = os.path.join(engine.data_dir(), "output")
    assert res["fallback"]["dir"] == output_dir
    assert res["written"] == [os.path.join(output_dir, "video.txt")]
    assert os.path.exists(res["written"][0])


# ── нехватка места — понятная ошибка, а не тихий фолбэк ─────────────────────

def test_no_space_left_raises_user_error(api, tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "video.mp4"
    video.write_bytes(b"fake video")

    real_open = open

    def fake_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "")
        if "w" in mode:
            raise OSError(errno.ENOSPC, "No space left on device", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    with pytest.raises(engine.UserError, match="места"):
        api.autosave(str(video), _segments(), ["txt"])
