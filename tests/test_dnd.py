"""Перетаскивание файлов: восстановление пути, когда библиотека окна его потеряла.

Библиотека сравнивает имя файла из события с именем, которое собрала при
перетаскивании, дословно. На macOS кириллица приходит в разложенной форме, а из
браузера — в собранной, строки не совпадают, и путь до нас не доходит. Здесь
проверяется запасное сопоставление (app/main.py: Api._recover_dropped_paths).
"""
import os
import unicodedata

import pytest

from app.main import Api


@pytest.fixture
def api(monkeypatch):
    a = Api()

    store = []

    def collected(pairs):
        store[:] = list(pairs)

        def fake(consume=False):
            out = list(store)
            if consume:
                store[:] = []
            return out

        monkeypatch.setattr(Api, "_window_dnd_paths", staticmethod(fake))

    a.store = store

    a.collected = collected
    return a


def test_имена_совпадают_дословно(api):
    api.collected([("a.mp3", "/tmp/a.mp3"), ("b.wav", "/tmp/b.wav")])
    files = [{"name": "b.wav"}, {"name": "a.mp3"}]
    assert api._recover_dropped_paths(files) == ["/tmp/b.wav", "/tmp/a.mp3"]


def test_разная_форма_записи_юникода(api):
    имя = "Рисерч катрин.mp3"
    api.collected([(unicodedata.normalize("NFD", имя), "/tmp/" + имя)])
    assert api._recover_dropped_paths([{"name": unicodedata.normalize("NFC", имя)}]) == ["/tmp/" + имя]


def test_имя_в_процентах(api):
    api.collected([("%D0%B7%D0%B2%D1%83%D0%BA.mp3", "/tmp/звук.mp3")])
    assert api._recover_dropped_paths([{"name": "звук.mp3"}]) == ["/tmp/звук.mp3"]


def test_регистр_не_мешает(api):
    api.collected([("Запись.MP3", "/tmp/Запись.MP3")])
    assert api._recover_dropped_paths([{"name": "запись.mp3"}]) == ["/tmp/Запись.MP3"]


def test_имена_не_сошлись_но_число_совпало(api):
    # Последняя надежда: событие и список — из одного перетаскивания, значит порядок тот же.
    api.collected([("что-то.mp3", "/tmp/что-то.mp3")])
    assert api._recover_dropped_paths([{"name": "совсем другое"}]) == ["/tmp/что-то.mp3"]


def test_путей_меньше_чем_файлов(api):
    # Угадывать нельзя: возьмём не тот файл — человек молча получит чужую расшифровку.
    api.collected([("a.mp3", "/tmp/a.mp3")])
    assert api._recover_dropped_paths([{"name": "x"}, {"name": "y"}]) == []


def test_остатки_прошлого_перетаскивания_не_подставляются(api):
    """Библиотека окна не удаляет из своего списка то, что не сумела сопоставить,
    и он копится. Берём только хвост длиной в число перетащенных файлов."""
    api.collected([("старое.mp3", "/tmp/старое.mp3"), ("новое.mp3", "/tmp/новое.mp3")])
    assert api._recover_dropped_paths([{"name": "не сошлось"}]) == ["/tmp/новое.mp3"]


def test_список_очищается_после_разбора(api):
    api.collected([("a.mp3", "/tmp/a.mp3")])
    assert api._recover_dropped_paths([{"name": "a.mp3"}]) == ["/tmp/a.mp3"]
    assert api.store == [], "разобранное и устаревшее должно уйти из списка"
    assert api._recover_dropped_paths([{"name": "a.mp3"}]) == []


def test_окно_ничего_не_собрало(api):
    api.collected([])
    assert api._recover_dropped_paths([{"name": "a.mp3"}]) == []


def test_два_одинаковых_имени_из_разных_папок(api):
    api.collected([("a.mp3", "/one/a.mp3"), ("a.mp3", "/two/a.mp3")])
    assert api._recover_dropped_paths([{"name": "a.mp3"}, {"name": "a.mp3"}]) == ["/one/a.mp3", "/two/a.mp3"]


def test_в_мосте_нет_открытых_полей():
    """Перед открытием окна pywebview рекурсивно обходит объект-мост, собирая
    методы для JS, и лезет внутрь любого поля без подчёркивания. Когда там лежало
    окно, обход на Windows уходил в дерево .NET-контрола: тысячи ошибок COM,
    предел рекурсии и приложение, которое потом не закрывалось. Любое новое поле
    без подчёркивания вернёт эту беду, поэтому проверяем состав явно."""
    a = Api()
    открытые = [n for n in dir(a) if not n.startswith("_") and not callable(getattr(a, n))]
    assert открытые == [], f"поля без подчёркивания в мосте: {открытые}"


def test_папка_разворачивается_в_записи(tmp_path):
    """Перетащить папку с записями — обычный способ, а не ошибка."""
    (tmp_path / "запись1.mp3").write_bytes(b"x")
    (tmp_path / "заметки.txt").write_bytes(b"x")
    вложенная = tmp_path / "вчера"
    вложенная.mkdir()
    (вложенная / "запись2.wav").write_bytes(b"x")
    (tmp_path / ".служебная").mkdir()
    (tmp_path / ".служебная" / "запись3.mp3").write_bytes(b"x")

    пути, пропущено = Api._expand_dropped([str(tmp_path)])
    assert [os.path.basename(p) for p in пути] == ["запись1.mp3", "запись2.wav"]
    assert пропущено == 0, "файлы внутри папки не считаем пропущенными, их никто не выбирал"


def test_не_медиа_файл_считается_пропущенным(tmp_path):
    док = tmp_path / "договор.pdf"
    док.write_bytes(b"x")
    звук = tmp_path / "речь.m4a"
    звук.write_bytes(b"x")
    пути, пропущено = Api._expand_dropped([str(док), str(звук)])
    assert пути == [str(звук)]
    assert пропущено == 1


def test_из_папки_берётся_не_больше_предела(tmp_path, monkeypatch):
    monkeypatch.setattr(Api, "_DROP_LIMIT", 3)
    for i in range(10):
        (tmp_path / f"{i}.mp3").write_bytes(b"x")
    пути, _ = Api._expand_dropped([str(tmp_path)])
    assert len(пути) == 3
