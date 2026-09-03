"""Transkrib — локальная расшифровка аудио и видео. Без облака, без токенов."""
import os

# Единственное место с версией: её читают build/transkrib.spec (ресурс версии exe),
# build/fetch_model.py (variant.json), заголовок окна и --check-env. CI кладёт
# готовую строку в TRANSKRIB_VERSION перед сборкой — при теге vX.Y.Z это сам тег,
# иначе база ниже плюс короткий хеш коммита (см. .github/workflows/build.yml).
# Без переменной (сборка из исходников на своей машине) используется база как есть.
__version__ = os.environ.get("TRANSKRIB_VERSION") or "0.2.0"
