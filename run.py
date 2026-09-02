"""Точка входа для PyInstaller и для `python run.py`."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from app.main import main
    main()
